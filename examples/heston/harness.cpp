// examples/heston/harness.cpp
//
// Heston calibration harness: measures whether the annealed Optimiser escapes
// local minima that plain gradient descent cannot, and whether the resulting
// fit is a *useful* Heston model rather than just a good fit to the points it
// trained on.
//
// Setup. Synthetic market generated from true params
// (kappa, theta, xi, rho, v0) = (2.0, 0.04, 0.30, -0.70, 0.04). Training quotes
// use only the middle strikes K in {90,100,110} -- the "liquid near-the-money"
// case -- across 5 maturities. The wings K in {80,120} are held out and scored
// in implied vol, so a fit that merely interpolates its own training points is
// visibly distinguishable from one that recovers the dynamics.
//
// Three arms per starting point:
//   1. box-constrained GD straight from the raw start   (the local baseline)
//   2. annealed Optimiser                               (exploration)
//   3. Optimiser result -> box-constrained GD polish    (the real test)
// If (3) lands far below (1), the Optimiser reached a basin GD could not.
//
// Measured 2026-09-22/23, start B, chain_num=50, seed 4002, corrected wrapper:
//
//   run_num/r_rate   pre-polish   post-polish   OOS iv rmse
//   1000 / 20        1.4369e-04   8.5101e-06    2.86e-03     <- best
//   1000 / 50        7.6195e-04   2.2963e-04    8.24e-03     (polish stuck)
//   3000 / 20        3.7166e-05   3.0725e-05    3.58e-03
//   3000 / 50        5.3878e-04   9.1510e-06    3.56e-03
//
//   GD only, from the raw start:  1.1696e-01, stuck, rho stays at +0.798.
//
// r_rate=20 beats 50 at both lengths. Single seed per cell, so treat gaps under
// ~2x as noise; the GD-vs-polish gap (four orders of magnitude) is not noise.
//
// Two findings about Optimiser internals that this harness produced, recorded
// here because they are not visible from the source alone:
//   - resample()'s weights exp(-(f_i - f_min)) carry no scale. With f ~ 1e-3
//     every exponent is ~0, so ESS sits at ~50/50 and the draw is effectively
//     uniform: selection does nothing and the elitist carry-over of column 0 is
//     what actually works, at frequency 1/r_rate. Dividing the exponent by the
//     population std *was* tried -- ESS fell to ~33, spread collapsed 1.0->0.16,
//     and the fit got worse (post-polish 8.51e-06 -> 1.29e-04, rho pinned at
//     -0.955). The diversity the flat weights preserve is load-bearing for
//     basin escape. Do not "fix" this without a different framing.
//   - run() returns the final population's best, not the best seen during the
//     run. A running-best tracker hooked into resample() is free (f_values is
//     already computed there) and monotone; it was worth 1.57x at r_rate=50 and
//     nothing at r_rate=20. Not applied to src/optimiser.cpp.
//
// Usage: harness [run_num] [r_rate]      defaults 1000 20

#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/heston_objective.hpp"
#include "cldiff/heston/corrected_log_space_heston_objective.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/optimiser.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Backtracking gradient descent that rejects any trial step leaving the
// polyhedron, the same way Optimiser::step() rejects an out-of-box proposal.
// Without this the polish walks straight past hard parameter bounds -- rho
// outside [-1,1] being the one that actually bit. Returns the final loss and
// writes the final point into y; out_stuck reports that the line search found
// no admissible improving step, as opposed to hitting max_iters.
double runGradientDescent(const Objective& obj, const Polyhedron& polyhedron,
                          Eigen::VectorXd& y, int max_iters, int max_backtracks,
                          bool& out_stuck) {
  double lr = 0.1;
  double current_loss = obj.value(y);
  out_stuck = false;

  for (int iter = 1; iter <= max_iters; iter++) {
    Eigen::VectorXd grad = obj.gradient(y);
    double trial_lr = lr;
    Eigen::VectorXd y_new;
    double new_loss = current_loss;
    bool improved = false;

    for (int bt = 0; bt < max_backtracks; bt++) {
      y_new = y - trial_lr * grad;
      if (polyhedron.contains(y_new)) {
        new_loss = obj.value(y_new);
        if (new_loss < current_loss) {
          improved = true;
          break;
        }
      }
      trial_lr *= 0.5;
    }

    if (!improved) {
      out_stuck = true;
      break;
    }

    y = y_new;
    current_loss = new_loss;
    lr = trial_lr * 1.5;
  }

  return current_loss;
}

int main(int argc, char** argv) {
  const double S0 = 100.0;
  const double r  = 0.02;
  const int    term_num = 128;
  const double L = 10.0;
  const double lambda_feller = 1.0;

  const int run_num = (argc > 1) ? std::atoi(argv[1]) : 1000;
  const int r_rate  = (argc > 2) ? std::atoi(argv[2]) : 20;
  const int chain_num = 50;
  const double h_opt = 1e-2;
  const double eps_opt = 1e-6;
  const int polish_iters = 500;
  const int polish_backtracks = 30;

  Eigen::VectorXd true_params(5);
  true_params << 2.0, 0.04, 0.30, -0.70, 0.04;
  CharacteristicHeston phi_true(true_params);

  const std::vector<double> maturities    = {0.25, 0.5, 1.0, 1.5, 2.0};
  const std::vector<double> train_strikes = {90.0, 100.0, 110.0};
  const std::vector<double> test_strikes  = {80.0, 120.0};

  MarketData market_train(S0, r);
  for (double T : maturities) {
    Cumulants cum = cumulants(phi_true, T);
    Range range = make_range(cum, L);
    for (double K : train_strikes) {
      market_train.addQuote(T, K, cos_pricer(phi_true, range, K, T, r, term_num, S0));
    }
  }

  HestonObjective objective(market_train, term_num, L, lambda_feller);
  CorrectedLogSpaceHestonObjective log_objective(objective);

  // Held-out wings, as true implied vols computed directly rather than through
  // MarketData, so the scoring path is independent of the training path.
  struct TestPoint { double T, K, true_iv; };
  std::vector<TestPoint> test_points;
  for (double T : maturities) {
    Cumulants cum = cumulants(phi_true, T);
    Range range = make_range(cum, L);
    for (double K : test_strikes) {
      double price = cos_pricer(phi_true, range, K, T, r, term_num, S0);
      BlackSchole bs(S0, K, T, r);
      test_points.push_back({T, K, bs.impliedVol(price)});
    }
  }

  auto evalOOS = [&](const Eigen::VectorXd& params, double& rmse, double& maxerr) {
    CharacteristicHeston phi(params);
    rmse = 0.0;
    maxerr = 0.0;
    for (const auto& tp : test_points) {
      Cumulants cum = cumulants(phi, tp.T);
      Range range = make_range(cum, L);
      double price = cos_pricer(phi, range, tp.K, tp.T, r, term_num, S0);
      BlackSchole bs(S0, tp.K, tp.T, r);
      double iv = bs.impliedVol(price);
      // A non-finite IV means the fitted params priced outside the arbitrage
      // bounds; score it as a full 100 vol points rather than dropping it.
      double err = std::isfinite(iv) ? std::abs(iv - tp.true_iv) : 1.0;
      rmse += err * err;
      maxerr = std::max(maxerr, err);
    }
    rmse = std::sqrt(rmse / test_points.size());
  };

  // Box, in log-space.
  Eigen::MatrixXd A(10, 5);
  Eigen::VectorXd b(10);
  A.setZero();
  auto setBounds = [&](int row, int col, double lo, double hi) {
    A(row, col) = -1.0;      b(row) = -lo;
    A(row + 1, col) = 1.0;   b(row + 1) = hi;
  };
  setBounds(0, 0, -5.0, 3.0);      // log(kappa): ~0.0067 to ~20.1
  setBounds(2, 1, -5.0, 0.4);      // log(theta): ~0.0067 to ~1.49
  setBounds(4, 2, -5.0, 1.5);      // log(xi):    ~0.0067 to ~4.48
  setBounds(6, 3, -0.999, 0.999);  // rho
  setBounds(8, 4, -5.0, 0.4);      // log(v0):    ~0.0067 to ~1.49
  Polyhedron polyhedron(A, b);

  struct Start { const char* name; Eigen::VectorXd p; std::uint64_t seed; };
  std::vector<Start> starts;
  auto addStart = [&](const char* name, double k, double t, double x, double rh,
                      double v, std::uint64_t seed) {
    Eigen::VectorXd p(5);
    p << k, t, x, rh, v;
    starts.push_back({name, p, seed});
  };
  addStart("A", 4.0, 0.10, 0.10,  0.10, 0.10, 4001);
  // B is the pathological one: wrong-sign rho against a true -0.70, plus high
  // kappa/xi/v0 and low theta. GD cannot cross the rho sign barrier from here.
  addStart("B", 6.0, 0.02, 1.20,  0.80, 0.30, 4002);

  std::printf("True params: kappa=2.00 theta=0.0400 xi=0.300 rho=-0.700 v0=0.0400\n");
  std::printf("chain_num=%d run_num=%d r_rate=%d h=%g eps=%g\n",
              chain_num, run_num, r_rate, h_opt, eps_opt);
  std::printf("train strikes {90,100,110}, held-out wings {80,120}\n\n");

  for (const auto& s : starts) {
    std::printf("=== start %s: kappa=%.2f theta=%.3f xi=%.2f rho=%.2f v0=%.3f ===\n",
                s.name, s.p(0), s.p(1), s.p(2), s.p(3), s.p(4));

    auto report = [&](const char* label, const Eigen::VectorXd& y, double loss,
                      bool stuck, double secs) {
      Eigen::VectorXd phys = CorrectedLogSpaceHestonObjective::toPhysical(y);
      double rmse, maxerr;
      evalOOS(phys, rmse, maxerr);
      std::printf("  %-22s loss=%-12.4e stuck=%-4s time=%6.1fs\n",
                  label, loss, stuck ? "yes" : "no", secs);
      std::printf("  %-22s kappa=%.3f theta=%.4f xi=%.3f rho=%.3f v0=%.4f  "
                  "oos_iv_rmse=%.4e oos_iv_max=%.4e\n",
                  "", phys(0), phys(1), phys(2), phys(3), phys(4), rmse, maxerr);
      std::fflush(stdout);
    };

    // --- 1. box-constrained GD from the raw start ---
    Eigen::VectorXd y_gd = CorrectedLogSpaceHestonObjective::toLogSpace(s.p);
    bool gd_stuck = false;
    auto t_gd = std::chrono::steady_clock::now();
    double gd_loss = runGradientDescent(log_objective, polyhedron, y_gd,
                                        polish_iters, polish_backtracks, gd_stuck);
    report("GD only", y_gd, gd_loss, gd_stuck,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gd).count());

    // --- 2. annealed Optimiser ---
    Rng rng(s.seed);
    Eigen::VectorXd y0 = CorrectedLogSpaceHestonObjective::toLogSpace(s.p);
    Eigen::MatrixXd x_0(5, chain_num);
    for (int c = 0; c < chain_num; c++) {
      Eigen::VectorXd col = y0;
      for (int j = 0; j < 5; j++) {
        col(j) += 0.3 * rng.normal();
      }
      if (!polyhedron.contains(col)) {
        col = y0;
      }
      x_0.col(c) = col;
    }

    auto t_opt = std::chrono::steady_clock::now();
    Optimiser optimiser(log_objective, polyhedron, rng, x_0, r_rate, chain_num,
                        run_num, h_opt, eps_opt);
    ProposalResult result = optimiser.run();
    double opt_secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_opt).count();
    report("Optimiser", result.best_particle, result.result, false, opt_secs);

    // --- 3. box-constrained GD polish from the Optimiser's result ---
    Eigen::VectorXd y_polish = result.best_particle;
    bool polish_stuck = false;
    auto t_pol = std::chrono::steady_clock::now();
    double polish_loss = runGradientDescent(log_objective, polyhedron, y_polish,
                                            polish_iters, polish_backtracks, polish_stuck);
    report("Optimiser + polish", y_polish, polish_loss, polish_stuck,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pol).count());
    std::printf("\n");
  }

  double rmse_true, maxerr_true;
  evalOOS(true_params, rmse_true, maxerr_true);
  std::printf("(reference) true params OOS iv_rmse=%.4e iv_max=%.4e "
              "(expect ~0, COS/Newton noise only)\n", rmse_true, maxerr_true);
  return 0;
}
