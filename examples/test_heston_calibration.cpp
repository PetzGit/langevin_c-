// examples/test_multistart_comparison.cpp
//
// Compares plain gradient descent against the annealed Optimiser across
// several diverse starting points, on the same synthetic market. The point:
// show whether the annealed method is actually more robust to bad
// initialization than a local method -- which is the whole justification
// for building it.
//
// Added: a GD "polish" pass run from the Optimiser's own result. This is
// the actual test of "did the Optimiser find a better basin than GD could
// reach on its own" -- if polishing drives the loss down near the true
// minimum, the Optimiser did real global work; if the polish loss stays
// near result.result, the Optimiser landed in the same local minimum GD
// would have found anyway, and the two methods aren't meaningfully
// different on that starting point.
//
// Also added: a deliberately far starting point (index 4), further from
// true_params than any of the originals, to stress-test basin escape
// harder than the existing set.

#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/heston_objective.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/optimiser.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <vector>

// Log-space parameterization: kappa, theta, xi and v0 are exp-transformed so
// they stay positive, rho passes through unchanged. The chain-rule scaling
// (dL/dy = dL/dp * p) is divided back out for the exp-transformed dims.
// Without that division the log-space gradient shrinks in proportion to the
// parameter's own value as it approaches zero, even when the true physical
// sensitivity is still large -- confirmed directly: at kappa=0.01, physical
// dL/dkappa was -0.134 but log-space dL/d(log kappa) was only -0.00134, a 100x
// suppression. That produced a real, repeatable failure mode: on a reduced
// (subset) training grid the uncorrected version of this class collapsed to
// the same degenerate corner (kappa pinned at its box floor, rho pinned at its
// box wall) regardless of starting point, and generalized badly out of sample.
// Dividing by p(i) restores a step whose *relative* size tracks the true
// physical sensitivity instead of being conflated with how small p already is,
// and fixes the collapse (kappa/xi/rho all land near truth instead,
// out-of-sample error drops ~5-9x in testing). rho (index 3) isn't
// exp-transformed, so it needs no correction.
class CorrectedLogSpaceHestonObjective : public Objective {
  public:
    explicit CorrectedLogSpaceHestonObjective(const HestonObjective& inner) : inner_(inner) {}

    double value(const Eigen::VectorXd& y) const override {
      return inner_.value(toPhysical(y));
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& y) const override {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(5);
      const double h = 3e-3;
      for (int i = 0; i < 5; i++) {
        Eigen::VectorXd y_plus = y;
        y_plus(i) += h;
        Eigen::VectorXd y_minus = y;
        y_minus(i) -= h;
        grad(i) = (value(y_plus) - value(y_minus)) / (2.0 * h);
      }
      const Eigen::VectorXd p = toPhysical(y);
      for (int i : {0, 1, 2, 4}) {
        grad(i) /= std::max(p(i), 1e-8);
      }
      return grad;
    }

    int dim() const override { return 5; }

    static Eigen::VectorXd toPhysical(const Eigen::VectorXd& y) {
      Eigen::VectorXd p(5);
      p(0) = std::exp(y(0));
      p(1) = std::exp(y(1));
      p(2) = std::exp(y(2));
      p(3) = y(3);
      p(4) = std::exp(y(4));
      return p;
    }

    static Eigen::VectorXd toLogSpace(const Eigen::VectorXd& p) {
      Eigen::VectorXd y(5);
      y(0) = std::log(p(0));
      y(1) = std::log(p(1));
      y(2) = std::log(p(2));
      y(3) = p(3);
      y(4) = std::log(p(4));
      return y;
    }

  private:
    const HestonObjective& inner_;
};

// Backtracking gradient descent, confined to the polyhedron. A trial step
// landing outside the box is rejected and the line search backtracks further,
// the same way Optimiser::step() discards an out-of-box proposal. Without this
// the descent walks straight past hard parameter bounds -- rho leaving [-1,1]
// being the one that actually bit, since the fits it produced were not valid
// Heston models at all. Returns final loss; writes final point into y (in/out).
// out_stuck reports whether it stopped because the line search found no
// admissible improving step (vs hitting max_iters).
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

int main() {
  const double S0 = 100.0;
  const double r  = 0.02;
  const int    term_num = 128;
  const double L = 10.0;
  const double lambda_feller = 1.0;

  Eigen::VectorXd true_params(5);
  true_params << 2.0, 0.04, 0.30, -0.70, 0.04;

  CharacteristicHeston phi_true(true_params);

  struct Point { double T; double K; };
  std::vector<Point> points;
  std::vector<double> maturities = {0.25, 0.5, 1.0, 1.5, 2.0};
  std::vector<double> strikes    = {80.0, 90.0, 100.0, 110.0, 120.0};
  for (double T : maturities) {
    for (double K : strikes) {
      points.push_back({T, K});
    }
  }

  MarketData market(S0, r);
  for (const auto& pt : points) {
    Cumulants cum = cumulants(phi_true, pt.T);
    Range range = make_range(cum, L);
    double price = cos_pricer(phi_true, range, pt.K, pt.T, r, term_num, S0);
    market.addQuote(pt.T, pt.K, price);
  }

  HestonObjective objective(market, term_num, L, lambda_feller);
  CorrectedLogSpaceHestonObjective log_objective(objective);

  // Box, in log-space, shared across all Optimiser runs.
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

  // Diverse starting points, in physical units.
  std::vector<Eigen::VectorXd> starts;
  auto addStart = [&](double k, double t, double x, double rh, double v) {
    Eigen::VectorXd p(5);
    p << k, t, x, rh, v;
    starts.push_back(p);
  };
  addStart(4.0, 0.10, 0.10, 0.10, 0.10);   // the one that worked for plain GD
  addStart(0.5, 0.01, 1.50, -0.90, 0.01);
  addStart(8.0, 0.50, 0.05, 0.50, 0.50);
  addStart(1.0, 0.04, 0.30, -0.70, 0.04);  // near truth, sanity check
  addStart(6.0, 0.02, 1.20, 0.80, 0.30);   // deliberately far: high kappa, low
                                            // theta, high xi, WRONG-SIGN rho
                                            // (true rho is -0.70), high v0 --
                                            // harder than the existing points

  const int chain_num = 5;
  const int run_num = 1000;   // trimmed down per current test run
  const int r_rate = 20;
  const double h_opt = 1e-2;
  const double eps_opt = 1e-6;

  const int polish_iters = 1000;
  const int polish_backtracks = 30;

  std::printf("%-3s %-42s %-13s %-13s %-13s %-9s %-9s\n",
              "#", "start (kappa,theta,xi,rho,v0)",
              "GD loss", "Opt loss", "Opt+Polish", "GD stuck", "Pol stuck");

  for (size_t i = 0; i < starts.size(); i++) {
    // --- Gradient descent, direct from the raw starting point ---
    Eigen::VectorXd y = CorrectedLogSpaceHestonObjective::toLogSpace(starts[i]);
    bool stuck = false;
    double gd_loss = runGradientDescent(log_objective, polyhedron, y, 500, 30, stuck);

    // --- Annealed Optimiser ---
    Rng rng(1000 + static_cast<std::uint64_t>(i));
    Eigen::VectorXd y0 = CorrectedLogSpaceHestonObjective::toLogSpace(starts[i]);
    Eigen::MatrixXd x_0(5, chain_num);
    for (int c = 0; c < chain_num; c++) {
      Eigen::VectorXd col = y0;
      for (int j = 0; j < 5; j++) {
        col(j) += 0.1 * rng.normal();
      }
      if (!polyhedron.contains(col)) {
        col = y0;
      }
      x_0.col(c) = col;
    }
    Optimiser optimiser(log_objective, polyhedron, rng, x_0, r_rate, chain_num, run_num, h_opt, eps_opt);
    ProposalResult result = optimiser.run();

    // --- GD polish, starting from the Optimiser's own result ---
    // This is the actual test: did the Optimiser find a genuinely better
    // basin than GD reaches on its own? If polish loss << Opt loss and
    // approaches the true minimum, yes. If polish loss stays close to
    // Opt loss, the Optimiser converged to the same (possibly local)
    // minimum a local method would have found from there anyway.
    Eigen::VectorXd y_polish = result.best_particle;
    bool polish_stuck = false;
    double polish_loss = runGradientDescent(log_objective, polyhedron, y_polish,
                                             polish_iters, polish_backtracks, polish_stuck);

    std::printf("%-3zu [%.2f, %.3f, %.2f, %.2f, %.3f]  %-13.4e %-13.4e %-13.4e %-9s %-9s\n",
                i, starts[i](0), starts[i](1), starts[i](2), starts[i](3), starts[i](4),
                gd_loss, result.result, polish_loss,
                stuck ? "yes" : "no", polish_stuck ? "yes" : "no");
  }

  return 0;
}
