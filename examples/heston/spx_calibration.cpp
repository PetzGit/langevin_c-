// Calibrates Heston to a real SPX option-chain snapshot produced by
// data_loader/data_loader.py (dat.csv + meta.txt), using the corrected
// log-space objective, chain_num=50/run_num=1000/r_rate=20 Optimiser, then a
// box-constrained GD polish.
//
// Run from data_loader/ (so dat.csv/meta.txt resolve from cwd):
//   cd data_loader && ../build/examples/heston_spx_calibration
#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/heston_objective.hpp"
#include "cldiff/heston/corrected_log_space_heston_objective.hpp"
#include "cldiff/heston/feller.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/optimiser.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <fstream>

double runGradientDescent(const Objective& obj, const Polyhedron& polyhedron, Eigen::VectorXd& y,
                           int max_iters, int max_backtracks, bool& out_stuck) {
  double lr = 0.1;
  double current_loss = obj.value(y);
  out_stuck = false;
  for (int it = 1; it <= max_iters; it++) {
    Eigen::VectorXd grad = obj.gradient(y);
    double trial_lr = lr;
    Eigen::VectorXd y_new;
    double new_loss = current_loss;
    bool improved = false;
    for (int bt = 0; bt < max_backtracks; bt++) {
      y_new = y - trial_lr * grad;
      if (polyhedron.contains(y_new)) {
        new_loss = obj.value(y_new);
        if (new_loss < current_loss) { improved = true; break; }
      }
      trial_lr *= 0.5;
    }
    if (!improved) { out_stuck = true; break; }
    y = y_new;
    current_loss = new_loss;
    lr = trial_lr * 1.5;
  }
  return current_loss;
}

int main() {
  std::ifstream meta("meta.txt");
  std::string key;
  double S0 = 0, r = 0, q = 0;
  while (meta >> key) {
    if (key == "S0") meta >> S0;
    else if (key == "r") meta >> r;
    else if (key == "q") meta >> q;
  }
  MarketData market(S0, r, q);
  market.loadFromCsv("dat.csv");
  double n_quotes = 0;
  for (auto& [T, quotes] : market.quotesByMaturity()) n_quotes += quotes.size();

  const int term_num = 128;
  const double L = 10.0, lambda_feller = 1.0;
  HestonObjective objective(market, term_num, L, lambda_feller);
  CorrectedLogSpaceHestonObjective log_objective(objective);

  Eigen::MatrixXd A(10, 5);
  Eigen::VectorXd b(10);
  A.setZero();
  auto setBounds = [&](int row, int col, double lo, double hi) {
    A(row, col) = -1.0;      b(row) = -lo;
    A(row + 1, col) = 1.0;   b(row + 1) = hi;
  };
  setBounds(0, 0, -5.0, 3.0);
  setBounds(2, 1, -5.0, 0.4);
  setBounds(4, 2, -5.0, 1.5);
  setBounds(6, 3, -0.999, 0.999);
  setBounds(8, 4, -5.0, 0.4);
  Polyhedron polyhedron(A, b);

  const int chain_num = 50, run_num = 1000, r_rate = 20;
  const double h_opt = 1e-2, eps_opt = 1e-6;
  const int polish_iters = 500, polish_backtracks = 30;

  Eigen::VectorXd start(5);
  start << 2.0, 0.04, 0.5, -0.7, 0.04;
  Rng rng(4002);
  Eigen::VectorXd y0 = CorrectedLogSpaceHestonObjective::toLogSpace(start);
  Eigen::MatrixXd x_0(5, chain_num);
  for (int c = 0; c < chain_num; c++) {
    Eigen::VectorXd col = y0;
    for (int j = 0; j < 5; j++) col(j) += 0.3 * rng.normal();
    if (!polyhedron.contains(col)) col = y0;
    x_0.col(c) = col;
  }
  Optimiser optimiser(log_objective, polyhedron, rng, x_0, r_rate, chain_num, run_num, h_opt, eps_opt);
  ProposalResult result = optimiser.run();

  Eigen::VectorXd y = result.best_particle;
  bool stuck = false;
  double polish_loss = runGradientDescent(log_objective, polyhedron, y, polish_iters, polish_backtracks, stuck);
  Eigen::VectorXd phys = CorrectedLogSpaceHestonObjective::toPhysical(y);

  std::printf("S0=%.2f r=%.4f q=%.4f  quotes=%d\n", S0, r, q, (int)n_quotes);
  std::printf("opt_loss=%.4e polish_loss=%.6e RMSE=%.4f vol pts  polish_stuck=%s Feller=%s\n",
              result.result, polish_loss, std::sqrt(polish_loss / n_quotes) * 100.0,
              stuck ? "yes" : "no", satisfiesFeller(phys) ? "yes" : "no");
  const char* names[5] = {"kappa", "theta", "xi", "rho", "v0"};
  for (int i = 0; i < 5; i++) std::printf("  %-8s = %.4f\n", names[i], phys(i));
  return 0;
}
