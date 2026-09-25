// examples/test_heston_objective.cpp
//
// Sanity check for HestonObjective::value/gradient:
// generate a handful of "market" quotes by pricing them under a known
// true Heston parameter set, then check that:
//   - value(true_params) is ~0 (only COS-truncation / Newton-tolerance noise)
//   - value(perturbed_params) > value(true_params)
//   - gradient(true_params) is ~0 (true params sit at the minimum)
//   - gradient(perturbed_params) is non-trivial and points "downhill"

#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/heston_objective.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <cmath>

int main() {
  const double S0 = 100.0;
  const double r  = 0.02;
  const int    term_num = 128;
  const double L = 10.0;
  const double lambda_feller = 1.0;

  // Chosen to satisfy Feller: 2*kappa*theta = 0.16 >= xi^2 = 0.09
  Eigen::VectorXd true_params(5);
  true_params << 2.0,   // kappa
                 0.04,  // theta
                 0.30,  // xi
                -0.70,  // rho
                 0.04;  // v0

  CharacteristicHeston phi_true(true_params);

  // A handful of (T, K) pairs to act as "market" quotes.
  struct Point { double T; double K; };
  std::vector<Point> points = {
    {0.5, 90.0}, {0.5, 100.0}, {0.5, 110.0},
    {1.0, 90.0}, {1.0, 100.0}, {1.0, 110.0},
  };

  MarketData market(S0, r);
  for (const auto& pt : points) {
    Cumulants cum = cumulants(phi_true, pt.T);
    Range range = make_range(cum, L);
    double price = cos_pricer(phi_true, range, pt.K, pt.T, r, term_num, S0);
    market.addQuote(pt.T, pt.K, price);
  }

  HestonObjective objective(market, term_num, L, lambda_feller);

  double loss_at_truth = objective.value(true_params);
  std::printf("value(true_params)       = %.10e  (expect ~0)\n", loss_at_truth);

  Eigen::VectorXd perturbed = true_params;
  perturbed(0) *= 1.5;  // bump kappa by 50%
  double loss_perturbed = objective.value(perturbed);
  std::printf("value(perturbed_params)  = %.10e  (expect > value at truth)\n", loss_perturbed);

  Eigen::VectorXd grad_at_truth = objective.gradient(true_params);
  std::printf("gradient(true_params)    = [%.3e, %.3e, %.3e, %.3e, %.3e]  (expect ~0)\n",
              grad_at_truth(0), grad_at_truth(1), grad_at_truth(2), grad_at_truth(3), grad_at_truth(4));

  Eigen::VectorXd grad_at_perturbed = objective.gradient(perturbed);
  std::printf("gradient(perturbed)      = [%.3e, %.3e, %.3e, %.3e, %.3e]\n",
              grad_at_perturbed(0), grad_at_perturbed(1), grad_at_perturbed(2), grad_at_perturbed(3), grad_at_perturbed(4));

  // One manual gradient-descent step: loss should decrease, confirming the
  // gradient points in a sensible direction (not verifying magnitude/scale,
  // just sign/direction).
  double step = 1e-5;
  Eigen::VectorXd stepped = perturbed - step * grad_at_perturbed;
  double loss_stepped = objective.value(stepped);
  std::printf("value after one grad step = %.10e  (expect < value(perturbed) = %.10e)\n",
              loss_stepped, loss_perturbed);

  return 0;
}
