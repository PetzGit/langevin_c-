#include "cldiff/heston/heston_objective.hpp"
#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"
#include "cldiff/heston/market_data.hpp"
#include <iostream>
HestonObjective::HestonObjective(const MarketData& market_data, int term_num, double L, double lambda_feller)
: market_data_(market_data), term_num_(term_num), L_(L), lambda_feller_(lambda_feller) {}

double HestonObjective::value(const Eigen::VectorXd& params) const {
  CharacteristicHeston phi(params);
  double loss = 0.0;
  const std::map <double, std::vector<Quote>>& quote_map = market_data_.quotesByMaturity();
  double r = market_data_.r();
  double s0 = market_data_.S0();
  for (const auto& [key, quotes] : quote_map) {
    Cumulants cum = cumulants(phi, key);
    Range range = make_range(cum, L_);
    for (const Quote& quote: quotes) {
      double K = quote.K;
      double price = cos_pricer(phi, range, K, key, r, term_num_, s0);
      BlackSchole iv_calc(s0, K, key, r);
      double calc_iv = iv_calc.impliedVol(price);
      if (!std::isfinite(quote.market_iv) || !std::isfinite(calc_iv)) {
        loss += quote.weight;
        continue;
      }
      loss += (calc_iv - quote.market_iv)*(calc_iv - quote.market_iv)*quote.weight;
    }
  }
  return loss + lambda_feller_*std::max(0.0, params(2)*params(2) - 2*params(1)*params(0));
}
//todo implement autodiff (this requires changing type signatures everywhere current imp
//is finite differences, which is numerically unstable somewhat and costly

Eigen::VectorXd HestonObjective::gradient(const Eigen::VectorXd& params) const {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(5);
  double h_base = 2e-3;
  for (int i{0}; i < 5; i++) {
    double h_i = h_base * std::max(1.0, std::abs(params(i)));
    Eigen::VectorXd plus_params = params;
    plus_params(i) += h_i;
    Eigen::VectorXd minus_params = params;
    minus_params(i) -= h_i;
    grad(i) = (value(plus_params) - value(minus_params))/(2*h_i);
  }
  return grad;
}

int HestonObjective::dim() const {
  return 5;
}
