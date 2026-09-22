#include <cldiff/heston/black_scholes.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
double normal_cdf( double x) {
  return 0.5*(1+std::erf(x/std::sqrt(2)));
}

double normal_pdf(double x) {
  static const double inv_sqrt_2pi = 1.0/ std::sqrt(2.0* M_PI);
  return inv_sqrt_2pi * std::exp(-0.5 * x*x);
}

BlackSchole::BlackSchole(double S0, double K, double T, double r)
: S0_(S0), K_(K), T_(T), r_(r) {}

double BlackSchole::price(double sigma) const {
  double d_1 = (std::log(S0_/K_) + (r_ + sigma*sigma*0.5)*T_)/(sigma*std::sqrt(T_));
  double d_2 = d_1 - sigma*std::sqrt(T_);
  return S0_*normal_cdf(d_1) - K_*std::exp(-r_*T_)*normal_cdf(d_2);
}

double BlackSchole::vega (double sigma) const {
  double d_1 = (std::log(S0_/K_) + (r_ + sigma*sigma*0.5)*T_)/(sigma*std::sqrt(T_));
  return S0_*std::sqrt(T_)*normal_pdf(d_1);
}
double BlackSchole::impliedVol(double C_mkt) const {
  const double intrinsic = std::max(S0_ - K_*std::exp(-r_*T_), 0.0);
  const double upper_bound = S0_;
  if (C_mkt < intrinsic || C_mkt > upper_bound || !std::isfinite(C_mkt)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double sigma_lo = 1e-4;
  const double sigma_hi = 5.0;
  double sigma = std::clamp(std::sqrt(2.0*M_PI / T_) * (C_mkt/S0_), sigma_lo, sigma_hi);

  int max = 100;
  int iter_num = 0;
  double tol = 1e-6;
  double eps = 1e-8;
  const double max_step = 0.5;

  double price_val = price(sigma);
  while (std::abs(price_val - C_mkt) >= tol && iter_num < max) {
    double v = vega(sigma);
    double v_safe = (std::abs(v) < eps) ? eps : v;
    double step = std::clamp((price_val - C_mkt)/v_safe, -max_step, max_step);
    sigma = std::clamp(sigma - step, sigma_lo, sigma_hi);
    price_val = price(sigma);
    iter_num += 1;
  }

  if (std::abs(price_val - C_mkt) < tol) return sigma;

  std::cout << C_mkt << " : " << std::abs(price_val - C_mkt)
            << " : sigma=" << sigma << " : newton_failed, trying bisection\n";

  double lo = sigma_lo, hi = sigma_hi;
  double price_lo = price(lo), price_hi = price(hi);
  if ((price_lo - C_mkt) * (price_hi - C_mkt) > 0.0) {
    std::cout << C_mkt << " : unreachable, not bracketed in [" << lo << "," << hi << "]\n";
    return std::numeric_limits<double>::quiet_NaN();
  }
  for (int i = 0; i < 100; ++i) {
    double mid = 0.5 * (lo + hi);
    double price_mid = price(mid);
    if (std::abs(price_mid - C_mkt) < tol) return mid;
    if ((price_mid - C_mkt) * (price_lo - C_mkt) < 0.0) { hi = mid; }
    else { lo = mid; price_lo = price_mid; }
  }
  std::cout << C_mkt << " : bisection also failed after 100 iters\n";
  return std::numeric_limits<double>::quiet_NaN();
}
