// examples/test_black_scholes.cpp
//
// Sanity check for BlackSchole::price/vega/impliedVol:
// price a call at a known sigma, then recover sigma from that price
// via impliedVol, and check we get the original value back.

#include "cldiff/heston/black_scholes.hpp"
#include <cstdio>
#include <cmath>

int main() {
  const double S0 = 100.0;
  const double K  = 100.0;
  const double T  = 1.0;
  const double r  = 0.02;

  BlackSchole bs(S0, K, T, r);

  const double sigma_true_values[] = {0.10, 0.20, 0.40, 0.80};

  for (double sigma_true : sigma_true_values) {
    double C = bs.price(sigma_true);
    double sigma_recovered = bs.impliedVol(C);
    double err = std::abs(sigma_recovered - sigma_true);

    std::printf(
      "sigma_true = %.6f  price = %.6f  sigma_recovered = %.10f  abs_err = %.3e\n",
      sigma_true, C, sigma_recovered, err
    );
  }

  return 0;
}
