#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

#include "cldiff/heston/characteristic_function.hpp"
#include "cldiff/heston/cumulants.hpp"
#include "cldiff/heston/cos_pricer.hpp"

int main() {
  // Fang & Oosterlee (2008) benchmark Heston parameters.
  const double kappa = 1.5768;
  const double theta = 0.0398;
  const double xi    = 0.5751;
  const double rho   = -0.5711;
  const double v0    = 0.0175;

  Eigen::VectorXd params(5);
  params << kappa, theta, xi, rho, v0;

  const CharacteristicHeston phi(params);
  std::cout << phi.eval(0.0,1.0) << "\n";
  const double S0 = 100.0;
  const double K  = 100.0;
  const double T  = 1.0;
  const double r  = 0.0;
  const double L  = 10.0;  // truncation multiplier

  Cumulants cum = cumulants(phi, T);
  const Range range = make_range(cum, L);

  const double reference = 5.785155450;  // Fang & Oosterlee (2008), T = 1

  std::cout << std::setprecision(10);
  std::cout << "Truncation range: [" << range.lower << ", " << range.upper << "]\n\n";

  // Check convergence as N grows.
  for (int N : {16, 32, 64, 128, 256}) {
    const double price = cos_pricer(phi, range, K, T, r, N, S0);
    std::cout << "N = " << std::setw(4) << N
               << "  price = " << price
               << "  abs error = " << std::abs(price - reference) << "\n";
  }

  return 0;
}
