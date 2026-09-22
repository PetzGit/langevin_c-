#include "cldiff/heston/cumulants.hpp"
#include <cmath>
#include <iostream>

Cumulants cumulants(const CharacteristicHeston& phi, double T) {
  const Eigen::VectorXd& p = phi.params();
  const double kappa = p(0);
  const double theta = p(1);
  const double xi    = p(2);   // sigma / vol-of-vol in the paper's notation
  const double rho   = p(3);
  const double v0    = p(4);

  const double ekt  = std::exp(-kappa * T);
  const double ek2t = std::exp(-2.0 * kappa * T);
  const double k2   = kappa * kappa;
  const double k3   = k2 * kappa;
  const double xi2  = xi * xi;

  // c1 — unchanged, driftless (rate*T added externally in cos_pricer's x)
  const double c_1 = (1.0 - ekt) * (theta - v0) / (2.0 * kappa) - theta * T / 2.0;

  // c2 — corrected formula (the classic Fang-Oosterlee 2008 c2 is known-wrong;
  // this is from the 2020 errata paper, eq. B5), split into v0-term and theta-term
  const double v0_term =
      v0 / (4.0 * k3) * (
          4.0 * k2 * (1.0 + (rho*xi*T - 1.0)*ekt)
        + kappa * (4.0*rho*xi*(ekt - 1.0) - 2.0*xi2*T*ekt)
        + xi2 * (1.0 - ek2t)
      );

  const double theta_term =
      theta / (8.0 * k3) * (
          8.0*k3*T
        - 8.0*k2*(1.0 + rho*xi*T + (rho*xi*T - 1.0)*ekt)
        + 2.0*kappa*((1.0 + 2.0*ekt)*xi2*T + 8.0*(1.0 - ekt)*rho*xi)
        + xi2*(ek2t + 4.0*ekt - 5.0)
      );

  double c_2 = v0_term + theta_term;

  if (c_2 < 0.0) {
    std::cerr << "NEGATIVE c2=" << c_2 << " at T=" << T
               << " kappa=" << kappa << " theta=" << theta
               << " xi=" << xi << " rho=" << rho << " v0=" << v0 << "\n";
    c_2 = 1e-8;
  }

  return Cumulants{c_1, c_2};
}
