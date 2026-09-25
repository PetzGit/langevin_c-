#include "cldiff/heston/cos_pricer.hpp"
#include <cmath>
#include <algorithm>

// Truncation half-width is capped so it can never outgrow what term_num terms
// can resolve. Without this, parameter draws with large c_2 (e.g. low kappa,
// high xi/v0 -- well within a typical calibration search box) blow the range
// out to tens of units; chi_i's exp(b_z) term then requires cancellation far
// beyond what 128 terms (or double precision) can deliver, and cos_pricer
// silently returns garbage (negative or absurdly large prices) instead of an
// error. 6 was chosen empirically: it reproduces unclamped prices to ~1e-10
// for every well-behaved parameter set tested (the density beyond 6 is
// already negligible there), while eliminating blow-ups for extreme draws.
constexpr double kMaxHalfWidth = 6.0;

Range make_range(Cumulants& cum, double L) {
  double width = std::min(L*std::sqrt(cum.c_2), kMaxHalfWidth);
  return Range{cum.c_1 - width, cum.c_1 + width};
}
//todo fix the pow thing
double cos_pricer(const CharacteristicHeston& phi, const Range& r, double strike, double T, double rate, int term_num, double s_0, double q) {
  const double L = r.upper - r.lower;
  // Forward uses the risk-neutral drift (rate - q); discounting below still
  // uses the pure rate -- these are two different roles a single "rate"
  // can't correctly serve once the underlying pays a dividend/carries a cost.
  const double x = std::log(s_0) + (rate - q)*T - std::log(strike);
  const double a_z = r.lower + x;
  const double b_z = r.upper + x;
  double sum = 0;
  for (int i{0}; i < term_num; i++) {
    const double omega_i = i*M_PI/L;
    const std::complex<double> phi_i = phi.eval(omega_i, T);
    const double chi_i = (std::pow(-1,i)*std::exp(b_z) - std::cos(omega_i*a_z) + omega_i*std::sin(omega_i*a_z))/(1 + omega_i*omega_i);
    double psi_i;
    double w_i;
    if (i==0) {
      psi_i = b_z;
      w_i = 0.5;
    } else {
      psi_i = std::sin(omega_i *a_z)/omega_i;
      w_i = 1;
    }
    const double V_i = 2*strike/L * (chi_i - psi_i);
    sum+= w_i*(phi_i.real() * std::cos(omega_i*r.lower) + phi_i.imag() * std::sin(omega_i*r.lower))*V_i;
  }
  return std::exp(-rate*T) * sum;
}
