#include "cldiff/heston/cos_pricer.hpp"
#include <cmath>
Range make_range(Cumulants& cum, double L) {
  double width = L*std::sqrt(cum.c_2);
  return Range{cum.c_1 - width, cum.c_1 + width};
}
//todo fix the pow thing
double cos_pricer(const CharacteristicHeston& phi, const Range& r, double strike, double T, double rate, int term_num, double s_0) {
  const double L = r.upper - r.lower;
  const double x = std::log(s_0) + rate*T - std::log(strike);
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
