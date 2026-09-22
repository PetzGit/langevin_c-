#include "cldiff/heston/characteristic_function.hpp"
#include <stdexcept>
//todo for specific implementation fix vector sizes, this is not good basically
double heston_params_check_size(double thing, const Eigen::VectorXd& params) {
  if (params.size() < 5) {throw std::invalid_argument("params list is incomplete");}
  return thing;
}

CharacteristicHeston::CharacteristicHeston(const Eigen::VectorXd& params) 
: k_(heston_params_check_size(params(0), params)), theta_(params(1)), xi_(params(2)), rho_(params(3)), v_0_(params(4)), params_(std::move(params)) {}

std::complex<double> CharacteristicHeston::eval(double u, double time) const {
  const std::complex<double> i(0.0,1.0);
  const std::complex<double> d = std::sqrt((k_ - i*rho_*xi_*u)*(k_ - i*rho_*xi_*u) + xi_*xi_*(i*u + u*u));
  const std::complex<double> num_g = (k_ - i * rho_ * u * xi_ -d);
  const std::complex<double> g = num_g/(k_ - i* rho_ * xi_*u + d);
  const std::complex<double> r_minus = num_g/(xi_*xi_);
  const std::complex<double> C = k_*(r_minus*time - 2/(xi_*xi_)*std::log((1.0-g*std::exp(-d*time))/(1.0-g)));
  const std::complex<double> D = r_minus*(1.0-std::exp(-d*time))/(1.0-g*std::exp(-d*time));
  return std::exp(C*theta_ + D*v_0_);
}

Eigen::VectorXd CharacteristicHeston::params() const{
  return params_;
}
