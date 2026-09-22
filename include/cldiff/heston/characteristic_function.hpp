#pragma once
#include <Eigen/Dense>
#include <complex>
//heston params will be supplied from sampler/optimiser itself. Optimiser owns the parameter space
class CharacteristicHeston{
  public:
    std::complex<double> eval(double u, double time) const;
    CharacteristicHeston(const Eigen::VectorXd& params);
    Eigen::VectorXd params() const;
  private:
    double k_;
    double theta_;
    double xi_;
    double rho_;
    double v_0_;
    Eigen::VectorXd params_;
};
