#include "cldiff/heston/corrected_log_space_heston_objective.hpp"
#include <cmath>

CorrectedLogSpaceHestonObjective::CorrectedLogSpaceHestonObjective(const HestonObjective& inner)
: inner_(inner) {}

double CorrectedLogSpaceHestonObjective::value(const Eigen::VectorXd& y) const {
  return inner_.value(toPhysical(y));
}

Eigen::VectorXd CorrectedLogSpaceHestonObjective::gradient(const Eigen::VectorXd& y) const {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(5);
  const double h = 3e-3;
  for (int i = 0; i < 5; i++) {
    Eigen::VectorXd yp = y, ym = y;
    yp(i) += h;
    ym(i) -= h;
    grad(i) = (value(yp) - value(ym)) / (2.0 * h);
  }
  const Eigen::VectorXd p = toPhysical(y);
  for (int i : {0, 1, 2, 4}) {
    grad(i) /= std::max(p(i), 1e-8);
  }
  return grad;
}

int CorrectedLogSpaceHestonObjective::dim() const {
  return 5;
}

Eigen::VectorXd CorrectedLogSpaceHestonObjective::toPhysical(const Eigen::VectorXd& y) {
  Eigen::VectorXd p(5);
  p(0) = std::exp(y(0));
  p(1) = std::exp(y(1));
  p(2) = std::exp(y(2));
  p(3) = y(3);
  p(4) = std::exp(y(4));
  return p;
}

Eigen::VectorXd CorrectedLogSpaceHestonObjective::toLogSpace(const Eigen::VectorXd& p) {
  Eigen::VectorXd y(5);
  y(0) = std::log(p(0));
  y(1) = std::log(p(1));
  y(2) = std::log(p(2));
  y(3) = p(3);
  y(4) = std::log(p(4));
  return y;
}
