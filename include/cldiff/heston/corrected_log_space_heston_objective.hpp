#pragma once
#include <Eigen/Dense>
#include "cldiff/target.hpp"
#include "cldiff/heston/heston_objective.hpp"

// Log-space parameterization (kappa, theta, xi, v0 exp-transformed to stay
// positive; rho passed through) with the chain-rule scaling divided back out
// for the exp-transformed dims. Without this, the log-space gradient shrinks
// in proportion to the parameter's own value, so a parameter near zero stops
// generating signal even when the physical sensitivity is large -- measured
// at kappa=0.01: physical dL/dkappa=-0.134, log-space dL/d(log kappa) only
// -0.00134 (100x suppression), producing a repeatable collapse to the box
// floor/wall from every starting point. Dividing by p(i) restores a step
// whose relative size tracks the true physical sensitivity.
class CorrectedLogSpaceHestonObjective : public Objective {
  public:
    explicit CorrectedLogSpaceHestonObjective(const HestonObjective& inner);
    double value(const Eigen::VectorXd& y) const override;
    Eigen::VectorXd gradient(const Eigen::VectorXd& y) const override;
    int dim() const override;
    static Eigen::VectorXd toPhysical(const Eigen::VectorXd& y);
    static Eigen::VectorXd toLogSpace(const Eigen::VectorXd& p);

  private:
    const HestonObjective& inner_;
};
