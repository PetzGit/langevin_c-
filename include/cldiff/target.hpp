#pragma once
#include <Eigen/Dense>

class Objective {
  public:
    virtual double value(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) const = 0;
    virtual int dim() const = 0;
    virtual ~Objective();
};
