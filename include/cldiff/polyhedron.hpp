#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
class Polyhedron {
  public:
    Polyhedron(Eigen::MatrixXd A, Eigen::VectorXd b);
    int size() const ; //todo rename this to dimension
    bool contains(const Eigen::VectorXd& x) const;
    Eigen::MatrixXd barrierHessian(const Eigen::VectorXd& x, double eps) const;
    Eigen::VectorXd barrierHessianDivergence(const Eigen::LLT<Eigen::MatrixXd> llt, const Eigen::VectorXd& x) const;
    const Eigen::MatrixXd& A() const;
    const Eigen::VectorXd& b() const;
  private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
 };
