#include "cldiff/polyhedron.hpp"
#include <stdexcept>
#include <Eigen/Cholesky>

Polyhedron::Polyhedron(Eigen::MatrixXd A, Eigen::VectorXd b)
  : A_(std::move(A)), b_(std::move(b))
{
  if (A_.rows() != b_.size()) {
    throw std::invalid_argument("A and b must have matching row/lengths counts");
  }
}
int Polyhedron::size() const {
  return A_.cols();
}
bool Polyhedron::contains(const Eigen::VectorXd& x) const {
  if(x.size() != A_.cols()) {
    throw std::invalid_argument("Dimensions of x must match dimensions of polyhedron");
  }
  return ((A_ * x).array() <= b_.array()).all();
}
//this method assumes that x lies within polyhedron. Sampler's responsibility to ensure x is within.
Eigen::MatrixXd Polyhedron::barrierHessian(const Eigen::VectorXd& x, double eps) const {
  Eigen::VectorXd s = b_ - A_ * x;
  int n = A_.cols();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n,n);
  for(int i{0}; i < A_.rows(); i++) {
    Eigen::VectorXd a_i = A_.row(i).transpose();
    res+= (a_i * a_i.transpose()) / (s(i)*s(i));
  }
  return res + eps*Eigen::MatrixXd::Identity(n,n);
}
Eigen::VectorXd Polyhedron::barrierHessianDivergence(const Eigen::LLT<Eigen::MatrixXd> llt, const Eigen::VectorXd& x) const {
  Eigen::VectorXd matrix_res = Eigen::MatrixXd::Zero(A_.cols(),1);
  Eigen::VectorXd s = b_ - A_ * x;
  for(int i{0}; i < A_.rows(); i++) {
    Eigen::VectorXd a_i = A_.row(i).transpose();
    Eigen::VectorXd C_a_i = llt.solve(a_i);
    double coeff_i = (a_i.dot(C_a_i))/(s(i)*s(i)*s(i));
    matrix_res += coeff_i * a_i;
  }
  matrix_res *= -2;
  Eigen::VectorXd res = llt.solve(matrix_res);
  return res;
}

