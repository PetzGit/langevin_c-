#pragma once
#include <Eigen/Dense>
#include "cldiff/target.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include <Eigen/Cholesky>
struct Proposal {
  Eigen::VectorXd Y;
  Eigen::VectorXd x_i;
};


class Sampler {
public:
  Sampler(const Objective& objective, const Polyhedron& polyhedron, Rng& rng, double eps, Eigen::VectorXd x, double h_max);
  void step();
  Eigen::MatrixXd run(int n);

private:
  Eigen::VectorXd proposeState(const Eigen::VectorXd& x, const Eigen::LLT<Eigen::MatrixXd>& llt) const;
  Proposal propose(const Eigen::VectorXd& x, const Eigen::VectorXd& m_x, double h, const Eigen::MatrixXd& L) const;

  const Objective& objective_;
  const Polyhedron& polyhedron_;
  Rng& rng_;
  double eps_;
  Eigen::VectorXd x_;
  double h_max_;
};
