// optimiser.hpp
#pragma once

#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/target.hpp"
#include <Eigen/Dense>

struct ProposalResult {
  Eigen::VectorXd best_particle;
  double result;
};

class Optimiser {
public:
  Optimiser(const Objective& objective, const Polyhedron& polyhedron, Rng& rng, Eigen::MatrixXd x_0,
            int r_rate, int chain_num, int run_num, double h, double eps);
  ProposalResult run();

private:
  void step();
  // todo: abstract this into a "chain" object which we feed into Sampler and Optimiser.
  // Currently just reimplementing the proposeState and proposal logic.
  Eigen::VectorXd updateChain(const Eigen::VectorXd& x) const;
  Eigen::VectorXd genTemp(int run_num) const;
  void resample();

  const Objective& objective_;
  const Polyhedron& polyhedron_;
  Rng& rng_;
  Eigen::MatrixXd x_;
  Eigen::VectorXd temp_;
  int r_rate_;
  int chain_num_;
  int run_num_;
  int step_num_;
  double h_;
  double eps_;
};
