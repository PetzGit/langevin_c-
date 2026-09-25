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

// Base annealed-Langevin-on-polyhedron optimiser. Default behaviour (a single
// shared temperature schedule applied identically to every chain) is
// unchanged from the original implementation -- existing callers see no
// difference. The one design decision that plausibly benefits from variation
// -- what temperature chain i gets at a given step -- is a protected virtual
// hook (chainBeta), so alternative annealing strategies (e.g. a parallel
// schedule that keeps some chains persistently warmer for ongoing
// exploration) can be added as subclasses without touching this class.
class Optimiser {
public:
  Optimiser(const Objective& objective, const Polyhedron& polyhedron, Rng& rng, Eigen::MatrixXd x_0,
            int r_rate, int chain_num, int run_num, double h, double eps);
  virtual ~Optimiser() = default;
  ProposalResult run();

protected:
  void step();
  // Temperature assigned to chain i at the current step_num_. Default:
  // the single shared schedule, identical for every chain (original
  // behaviour). Override to give chains heterogeneous schedules.
  virtual double chainBeta(int chain_index) const;
  // todo: abstract this into a "chain" object which we feed into Sampler and Optimiser.
  // Currently just reimplementing the proposeState and proposal logic.
  Eigen::VectorXd updateChain(const Eigen::VectorXd& x, double beta) const;
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
