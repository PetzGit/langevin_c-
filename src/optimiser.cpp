#include "cldiff/optimiser.hpp"
#include <Eigen/Cholesky>
#include <stdexcept>
#include <cmath>

Optimiser::Optimiser(const Objective& objective, const Polyhedron& polyhedron, Rng& rng, Eigen::MatrixXd x_0,
                      int r_rate, int chain_num, int run_num, double h, double eps)
    : objective_(objective), polyhedron_(polyhedron), rng_(rng), x_(std::move(x_0)),
      r_rate_(r_rate), chain_num_(chain_num), run_num_(run_num), step_num_(0), h_(h), eps_(eps)
{
  if (polyhedron_.size() != x_.rows() || polyhedron_.size() != objective_.dim() || h_ <= 0) {
    throw std::invalid_argument("Invalid arguments: check dimensions and h");
  }
  for (int i = 0; i < x_.cols(); ++i) {
    if (!polyhedron_.contains(x_.col(i))) {
      throw std::invalid_argument("Every initial particle must be interior to the polyhedron");
    }
  }
  temp_ = genTemp(run_num_);
}

Eigen::VectorXd Optimiser::genTemp(int run_num) const {
  Eigen::VectorXd temp(run_num);
  const int phase1_end = static_cast<int>(0.875 * run_num);
  const int phase2_end = static_cast<int>(0.975 * run_num);

  for (int k = 0; k < run_num; k++) {
    const double base = 1.0 / std::log(3.0 + 2.0 * h_ * k);
    if (k < phase1_end) {
      temp(k) = base;
    } else if (k < phase2_end) {
      temp(k) = std::pow(base, 4.0);
    } else {
      temp(k) = 0.0;
    }
  }
  return temp;
}

Eigen::VectorXd Optimiser::updateChain(const Eigen::VectorXd& x) const {
  const double beta = temp_(step_num_);
  const Eigen::MatrixXd H_eps = polyhedron_.barrierHessian(x, eps_);
  const Eigen::LLT<Eigen::MatrixXd> llt(H_eps);
  const Eigen::VectorXd div_c_eps = polyhedron_.barrierHessianDivergence(llt, x);
  const Eigen::VectorXd grad_term = -1.0 * llt.solve(objective_.gradient(x));

  const int d = x.size();
  Eigen::VectorXd n_i(d);
  for (int i = 0; i < d; i++) {
    n_i(i) = rng_.normal();
  }
  const Eigen::VectorXd eta = llt.matrixL().transpose().solve(n_i);

  return x + h_ * (grad_term + div_c_eps / beta) + std::sqrt(2.0 * h_ / beta) * eta;
}

void Optimiser::step() {
  for (int i = 0; i < chain_num_; i++) {
    Eigen::VectorXd proposal = updateChain(x_.col(i));
    if (!polyhedron_.contains(proposal)) {
      continue;
    }
    x_.col(i) = proposal;
  }
  if ((step_num_ + 1) % r_rate_ == 0) {
    resample();
  }
  step_num_ += 1;
}

void Optimiser::resample() {
  const int N = x_.cols();

  Eigen::VectorXd f_values(N);
  for (int i = 0; i < N; i++) {
    f_values(i) = objective_.value(x_.col(i));
  }
  const double f_min = f_values.minCoeff();

  Eigen::VectorXd evals(N);
  int best_ind = 0;
  for (int i = 0; i < N; i++) {
    evals(i) = std::exp(-(f_values(i) - f_min));
    if (f_values(i) < f_values(best_ind)) {
      best_ind = i;
    }
  }
  evals /= evals.sum();

  Eigen::VectorXd cum_sum(N);
  cum_sum(0) = evals(0);
  for (int k = 1; k < N; k++) {
    cum_sum(k) = cum_sum(k - 1) + evals(k);
  }

  Eigen::MatrixXd new_x(x_.rows(), N);
  new_x.col(0) = x_.col(best_ind);
  for (int j = 1; j < N; j++) {
    const double target = rng_.uniform();
    int l = 0;
    while (l < N - 1 && cum_sum(l) < target) {
      ++l;
    }
    new_x.col(j) = x_.col(l);
  }
  x_ = std::move(new_x);

  // todo: replace this linear scan with a binary search (std::lower_bound
  // against cum_sum) for O(log N) per draw instead of O(N).
}

ProposalResult Optimiser::run() {
  while (step_num_ < run_num_) {
    step();
  }

  double best_eval = objective_.value(x_.col(0));
  int best_ind = 0;
  for (int i = 1; i < x_.cols(); i++) {
    const double val = objective_.value(x_.col(i));
    if (val < best_eval) {
      best_eval = val;
      best_ind = i;
    }
  }
  return ProposalResult{x_.col(best_ind), best_eval};
}
