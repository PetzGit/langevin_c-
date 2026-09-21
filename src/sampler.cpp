#include "cldiff/sampler.hpp"
#include <stdexcept>
#include <Eigen/Cholesky>
#include <cmath>

Sampler::Sampler(const Objective& objective, const Polyhedron& polyhedron, Rng& rng, double eps, Eigen::VectorXd x, double h_max)
    : objective_(objective), polyhedron_(polyhedron), rng_(rng), eps_(eps), x_(std::move(x)), h_max_(h_max)
{
    if ((x_.size() != polyhedron_.size()) || (x_.size() != objective_.dim()) || !polyhedron_.contains(x_) || h_max_ < 0 ) {
        throw std::invalid_argument("Invalid arguments: please check the dimensions and interiority of x and the value of h_max");
    }
}
Eigen::MatrixXd Sampler::run(int n) {
    Eigen::MatrixXd trajectory(x_.size(),n);
    for(int i{0}; i < n; i++) {
      trajectory.col(i) = x_;
      step();
    }
    return trajectory;
}


void Sampler::step() {
    const Eigen::MatrixXd H_eps = polyhedron_.barrierHessian(x_, eps_);
    const Eigen::LLT<Eigen::MatrixXd> llt(H_eps);
    const Eigen::MatrixXd L = llt.matrixL();
    const Eigen::VectorXd m_x = proposeState(x_, llt);
    const double h = rng_.uniform() * h_max_;
    const Proposal proposal1 = propose(x_, m_x, h, L);
    const Eigen::VectorXd y = proposal1.Y;
    if (!polyhedron_.contains(y)) {
        return;
    }
    const double log_diff = objective_.value(x_) - objective_.value(y);
    const double quad1 = 2.0 * h * proposal1.x_i.squaredNorm();
    const double log_det1 = -2.0 * L.diagonal().array().log().sum();
    const Eigen::MatrixXd H_y_eps = polyhedron_.barrierHessian(y, eps_);
    const Eigen::LLT<Eigen::MatrixXd> llt_y(H_y_eps);
    const Eigen::MatrixXd L_y = llt_y.matrixL();
    const Eigen::VectorXd m_y = proposeState(y, llt_y);
    const Eigen::VectorXd diff = x_ - (y + h * m_y);
    const double quad2 = diff.dot(H_y_eps * diff);
    const double log_det2 = -2.0 * L_y.diagonal().array().log().sum();
    const double acceptance = log_diff + (0.25/h) * quad1 + (0.5)*log_det1 - (0.25/h)* quad2 - (0.5)*log_det2;
    if (std::min(0.0, acceptance) > std::log(rng_.uniform())) {
        x_ = y;
        return;
    }
}
Eigen::VectorXd Sampler::proposeState(const Eigen::VectorXd& x, const Eigen::LLT<Eigen::MatrixXd>& llt, const double beta) const {
    const Eigen::VectorXd div_c_eps = polyhedron_.barrierHessianDivergence(llt, x);
    const Eigen::VectorXd grad_term = -1.0 * llt.solve(objective_.gradient(x))/ beta;
    return grad_term + div_c_eps;
}
Proposal Sampler::propose(const Eigen::VectorXd& x, const Eigen::VectorXd& m_x, const double h, const Eigen::MatrixXd& L, const double beta) const {
    const int d = m_x.size();
    Eigen::VectorXd x_i(d);
    for (int i{0}; i < d; i++) {
        x_i(i) = rng_.normal();
    }
    const Eigen::VectorXd eta = L.triangularView<Eigen::Lower>().transpose().solve(x_i);
    return Proposal{x + h * m_x + std::sqrt(2.0 * h / beta) * eta, x_i};
}
