// examples/run_optimiser_nonconvex_multiseed.cpp
//
// Validation test, not library code: multi-seed comparison of the
// interacting (resampling) annealed optimiser against N independent
// non-interacting chains on the paper's Section 4.3 benchmark.
//
// f(x) = sum_i ( x_i^2 - cos(6*pi*x_i^2) )
// Global min at x=0, f(0)=-1 per dimension (0 - cos(0) = -1), so for
// d dimensions the global optimum value is -d. Several non-global local
// minima sit nearby (cos(6*pi*x_i^2) oscillates), which is what makes
// this a real robustness-to-local-minima test rather than a trivial one.
//
// "Non-interacting" here means: same chains, same annealing schedule,
// but r_rate set beyond run_num so resample() never fires -- i.e. N
// independent copies of the same Langevin diffusion with no interaction,
// which is the comparison the paper's abstract itself makes.

#include "cldiff/optimiser.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/target.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

class NonConvexObjective : public Objective {
public:
    explicit NonConvexObjective(int d) : d_(d) {}

    double value(const Eigen::VectorXd& x) const override {
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            sum += x(i) * x(i) - std::cos(6.0 * M_PI * x(i) * x(i));
        }
        return sum;
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd g(x.size());
        for (int i = 0; i < x.size(); i++) {
            g(i) = 2.0 * x(i) + 12.0 * M_PI * x(i) * std::sin(6.0 * M_PI * x(i) * x(i));
        }
        return g;
    }

    int dim() const override { return d_; }

private:
    int d_;
};

struct RunSummary {
    double best_value;
    bool success;
};

RunSummary runOnce(NonConvexObjective& obj, const Polyhedron& poly,
                    const Eigen::MatrixXd& x0, int r_rate, int chain_num,
                    int run_num, double h, double eps, int seed,
                    double target, double tol) {
    Rng rng(seed);
    Optimiser opt(obj, poly, rng, x0, r_rate, chain_num, run_num, h, eps);
    ProposalResult result = opt.run();
    return RunSummary{result.result, std::abs(result.result - target) < tol};
}

int main() {
    const int d = 20;        // up from 3 -- more coordinates that all have to
                              // land in the right basin for a non-interacting
                              // chain to succeed purely by luck
    const int N = 50;
    const int r_rate = 20;
    const int run_num = 1000; // down from 100000 -- less time for a single
                               // chain's own noise to escape a bad basin
    const double h = 0.01;
    const double eps = 1e-5;

    const double target = -1.0 * d;   // global optimum value
    const double tol = 0.02 * d;      // scaled with d so the bar stays
                                       // proportionally as tight per-coordinate

    const std::vector<int> seeds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const Eigen::VectorXd b = Eigen::VectorXd::Ones(2 * d);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * d, d);
    for (int i = 0; i < d; i++) {
        A(2 * i, i) = 1;
        A(2 * i + 1, i) = -1;
    }
    Polyhedron poly(A, b);

    Eigen::MatrixXd x0 = Eigen::MatrixXd::Constant(d, N, 0.7);
    NonConvexObjective obj(d);

    // r_rate beyond run_num => resample() never fires => N independent chains.
    const int r_rate_noninteracting = run_num + 1;

    int interacting_successes = 0;
    int noninteracting_successes = 0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "target = " << target << ", tol = " << tol << "\n\n";
    std::cout << std::setw(6) << "seed"
              << std::setw(18) << "interacting"
              << std::setw(10) << "ok"
              << std::setw(20) << "non-interacting"
              << std::setw(10) << "ok" << "\n";

    for (int seed : seeds) {
        RunSummary interacting = runOnce(obj, poly, x0, r_rate, N, run_num,
                                          h, eps, seed, target, tol);
        RunSummary noninteracting = runOnce(obj, poly, x0, r_rate_noninteracting,
                                             N, run_num, h, eps, seed, target, tol);

        interacting_successes += interacting.success;
        noninteracting_successes += noninteracting.success;

        std::cout << std::setw(6) << seed
                  << std::setw(18) << interacting.best_value
                  << std::setw(10) << (interacting.success ? "yes" : "no")
                  << std::setw(20) << noninteracting.best_value
                  << std::setw(10) << (noninteracting.success ? "yes" : "no")
                  << "\n";
    }

    std::cout << "\ninteracting success rate:     "
              << interacting_successes << "/" << seeds.size() << "\n";
    std::cout << "non-interacting success rate: "
              << noninteracting_successes << "/" << seeds.size() << "\n";

    return 0;
}
