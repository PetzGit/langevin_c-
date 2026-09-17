#include "cldiff/sampler.hpp"
#include "cldiff/polyhedron.hpp"
#include "cldiff/rng.hpp"
#include "cldiff/target.hpp"
#include <Eigen/Dense>
#include <iostream>
class GaussianObjective : public Objective {
public:
    explicit GaussianObjective(int d) : d_(d) {}

    double value(const Eigen::VectorXd& x) const override {
        return 0.5 * x.squaredNorm();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return x;
    }
    int dim() const override { return d_; }

private:
    int d_;
};



int main() {
 const Eigen::VectorXd b = Eigen::VectorXd::Ones(6);
 Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6,3);
 for(int i{0}; i < 3; i++) {
   A(2*i,i) = 1;
   A(2*i+1,i) = -1;
 }
 Polyhedron poly(A,b);
 Rng rng(2);
 GaussianObjective gaussian(3);
 double eps = 1e-5;
 double h_max = 0.1;
 Eigen::VectorXd x_0 = Eigen::VectorXd::Zero(3);
 Sampler sampler(gaussian, poly, rng, eps, x_0, h_max);
 Eigen::MatrixXd output = sampler.run(100000);
 //Eigen::VectorXd mean_position = output.rowwise().mean();
 std::cout << output << std::endl;
 return 0;
}
