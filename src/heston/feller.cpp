#include "cldiff/heston/feller.hpp"

bool satisfiesFeller(const Eigen::VectorXd& params) {
  return 2*params(0)*params(1) >= params(2)*params(2);
}
