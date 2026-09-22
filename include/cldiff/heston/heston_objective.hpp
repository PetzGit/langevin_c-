#pragma once
#include <Eigen/Dense>
#include "cldiff/target.hpp"
#include "cldiff/heston/market_data.hpp"

class HestonObjective : public Objective {
  public:
    HestonObjective(const MarketData& market_data, int term_num, double L, double lambda_feller);
    double value(const Eigen::VectorXd& params) const override;
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const override;
    int dim() const override;

  private:
    const MarketData& market_data_;
    int term_num_;
    double L_;
    double lambda_feller_;
};

