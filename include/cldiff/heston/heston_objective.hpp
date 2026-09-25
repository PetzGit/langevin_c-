#pragma once
#include <Eigen/Dense>
#include "cldiff/target.hpp"
#include "cldiff/heston/market_data.hpp"

class HestonObjective : public Objective {
  public:
    // feller_smooth_scale: width over which the Feller penalty
    // max(0, xi^2 - 2*kappa*theta) is smoothed into a softplus. The hard
    // max(0,x) is continuous but has a non-differentiable kink at x=0;
    // whenever a finite-difference stencil straddles that kink (common in
    // practice -- real calibrated Heston fits routinely sit right at the
    // Feller boundary), the estimated gradient can be badly wrong, even
    // sign-flipped, even though the true (piecewise) function is well
    // approximated by a smooth curve away from the immediate kink. Default
    // 0.01 is comfortably wider than the finite-difference step sizes used
    // elsewhere in this objective's gradient (~1e-3 in log-space).
    HestonObjective(const MarketData& market_data, int term_num, double L, double lambda_feller,
                     double feller_smooth_scale = 0.01);
    double value(const Eigen::VectorXd& params) const override;
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const override;
    int dim() const override;

  private:
    const MarketData& market_data_;
    int term_num_;
    double L_;
    double lambda_feller_;
    double feller_smooth_scale_;
};

