#pragma once

class BlackSchole {
  public:
    BlackSchole(double S0, double K, double T, double r, double q = 0.0);
    double price(double sigma) const;
    double vega(double sigma) const;
    double impliedVol(double C_mkt) const;

  private:
    double S0_;
    double K_;
    double T_;
    double r_;
    double q_;
};
