#pragma once
#include <map>
#include <vector>
#include <string>
#include "cldiff/heston/black_scholes.hpp"

struct Quote {
  double K;
  double market_price;
  double market_iv;
  double weight;
};

class MarketData {
  public:
    MarketData(double S0, double r);

    void addQuote(double T, double K, double market_price, double weight = 1.0);
    void loadFromCsv(const std::string& path);

    const std::map<double, std::vector<Quote>>& quotesByMaturity() const;
    double S0() const;
    double r() const;
  
  private:
    double S0_;
    double r_;
    std::map<double, std::vector<Quote>> quotes_by_maturity_;
};
