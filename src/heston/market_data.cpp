#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include <iostream>
MarketData::MarketData(double S0, double r)
: S0_(S0), r_(r) {
  
}

void MarketData::addQuote(double T, double K, double market_price, double weight) {
  BlackSchole iv_calc(S0_, K, T, r_);
  double iv = iv_calc.impliedVol(market_price);
  if (iv!=iv) {
    std::cerr << "Warning: implied vol did not converge for K=" << K << ", T=" << T << "\n";
    return;
  }
  Quote q{K, market_price, iv, weight};
  quotes_by_maturity_[T].push_back(q);
}
double MarketData::S0() const { return S0_;}
double MarketData::r() const { return r_;}
const std::map<double, std::vector<Quote>>& MarketData::quotesByMaturity() const {
  return quotes_by_maturity_;
}

