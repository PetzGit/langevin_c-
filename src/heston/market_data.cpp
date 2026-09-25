#include "cldiff/heston/market_data.hpp"
#include "cldiff/heston/black_scholes.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
MarketData::MarketData(double S0, double r, double q)
: S0_(S0), r_(r), q_(q) {

}

void MarketData::addQuote(double T, double K, double market_price, double weight) {
  BlackSchole iv_calc(S0_, K, T, r_, q_);
  double iv = iv_calc.impliedVol(market_price);
  if (iv!=iv) {
    std::cerr << "Warning: implied vol did not converge for K=" << K << ", T=" << T << "\n";
    return;
  }
  Quote quote{K, market_price, iv, weight};
  quotes_by_maturity_[T].push_back(quote);
}
double MarketData::S0() const { return S0_;}
double MarketData::r() const { return r_;}
double MarketData::q() const { return q_;}
const std::map<double, std::vector<Quote>>& MarketData::quotesByMaturity() const {
  return quotes_by_maturity_;
}

void MarketData::loadFromCsv(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("MarketData::loadFromCsv: could not open " + path);
  }
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double T, K, price, weight = 1.0;
    if (!(iss >> T >> K >> price)) {
      continue;
    }
    iss >> weight;  // optional trailing column; keeps default 1.0 if absent
    addQuote(T, K, price, weight);
  }
}

