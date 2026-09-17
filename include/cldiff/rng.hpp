#pragma once
#include <random>
#include <cstdint>

class Rng {
public:
  explicit Rng(std::uint64_t seed) : gen_(seed), uniform_(0.0,1.0), normal_(0.0, 1.0) {}
  double uniform();
  double normal();
private:
  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> uniform_;
  std::normal_distribution<double> normal_;
};
