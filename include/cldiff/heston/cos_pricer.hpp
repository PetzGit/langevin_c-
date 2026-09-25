#pragma once
#include "cumulants.hpp"
#include "characteristic_function.hpp" 

struct Range {
  double lower;
  double upper;
};

Range make_range(Cumulants& cum, double L);

double cos_pricer(const CharacteristicHeston& phi, const Range& r, double strike, double T, double rate, int term_num, double s_0, double q = 0.0);
