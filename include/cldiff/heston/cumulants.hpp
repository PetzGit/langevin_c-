#pragma once
#include <cldiff/heston/characteristic_function.hpp>
//todo implement potential c_4

struct Cumulants {
  double c_1;
  double c_2;
};

Cumulants cumulants(const CharacteristicHeston& phi, double T);
