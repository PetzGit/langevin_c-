#include "cldiff/rng.hpp"

double Rng::uniform() {
  return uniform_(gen_);
}
double Rng::normal() {
  return normal_(gen_);
}
