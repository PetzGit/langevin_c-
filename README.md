# cldiff

Barrier-Hessian ("Dikin metric") Langevin sampling and optimization on
polyhedra, plus a Heston stochastic-volatility calibration pipeline built on
top of it as an application. C++17, Eigen-based, CMake build.

Implements the method of Chok & Petzinna (2026), *Constrained Dikin–Langevin
diffusion for polyhedra* -- see [Reference](#reference) below.

## Layout

```
include/cldiff/          public headers
  polyhedron.hpp          A x <= b half-space representation + Dikin barrier geometry
  rng.hpp                 mt19937_64 wrapper (uniform/normal draws)
  target.hpp              Objective: abstract value()/gradient()/dim() interface
  sampler.hpp             Sampler: Dikin-Langevin sampler for polytope-constrained targets
  optimiser.hpp           Optimiser: annealed, multi-chain version of Sampler for optimization
  heston/                 Heston calibration module (below)
src/                      implementations, mirrors include/cldiff/
examples/                 buildable demos, one executable each (see below)
data_loader/              yfinance -> CSV pipeline for real SPX option-chain data
tests/                    Catch2 scaffolding (not currently wired into the build)
```

## Core: Polyhedron / Sampler / Optimiser

- **Polyhedron**: a convex region `{x : Ax <= b}`. Provides `contains()` and
  the log-barrier Hessian (`barrierHessian`) that gives the Dikin ellipsoid at
  a point `x` -- the local metric the sampler/optimiser use to stay inside
  the polytope without ever evaluating a hard constraint mid-step.
- **Sampler**: single-chain Dikin-Langevin sampler. Proposes moves scaled by
  the local barrier geometry, so step size naturally shrinks near the
  boundary and grows in the interior.
- **Optimiser**: `chain_num` parallel Sampler-like chains with a shared
  simulated-annealing temperature schedule (`genTemp`, three-phase cooling)
  and periodic resampling (every `r_rate` steps) toward the best-performing
  chains. Returns the best point found (`ProposalResult`). `chainBeta()` is a
  protected virtual hook so alternative annealing strategies can subclass
  `Optimiser` without touching the base implementation.

Demos: `examples/run_box_gaussian.cpp` (Sampler on a Gaussian target),
`examples/run_optimiser_nonconvex.cpp` (Optimiser on a nonconvex benchmark --
the sanity check that the core algorithm works before it's pointed at
anything financial).

## Heston module (`include/cldiff/heston`, `src/heston`)

Calibrates the Heston stochastic-volatility model to a set of
(maturity, strike, price) option quotes.

- **characteristic_function.hpp**: `CharacteristicHeston`, the Heston
  characteristic function.
- **cumulants.hpp / cos_pricer.hpp**: cumulant-based truncation range and the
  COS-method Fourier pricer (Fang & Oosterlee 2008). `make_range` clamps the
  truncation half-width to avoid catastrophic cancellation at extreme
  parameter draws.
- **black_scholes.hpp**: Black-Scholes price/vega/implied-vol (Newton solve),
  with a dividend yield `q` so forwards aren't biased when pricing an index.
- **market_data.hpp**: `MarketData` holds quotes by maturity and converts
  each to an implied vol on load; `loadFromCsv` reads whitespace-separated
  `T K price [weight]` rows.
- **feller.hpp**: checks the Feller condition `2*kappa*theta >= xi^2`.
- **heston_objective.hpp**: `HestonObjective` -- the calibration loss (sum of
  squared implied-vol errors across all quotes) plus a smooth softplus
  relaxation of the Feller penalty. The penalty is smoothed rather than a
  hard `max(0, x)` because a finite-difference gradient stencil straddling a
  hard kink can be sign-flipped -- real calibrated fits often sit right at
  the Feller boundary, so this isn't a rare edge case.
- **corrected_log_space_heston_objective.hpp**: `CorrectedLogSpaceHestonObjective`
  -- wraps `HestonObjective` in a log-space reparameterization (kappa, theta,
  xi, v0 exp-transformed to stay positive; rho passed through) with the
  chain-rule scaling divided back out. Without this correction the gradient
  the optimizer sees shrinks in proportion to a parameter's own value, so
  parameters near zero stop generating signal even when the true physical
  sensitivity is still large -- this is the fix that makes calibration
  actually converge instead of collapsing to the box boundary. **Use this
  class, not the raw `HestonObjective`, when calibrating.**

### Examples

- `examples/test_pricer.cpp`, `test_black_scholes.cpp`, `test_heston_objective.cpp`:
  unit-style sanity checks for the pricer, Black-Scholes, and objective in
  isolation.
- `examples/test_heston_calibration.cpp`: calibration on synthetic data.
- `examples/heston/harness.cpp` (target `heston_reference_recovery`):
  reference validation run -- fits synthetic data (known true params) from
  two starting points, one of them adversarial, using three arms (GD alone,
  annealed Optimiser, Optimiser + GD polish) and scores out-of-sample on
  held-out strikes. Demonstrates the Optimiser reaching basins plain GD
  cannot, and that the corrected objective recovers the true parameters
  rather than collapsing to the box boundary.
- `examples/heston/spx_calibration.cpp` (target `heston_spx_calibration`):
  the same corrected-objective + Optimiser + polish pipeline run against a
  real SPX snapshot from `data_loader/`. Run from `data_loader/` so
  `dat.csv`/`meta.txt` resolve from the working directory.

## data_loader

`data_loader.py` pulls a liquid SPX option chain via `yfinance`, estimates
the risk-free rate (`^IRX`) and dividend yield (put-call parity on near-money
strikes), filters for liquidity (open interest + bid-ask spread), and writes
`dat.csv` (`T K price` rows) and `meta.txt` (`S0 r q`) for
`heston_spx_calibration` to consume.

## Build

```
mkdir -p build && cd build
cmake ..
make -j
```

Fetches Eigen 3.4.0 and Catch2 at configure time. All targets link against
the `cldiff` static library, built with `-Wall -Wextra -Werror`.

## Reference

Chok, J. & Petzinna, D. (2026). Constrained Dikin–Langevin diffusion for
polyhedra. *IMA Journal of Applied Mathematics*, 91(2), 210-228.
https://doi.org/10.1093/imamat/hxag012
