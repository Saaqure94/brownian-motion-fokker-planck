# Brownian Motion with Drift — One Problem, Three Independent Solutions

A small, self-validating project: solve the Fokker–Planck equation for a drifting,
diffusing particle **three independent ways**, show they agree, and analyse the
results in SQL.

## The problem

A particle starts at the origin and follows the stochastic differential equation

```
dX = μ dt + σ dW
```

Its probability density `p(x, t)` obeys the Fokker–Planck equation — a heat
equation with an advection (drift) term:

```
∂p/∂t = −μ ∂p/∂x + (σ²/2) ∂²p/∂x²,      p(x, 0) = δ(x)
```

## Three routes to the same answer

| # | Method | File | Idea |
|---|--------|------|------|
| 1 | **Analytical** | `fokker_planck.py` | Change to the moving frame `y = x − μt` — the drift cancels, leaving the pure heat equation. Solve by Fourier transform → Gaussian with mean `μt`, variance `σ²t`. |
| 2 | **Finite differences** | `fokker_planck.py` | Explicit FTCS scheme on the PDE. Stable iff `r = DΔt/Δx² ≤ ½` — demonstrated deliberately in both directions. |
| 3 | **Monte Carlo** | `fokker_planck.py` | Euler–Maruyama simulation of up to 100,000 independent paths; empirical distribution vs theory. |

Every run (parameters, errors, runtimes, stability flags) is stored in **SQLite**
and analysed with SQL — including a window-function query showing Monte Carlo
error falling like `1/√N`, and a query exposing the finite-difference stability
boundary.

## Results

- **Three-way agreement:** analytical curve, FD solution, and MC histogram lie on
  top of each other (`plots/three_way_agreement.png`).
- **Monte Carlo convergence:** each 10× increase in paths multiplies the error by
  ≈ 0.32 — matching the theoretical `1/√10 ≈ 0.316` (`plots/mc_convergence.png`).
- **Stability boundary:** the scheme is accurate up to `r = 0.5` and blows up
  immediately beyond it — same physics, same code, only the timestep changed
  (`plots/stability_demo.png`).

## Run it

```bash
python experiments.py        # runs all sweeps, writes results.db, renders plots,
                             # prints the SQL analyses
python test_fokker_planck.py # 5 tests: mass conservation, moments, FD accuracy,
                             # instability detection, MC convergence
```

Requires only `numpy` and `matplotlib` (SQLite is in the standard library).

## Why this project

Accuracy and reliability in numerical systems can't be taken on faith — they have
to be demonstrated. Independent methods agreeing, automated tests, and stored,
queryable results are how I approach that in real data engineering work too.
