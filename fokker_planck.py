"""
Brownian motion with drift: three independent solutions of the Fokker-Planck equation.

The process:   dX = mu*dt + sigma*dW,  X(0) = 0
The PDE:       dp/dt = -mu * dp/dx + (sigma^2 / 2) * d2p/dx2,   p(x, 0) = delta(x)
The solution:  p(x, t) = Normal(mean = mu*t, variance = sigma^2 * t)

Three routes to the same answer:
  1. analytical_density   - exact Gaussian (moving frame -> heat equation -> Fourier)
  2. FiniteDifferenceSolver - explicit FTCS scheme on the PDE
  3. monte_carlo_simulation - Euler-Maruyama simulation of many random paths
"""
from __future__ import annotations
from dataclasses import dataclass
import time

import numpy as np


# ---------------------------------------------------------------------------
# 1. Analytical solution
# ---------------------------------------------------------------------------

def analytical_density(x: np.ndarray, t: float, mu: float, sigma: float) -> np.ndarray:
    """Exact solution: Gaussian with mean mu*t and variance sigma^2*t."""
    if t <= 0:
        raise ValueError("t must be positive (delta function at t=0).")
    variance = sigma ** 2 * t
    return np.exp(-((x - mu * t) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


# ---------------------------------------------------------------------------
# 2. Finite-difference solver (explicit FTCS)
# ---------------------------------------------------------------------------

@dataclass
class FDResult:
    x: np.ndarray          # spatial grid
    p: np.ndarray          # density at t_final
    r: float               # stability number  r = D*dt/dx^2  (stable iff r <= 0.5)
    stable: bool           # did the solution stay finite and sensible
    dt: float
    dx: float
    runtime_ms: float


class FiniteDifferenceSolver:
    """
    Explicit forward-time, centred-space (FTCS) scheme for
        dp/dt = -mu * dp/dx + D * d2p/dx2,   D = sigma^2 / 2.

    Stability requires r = D*dt/dx^2 <= 1/2. Exceed it and the scheme
    amplifies grid-scale noise exponentially - which we demonstrate on purpose.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.D = 0.5 * sigma ** 2
        self.sigma = sigma

    def solve(self, t_final: float, dx: float = 0.02, r: float = 0.4,
              t_start: float = 0.05) -> FDResult:
        start = time.perf_counter()

        # Domain wide enough to hold the drifted, spread-out Gaussian.
        mean_T = self.mu * t_final
        sd_T = self.sigma * np.sqrt(t_final)
        lo, hi = mean_T - 8 * sd_T - 1.0, mean_T + 8 * sd_T + 1.0
        nx = int(round((hi - lo) / dx)) + 1
        x = np.linspace(lo, hi, nx)
        dx = x[1] - x[0]

        # Timestep from the requested stability number r = D*dt/dx^2.
        dt = r * dx ** 2 / self.D
        n_steps = max(1, int(round((t_final - t_start) / dt)))

        # Start from the exact narrow Gaussian at t_start (avoids the raw delta).
        p = analytical_density(x, t_start, self.mu, self.sigma)

        c = self.mu * dt / (2 * dx)      # advection coefficient (centred)
        d = self.D * dt / dx ** 2        # diffusion coefficient ( = r )

        # np.errstate: the deliberately unstable runs overflow by design;
        # we detect that via the `stable` flag instead of spraying warnings.
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(n_steps):
                p_new = p.copy()
                p_new[1:-1] = (p[1:-1]
                               - c * (p[2:] - p[:-2])
                               + d * (p[2:] - 2 * p[1:-1] + p[:-2]))
                p_new[0] = p_new[-1] = 0.0   # Dirichlet: density ~ 0 far away
                p = p_new

        mass = np.trapezoid(p, x)
        stable = bool(np.all(np.isfinite(p)) and abs(mass - 1.0) < 0.05
                      and float(np.max(np.abs(p))) < 1e3)

        runtime_ms = (time.perf_counter() - start) * 1000
        return FDResult(x=x, p=p, r=d, stable=stable, dt=dt, dx=dx,
                        runtime_ms=runtime_ms)


# ---------------------------------------------------------------------------
# 3. Monte Carlo simulation (Euler-Maruyama)
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    samples: np.ndarray    # terminal positions X(T) of every path
    empirical_mean: float
    empirical_var: float
    runtime_ms: float


def monte_carlo_simulation(n_paths: int, t_final: float, mu: float, sigma: float,
                           n_steps: int = 100, seed: int | None = None) -> MCResult:
    """
    Simulate n_paths independent trajectories of dX = mu*dt + sigma*dW
    with the Euler-Maruyama scheme:  X_{k+1} = X_k + mu*dt + sigma*sqrt(dt)*Z.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    dt = t_final / n_steps

    increments = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    terminal = increments.sum(axis=1)   # X(T) for every path (X(0) = 0)

    runtime_ms = (time.perf_counter() - start) * 1000
    return MCResult(samples=terminal,
                    empirical_mean=float(terminal.mean()),
                    empirical_var=float(terminal.var()),
                    runtime_ms=runtime_ms)
