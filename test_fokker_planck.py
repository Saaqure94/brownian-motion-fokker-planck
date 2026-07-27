"""
Test suite - written the way I work: the checks that would catch a wrong
answer are automated, not eyeballed.  Run:  python test_fokker_planck.py
(or pytest, if available).
"""

import numpy as np

from fokker_planck import (analytical_density, FiniteDifferenceSolver,
                           monte_carlo_simulation)

MU, SIGMA, T = 1.5, 1.0, 2.0


def test_analytical_density_integrates_to_one():
    x = np.linspace(-15, 20, 4001)
    mass = np.trapz(analytical_density(x, T, MU, SIGMA), x)
    assert abs(mass - 1.0) < 1e-8


def test_analytical_moments_match_theory():
    x = np.linspace(-15, 20, 4001)
    p = analytical_density(x, T, MU, SIGMA)
    mean = np.trapz(x * p, x)
    var = np.trapz((x - mean) ** 2 * p, x)
    assert abs(mean - MU * T) < 1e-6          # mean = mu*t
    assert abs(var - SIGMA ** 2 * T) < 1e-4   # variance = sigma^2*t


def test_finite_difference_matches_analytical_when_stable():
    res = FiniteDifferenceSolver(MU, SIGMA).solve(T, dx=0.02, r=0.4)
    assert res.stable
    p_exact = analytical_density(res.x, T, MU, SIGMA)
    l2 = np.sqrt(np.trapezoid((res.p - p_exact) ** 2, res.x))
    assert l2 < 1e-3


def test_finite_difference_detects_instability():
    res = FiniteDifferenceSolver(MU, SIGMA).solve(T, dx=0.02, r=0.6)
    assert not res.stable                     # r > 0.5 must blow up


def test_monte_carlo_moments_converge():
    res = monte_carlo_simulation(200_000, T, MU, SIGMA, n_steps=100, seed=42)
    assert abs(res.empirical_mean - MU * T) < 0.02
    assert abs(res.empirical_var - SIGMA ** 2 * T) < 0.05


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
