"""
Run all experiments for the presentation, store every run in SQLite,
print the presentation queries, and produce the three figures.

Usage:  python experiments.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fokker_planck import (analytical_density, FiniteDifferenceSolver,
                           monte_carlo_simulation)

# Experiment parameters -----------------------------------------------------
MU, SIGMA, T_FINAL = 1.5, 1.0, 2.0          # mean = 3.0, variance = 2.0 at T
DB_PATH = Path("results.db")
PLOTS = Path("plots"); PLOTS.mkdir(exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    method        TEXT NOT NULL,             -- 'monte_carlo' | 'finite_difference'
    mu            REAL, sigma REAL, t_final REAL,
    n_paths       INTEGER,                   -- MC only
    dt            REAL, dx REAL, r REAL,     -- FD only (r = D*dt/dx^2)
    empirical_mean  REAL, empirical_var REAL,
    analytical_mean REAL, analytical_var REAL,
    abs_mean_error  REAL, abs_var_error REAL,
    l2_error      REAL,
    stable        INTEGER,
    runtime_ms    REAL,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

QUERY_CONVERGENCE = """
-- Monte Carlo convergence: error should shrink like 1/sqrt(N).
-- Each 10x increase in N should multiply the error by ~0.316.
SELECT n_paths,
       ROUND(AVG(abs_mean_error), 5)                    AS avg_mean_err,
       ROUND(AVG(abs_mean_error) /
             LAG(AVG(abs_mean_error))
                 OVER (ORDER BY n_paths), 3)            AS ratio_vs_prev
FROM runs
WHERE method = 'monte_carlo'
GROUP BY n_paths
ORDER BY n_paths;
"""

QUERY_STABILITY = """
-- Finite-difference stability boundary: stable iff r = D*dt/dx^2 <= 0.5.
SELECT ROUND(r, 3)  AS r,
       ROUND(dt, 6) AS dt,
       stable,
       ROUND(l2_error, 6) AS l2_error
FROM runs
WHERE method = 'finite_difference'
ORDER BY r;
"""


def analytical_moments():
    return MU * T_FINAL, SIGMA ** 2 * T_FINAL


def run_monte_carlo(con):
    a_mean, a_var = analytical_moments()
    for n_paths in (100, 1_000, 10_000, 100_000):
        for rep in range(10):
            res = monte_carlo_simulation(n_paths, T_FINAL, MU, SIGMA,
                                         n_steps=100, seed=1000 * rep + n_paths)
            con.execute(
                "INSERT INTO runs (method, mu, sigma, t_final, n_paths,"
                " empirical_mean, empirical_var, analytical_mean, analytical_var,"
                " abs_mean_error, abs_var_error, runtime_ms)"
                " VALUES ('monte_carlo',?,?,?,?,?,?,?,?,?,?,?)",
                (MU, SIGMA, T_FINAL, n_paths,
                 res.empirical_mean, res.empirical_var, a_mean, a_var,
                 abs(res.empirical_mean - a_mean), abs(res.empirical_var - a_var),
                 res.runtime_ms))
    con.commit()


def run_finite_difference(con):
    a_mean, a_var = analytical_moments()
    solver = FiniteDifferenceSolver(MU, SIGMA)
    results = {}
    for r in (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6):
        res = solver.solve(T_FINAL, dx=0.02, r=r)
        p_exact = analytical_density(res.x, T_FINAL, MU, SIGMA)
        diff = res.p - p_exact
        l2 = float(np.sqrt(np.trapezoid(diff ** 2, res.x)))
        l2_store = l2 if np.isfinite(l2) else None
        emp_mean = float(np.trapezoid(res.x * res.p, res.x)) if res.stable else None
        emp_var = (float(np.trapezoid((res.x - emp_mean) ** 2 * res.p, res.x))
                   if res.stable else None)
        con.execute(
            "INSERT INTO runs (method, mu, sigma, t_final, dt, dx, r,"
            " empirical_mean, empirical_var, analytical_mean, analytical_var,"
            " l2_error, stable, runtime_ms)"
            " VALUES ('finite_difference',?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (MU, SIGMA, T_FINAL, res.dt, res.dx, res.r,
             emp_mean, emp_var, a_mean, a_var, l2_store,
             int(res.stable), res.runtime_ms))
        results[r] = res
    con.commit()
    return results


def make_plots(fd_results):
    x_fine = np.linspace(-4, 10, 1000)
    p_exact = analytical_density(x_fine, T_FINAL, MU, SIGMA)

    # --- Figure 1: three independent methods agree -------------------------
    mc = monte_carlo_simulation(100_000, T_FINAL, MU, SIGMA, n_steps=100, seed=7)
    fd = fd_results[0.4]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mc.samples, bins=80, density=True, alpha=0.35,
            label="Monte Carlo (100k paths)", color="tab:green")
    ax.plot(fd.x, fd.p, "--", lw=2.2, label="Finite difference (r=0.4)",
            color="tab:orange")
    ax.plot(x_fine, p_exact, lw=2.2, label="Analytical  N($\\mu$t, $\\sigma^2$t)",
            color="tab:blue")
    ax.set(xlabel="x", ylabel="p(x, T)",
           title=f"Three independent solutions, $\\mu$={MU}, $\\sigma$={SIGMA}, T={T_FINAL}")
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS / "three_way_agreement.png", dpi=150); plt.close(fig)

    # --- Figure 2: Monte Carlo convergence ~ 1/sqrt(N) ---------------------
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT n_paths, AVG(abs_mean_error) FROM runs"
        " WHERE method='monte_carlo' GROUP BY n_paths ORDER BY n_paths").fetchall()
    con.close()
    n = np.array([row[0] for row in rows], dtype=float)
    err = np.array([row[1] for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n, err, "o-", lw=2, label="Observed |mean error|")
    ax.loglog(n, err[0] * np.sqrt(n[0] / n), "k--", alpha=0.6,
              label="1/$\\sqrt{N}$ reference")
    ax.set(xlabel="Number of paths N", ylabel="Absolute error of the mean",
           title="Monte Carlo error falls like 1/$\\sqrt{N}$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    fig.savefig(PLOTS / "mc_convergence.png", dpi=150); plt.close(fig)

    # --- Figure 3: the stability demonstration -----------------------------
    stable, unstable = fd_results[0.4], fd_results[0.6]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    axes[0].plot(stable.x, stable.p, color="tab:blue")
    axes[0].plot(x_fine, p_exact, "k--", alpha=0.5)
    axes[0].set(title=f"r = {stable.r:.2f}  (stable)", xlabel="x", ylabel="p(x, T)")
    axes[1].plot(unstable.x, unstable.p, color="tab:red")
    axes[1].set(title=f"r = {unstable.r:.2f}  (unstable: max|p| = "
                      f"{np.max(np.abs(unstable.p)):.1e})", xlabel="x")
    fig.suptitle("Same scheme, same physics - only the timestep changed"
                 "  (stability: r = D$\\Delta$t/$\\Delta$x$^2$ $\\leq$ 1/2)")
    fig.tight_layout()
    fig.savefig(PLOTS / "stability_demo.png", dpi=150); plt.close(fig)


def print_query(con, title, sql):
    print(f"\n=== {title} ===")
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    print(" | ".join(cols))
    for row in cur.fetchall():
        print(" | ".join(str(v) for v in row))


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    print("Running Monte Carlo sweeps...")
    run_monte_carlo(con)
    print("Running finite-difference sweeps (incl. deliberate instability)...")
    fd_results = run_finite_difference(con)
    print("Rendering figures...")
    make_plots(fd_results)

    print_query(con, "Monte Carlo convergence (window function)", QUERY_CONVERGENCE)
    print_query(con, "Finite-difference stability boundary", QUERY_STABILITY)
    con.close()
    print(f"\nDone. Database: {DB_PATH}  |  Figures: {PLOTS}/")


if __name__ == "__main__":
    main()
