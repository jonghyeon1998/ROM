#!/usr/bin/env python3
"""
run_all_paper_experiments.py
============================
Master script that runs all five KROM paper experiments sequentially,
saves per-experiment sweep figures, and writes a combined summary figure.

Usage:
    python experiments/paper/run_all_paper_experiments.py

Each experiment produces:
  - N-sweep plot  (error vs number of training trajectories)
  - ρ-sweep plot  (error vs sparsity radius, fixed N)

Results are saved to:
  experiments/paper/results/
    ├── exp1_elliptic_sweeps.png
    ├── exp2_darcy_sweeps.png
    ├── exp3_burgers_sweeps.png
    ├── exp4_allen_cahn_sweeps.png
    ├── exp5_ns_sweeps.png
    └── paper_summary.png

To run a quick smoke test (reduced N and ρ ranges), set QUICK=True below.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Quick-mode flag ──────────────────────────────────────────────────────────
# Set QUICK=True for a fast smoke test (small N/ρ ranges, coarser grids).
QUICK = True

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Imports (deferred so we can run individual experiments standalone)
# ---------------------------------------------------------------------------
def _run_exp1():
    import experiments.paper.paper_exp1_elliptic as e1
    N_vals  = [10, 50, 100, 200] if QUICK else e1.N_SWEEP
    rho_vals = [2.0, 4.0, 6.0] if QUICK else e1.RHO_SWEEP
    print('\n' + '='*60)
    print('EXP 1: Semilinear Elliptic')
    print('='*60)
    t0 = time.perf_counter()
    n_res  = e1.run_n_sweep(N_values=N_vals, verbose=True)
    rho_res = e1.run_rho_sweep(rho_values=rho_vals, verbose=True)
    fig = e1.plot_sweeps(n_res, rho_res, save_dir=RESULTS_DIR)
    plt.close(fig)
    print(f'  Done in {time.perf_counter()-t0:.0f}s')
    return n_res, rho_res


def _run_exp2():
    import experiments.paper.paper_exp2_darcy as e2
    N_vals  = [10, 50, 100, 200] if QUICK else e2.N_SWEEP
    rho_vals = [2.0, 4.0, 6.0] if QUICK else e2.RHO_SWEEP
    print('\n' + '='*60)
    print('EXP 2: Darcy Flow')
    print('='*60)
    t0 = time.perf_counter()
    n_res  = e2.run_n_sweep(N_values=N_vals, verbose=True)
    rho_res = e2.run_rho_sweep(rho_values=rho_vals, verbose=True)
    fig = e2.plot_sweeps(n_res, rho_res, save_dir=RESULTS_DIR)
    plt.close(fig)
    print(f'  Done in {time.perf_counter()-t0:.0f}s')
    return n_res, rho_res


def _run_exp3():
    import experiments.paper.paper_exp3_burgers as e3
    N_vals  = [10, 40, 80] if QUICK else e3.N_SWEEP
    rho_vals = [2.0, 5.0, 8.0] if QUICK else e3.RHO_SWEEP
    print('\n' + '='*60)
    print('EXP 3: Burgers 1D')
    print('='*60)
    t0 = time.perf_counter()
    n_res  = e3.run_n_sweep(N_values=N_vals, verbose=True)
    rho_res = e3.run_rho_sweep(rho_values=rho_vals, verbose=True)
    fig = e3.plot_sweeps(n_res, rho_res, save_dir=RESULTS_DIR)
    plt.close(fig)
    print(f'  Done in {time.perf_counter()-t0:.0f}s')
    return n_res, rho_res


def _run_exp4():
    import experiments.paper.paper_exp4_allen_cahn as e4
    N_vals  = [10, 50, 100, 200] if QUICK else e4.N_SWEEP
    rho_vals = [2.0, 5.0, 8.0] if QUICK else e4.RHO_SWEEP
    print('\n' + '='*60)
    print('EXP 4: Allen-Cahn 2D')
    print('='*60)
    t0 = time.perf_counter()
    n_res  = e4.run_n_sweep(N_values=N_vals, verbose=True)
    rho_res = e4.run_rho_sweep(rho_values=rho_vals, verbose=True)
    fig = e4.plot_sweeps(n_res, rho_res, save_dir=RESULTS_DIR)
    plt.close(fig)
    print(f'  Done in {time.perf_counter()-t0:.0f}s')
    return n_res, rho_res


def _run_exp5():
    import experiments.paper.paper_exp5_navier_stokes as e5
    N_vals  = [10, 50, 100, 200] if QUICK else e5.N_SWEEP
    rho_vals = [2.0, 5.0, 8.0] if QUICK else e5.RHO_SWEEP
    print('\n' + '='*60)
    print('EXP 5: Navier-Stokes 2D')
    print('='*60)
    t0 = time.perf_counter()
    n_res  = e5.run_n_sweep(N_values=N_vals, verbose=True)
    rho_res = e5.run_rho_sweep(rho_values=rho_vals, verbose=True)
    fig = e5.plot_sweeps(n_res, rho_res, save_dir=RESULTS_DIR)
    plt.close(fig)
    print(f'  Done in {time.perf_counter()-t0:.0f}s')
    return n_res, rho_res


# ---------------------------------------------------------------------------
# Combined summary figure (5 experiments × N-sweep, matches paper layout)
# ---------------------------------------------------------------------------
def make_summary_figure(all_n_results: dict, all_rho_results: dict):
    titles = {
        'exp1': 'Elliptic',
        'exp2': 'Darcy',
        'exp3': 'Burgers',
        'exp4': 'Allen-Cahn',
        'exp5': 'NS 2D',
    }
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for col, (key, label) in enumerate(titles.items()):
        nr  = all_n_results[key]
        rr  = all_rho_results[key]
        x_n   = nr['N']
        x_rho = rr['rho']

        ax = axes[0, col]
        ax.semilogy(x_n, nr['empirical_sparse'], 'o-', ms=5, color='steelblue',  label='Empirical')
        ax.semilogy(x_n, nr['matern_sparse'],    's--', ms=5, color='darkorange', label='Matérn')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('N'); ax.set_ylabel('Rel. L2 err' if col == 0 else '')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if col == 0:
            axes[0, 0].set_ylabel('Rel. L2 error  (N sweep)')

        ax = axes[1, col]
        ax.semilogy(x_rho, rr['empirical_sparse'], 'o-', ms=5, color='steelblue',  label='Empirical')
        ax.semilogy(x_rho, rr['matern_sparse'],    's--', ms=5, color='darkorange', label='Matérn')
        ax.set_xlabel('ρ')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if col == 0:
            axes[1, 0].set_ylabel('Rel. L2 error  (ρ sweep)')

    fig.suptitle('KROM Paper Experiments — N sweep (top) and ρ sweep (bottom)', fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / 'paper_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSummary figure saved to {out}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    total_t0 = time.perf_counter()

    all_n   = {}
    all_rho = {}

    all_n['exp1'], all_rho['exp1'] = _run_exp1()
    all_n['exp2'], all_rho['exp2'] = _run_exp2()
    all_n['exp3'], all_rho['exp3'] = _run_exp3()
    all_n['exp4'], all_rho['exp4'] = _run_exp4()
    all_n['exp5'], all_rho['exp5'] = _run_exp5()

    fig = make_summary_figure(all_n, all_rho)

    print(f'\nAll experiments done in {(time.perf_counter()-total_t0)/60:.1f} min')
    print(f'Results in {RESULTS_DIR}')

    plt.show()
