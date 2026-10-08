"""Quick accuracy verification for all KROM benchmarks at rho=4.

Skips the dense kernel solve entirely by monkey-patching dense_precision_factor
to a no-op before any experiment module is loaded.  All accuracy numbers are
sparse KROM vs the FOM ground truth.

Usage:
    cd /path/to/ROM
    python experiments/quick_verify.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Patch dense_precision_factor before any experiment imports it ---
import src.krom.factors as _factors_mod

_SENTINEL = object()

class _DenseSkipped:
    """Silent no-op factor: dense GN runs return initial state (garbage accuracy),
    but don't crash. We never read dense results in the summary table."""
    def apply(self, x):          return x * 0
    def apply_transpose(self, x): return x * 0
    def apply_jacobian(self, J):  return J * 0

_factors_mod.dense_precision_factor = lambda theta, nugget=1e-10: _DenseSkipped()
# -------------------------------------------------------------------

import time
import torch

RHO = 4.0
print(f'KROM quick verification  (rho={RHO}, sparse only vs FOM)\n')
results = {}

# =============================================================================
# 1. Darcy Flow (2D, 25x25)
# =============================================================================
print('─' * 60)
print('1. Darcy Flow  (grid_size=25, N_int=529)')
from experiments.Darcy_Flow_clean import run_experiment as run_darcy
t0 = time.perf_counter()
r = run_darcy(grid_size=25, rho=RHO, num_train=12, num_test=10, gn_steps=3)
elapsed = time.perf_counter() - t0
print(f'   Empirical sparse  rel-L2 = {r["empirical_sparse"]["mean_rel_l2"]:.4f}')
print(f'   Matern    sparse  rel-L2 = {r["matern_sparse"]["mean_rel_l2"]:.4f}')
print(f'   Wall time: {elapsed:.1f}s\n')
results['darcy'] = r

# =============================================================================
# 2. Elliptic Nonlinear PDE (2D, 33x33)
# =============================================================================
print('─' * 60)
print('2. Elliptic Nonlinear PDE  (33x33, N_int=961)')
from experiments.Elliptic_Nonlinear_PDE_clean import run_experiment as run_elliptic
from experiments.Elliptic_Nonlinear_PDE_clean import DEFAULT_GRID_CONFIG
t0 = time.perf_counter()
r = run_elliptic(grid_config=DEFAULT_GRID_CONFIG, rho=RHO, num_snapshots=256, gn_steps=4)
elapsed = time.perf_counter() - t0
print(f'   Empirical sparse  rel-L2 = {r["empirical_sparse"]["mean_rel_l2"]:.4f}')
print(f'   Matern    sparse  rel-L2 = {r["matern_sparse"]["mean_rel_l2"]:.4f}')
print(f'   Wall time: {elapsed:.1f}s\n')
results['elliptic'] = r

# =============================================================================
# 3. Allen-Cahn (2D, 41x41)
# =============================================================================
print('─' * 60)
print('3. Allen-Cahn  (25x25, N_int=529)')
from experiments.Allen_Cahn_clean import run_experiment as run_ac
from src.krom.pde_baselines import AllenCahnCNConfig
AC_CFG_SMALL = AllenCahnCNConfig(nx=25, ny=25, dt=1e-2, tmax=0.5, epsilon=1e-2)
t0 = time.perf_counter()
r = run_ac(config=AC_CFG_SMALL, rho=RHO, num_train=8, num_test=4, gn_steps=2)
elapsed = time.perf_counter() - t0
print(f'   Empirical sparse  rel-L2 = {r["empirical_sparse"]["mean_final_rel_l2"]:.4f}')
print(f'   Matern    sparse  rel-L2 = {r["matern_sparse"]["mean_final_rel_l2"]:.4f}')
print(f'   Wall time: {elapsed:.1f}s\n')
results['allen_cahn'] = r

# =============================================================================
# 4. Burgers (1D, nx=2001)
# =============================================================================
print('─' * 60)
print('4. Burgers  (1D, nx=401, N_int=399)')
from experiments.Burgers_clean import run_experiment as run_burgers
from src.krom.pde_baselines import BurgersCNConfig
B_CFG_SMALL = BurgersCNConfig(nx=401, dt=0.02, tmax=1.0, viscosity=1e-3, newton_max_iter=20)
t0 = time.perf_counter()
r = run_burgers(config=B_CFG_SMALL, rho=RHO, num_train=10, num_test=6, gn_steps=3)
elapsed = time.perf_counter() - t0
print(f'   Empirical sparse  rel-L2 = {r["empirical_sparse"]["mean_final_rel_l2"]:.4f}')
print(f'   Matern    sparse  rel-L2 = {r["matern_sparse"]["mean_final_rel_l2"]:.4f}')
print(f'   Wall time: {elapsed:.1f}s\n')
results['burgers'] = r

# =============================================================================
# 5. Kolmogorov Flow (2D, nx=32)
# =============================================================================
print('─' * 60)
print('5. Kolmogorov Flow  (nx=32, N_int=1024)')
try:
    from experiments.Kolmogorov_ROM_Comparison import (
        KolmogorovConfig, ExperimentConfig, run_comparison
    )
    kol_cfg = KolmogorovConfig(nx=32)
    exp_cfg = ExperimentConfig(krom_rho_values=[RHO], pod_k_values=[], n_test=4)
    t0 = time.perf_counter()
    r = run_comparison(kol_cfg, exp_cfg, verbose=False)
    elapsed = time.perf_counter() - t0
    krom_err = r['krom'][RHO]['mean_final_err']
    print(f'   KROM sparse  mean rel-L2 = {krom_err:.4f}')
    print(f'   Wall time: {elapsed:.1f}s\n')
    results['kolmogorov'] = r
except Exception as e:
    print(f'   (skipped: {e})\n')

# =============================================================================
# Summary
# =============================================================================
print('=' * 60)
print(f'{"Benchmark":<28} {"Empirical sparse":>18} {"Matern sparse":>15}')
print('-' * 60)
for name, key, metric in [
    ('Darcy (25x25)',         'darcy',      'mean_rel_l2'),
    ('Elliptic (33x33)',      'elliptic',   'mean_rel_l2'),
    ('Allen-Cahn (25x25)',    'allen_cahn', 'mean_final_rel_l2'),
    ('Burgers (nx=401)',      'burgers',    'mean_final_rel_l2'),
]:
    if key in results:
        emp = results[key].get('empirical_sparse', {}).get(metric, float('nan'))
        mat = results[key].get('matern_sparse',    {}).get(metric, float('nan'))
        print(f'{name:<28} {emp:>18.4f} {mat:>15.4f}')
print('=' * 60)
