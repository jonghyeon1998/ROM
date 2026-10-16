# %%
"""
Paper Experiment 3 — Burgers' Equation (1D)
============================================
Matches Section 3.3 / Table 1 of KROM (Bacho, Lee, Owhadi, 2026).

PDE:    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²  on [−1,1]×[0,1], u(±1,t)=0
ν:      0.001
M:      2001 collocation points (including boundaries)
N:      sweep 10,20,...,80 training trajectories
        Paper: 8 base ICs × 10 spatial shifts = 80; we sample 80 directly.
FOM:    Crank-Nicolson with Δt=0.04 (same as KROM rollout; paper used WENO,
        replaced here with CN for efficiency)
ρ:      5 (default), sweep ρ ∈ {2,4,6,8,10}
Test:   u(x,0) = −sin(πx)
Kernel: empirical K_N (sparse) and Matérn-5/2 (θ=0.05)
GN:     2 steps
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.empirical import temporal_feature_matrix
from src.krom.factors import dense_precision_factor  # used only when with_dense=True
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import BurgersCNConfig, burgers_crank_nicolson_rollout
from src.krom.sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from src.krom.workflows import (
    build_burgers_empirical_theta,
    build_burgers_matern_theta,
    burgers_residual_and_jacobian,
    burgers_residual_operator,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Paper parameters (Table 1, Section 3.3)
# ---------------------------------------------------------------------------
PAPER_CONFIG = BurgersCNConfig(
    x_start=-1.0, x_end=1.0,
    nx=2001, dt=0.04, tmax=1.0,
    viscosity=1e-3, newton_max_iter=20,
)
RHO_DEFAULT  = 5.0
RHO_SWEEP    = [2.0, 4.0, 6.0, 8.0, 10.0]
N_SWEEP      = list(range(10, 81, 10))   # 10,20,...,80
N_FIXED_RHO  = 80
N_TRAIN_MAX  = 80
N_TEST       = 10
MATERN_LS    = 0.05
GN_STEPS     = 2
GN_DAMPING   = 1e-8
NUGGET       = 1e-9
AUTO_DIRECT_THRESHOLD = 8192   # Burgers state = 2×1999 = 3998; must exceed this for direct solve


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-30))


# ---------------------------------------------------------------------------
# IC generation matching paper Section 3.3
# 8 base trig ICs × 10 spatial shifts of Δs=0.2 in [−0.8,0.8]
# Base ICs: u₀(x) = Σᵢ aᵢ (cos(bᵢπξ) + sin(bᵢπξ)), ξ=(x+1)/2 ∈ [0,1]
# Shifted: u₀^(s)(x,0) = u₀(x − s, 0)  evaluated on [−1,1] with zero BC
# ---------------------------------------------------------------------------
def _trig_ic(x: torch.Tensor, seed_offset: int, num_terms: int = 4) -> torch.Tensor:
    """Random trig IC on ξ=(x+1)/2 ∈ [0,1], zero at x=±1."""
    torch.manual_seed(seed_offset)
    xi = (x + 1.0) / 2.0   # map [−1,1] → [0,1]
    u  = torch.zeros_like(x)
    b_vals = torch.randint(1, 6, (num_terms,)).tolist()
    a_vals = torch.randn(num_terms).tolist()
    for a, b in zip(a_vals, b_vals):
        u += a * (torch.cos(b * math.pi * xi) + torch.sin(b * math.pi * xi))
    # Enforce zero BC
    u[0] = 0.0; u[-1] = 0.0
    # Normalize amplitude
    scale = u.abs().max().clamp(min=1e-8)
    return 0.5 * u / scale


def generate_training_ics(config: BurgersCNConfig, seed: int = 0) -> list[torch.Tensor]:
    """
    Generate 80 training ICs:
      8 base trig ICs × 10 spatial shifts s ∈ {−0.8,−0.6,...,0.6,0.8}.
    Shifted IC: u_s(x,0) = u_base(x−s,0) interpolated onto grid, zero BC.
    """
    x = torch.linspace(config.x_start, config.x_end, config.nx, dtype=torch.float64)
    shifts = torch.linspace(-0.8, 0.8, 10)
    ics = []
    for base_idx in range(8):
        base_ic = _trig_ic(x, seed * 100 + base_idx)
        for s in shifts:
            xs = float(s)
            # Interpolate base_ic(x - xs) → evaluate base IC at shifted points
            x_src = x - xs
            ic = torch.zeros_like(x)
            in_range = (x_src >= config.x_start) & (x_src <= config.x_end)
            if in_range.any():
                # Linear interpolation
                frac = (x_src[in_range] - config.x_start) / (config.x_end - config.x_start)
                idx_f = frac * (config.nx - 1)
                idx_lo = idx_f.long().clamp(0, config.nx - 2)
                idx_hi = (idx_lo + 1).clamp(0, config.nx - 1)
                w = idx_f - idx_lo.float()
                ic[in_range] = base_ic[idx_lo] * (1.0 - w) + base_ic[idx_hi] * w
            ic[0] = 0.0; ic[-1] = 0.0
            ics.append(ic)
    return ics   # 80 tensors of shape (2001,)


def test_ic_sin(config: BurgersCNConfig) -> torch.Tensor:
    """Paper test IC: u(x,0) = −sin(πx)."""
    x = torch.linspace(config.x_start, config.x_end, config.nx, dtype=torch.float64)
    return -torch.sin(math.pi * x)


# ---------------------------------------------------------------------------
# Generate dataset for KROM (u, ux, uxx at each time step)
# ---------------------------------------------------------------------------
def rollout_fom(ic: torch.Tensor, config: BurgersCNConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Crank-Nicolson FOM, return (u, ux, uxx) each of shape (nx, n_steps).

    burgers_crank_nicolson_rollout returns (x, times, snapshots, grad_snapshots, lap_snapshots).
    """
    _x, _t, u, ux, uxx = burgers_crank_nicolson_rollout(ic, config)
    return u, ux, uxx


def generate_dataset(ics: list[torch.Tensor], config: BurgersCNConfig) -> dict:
    """Run FOM for each IC, collect trajectory tensors."""
    n_traj  = len(ics)
    n_steps = int(round(config.tmax / config.dt)) + 1
    nx      = config.nx
    u_all   = torch.zeros(n_traj, nx, n_steps, dtype=torch.float64)
    ux_all  = torch.zeros_like(u_all)
    uxx_all = torch.zeros_like(u_all)
    for i, ic in enumerate(ics):
        u_all[i], ux_all[i], uxx_all[i] = rollout_fom(ic, config)
    return {'u': u_all, 'ux': ux_all, 'uxx': uxx_all}


# ---------------------------------------------------------------------------
# KROM factor construction
# ---------------------------------------------------------------------------
def build_krom_factors(
    data: dict,
    config: BurgersCNConfig,
    rho: float,
    k_neighbors: int = 3,
    with_dense: bool = False,
) -> dict:
    x = torch.linspace(config.x_start, config.x_end, config.nx, dtype=torch.float64)
    n_int = config.nx - 2
    interior = x[1:-1, None]
    boundary = torch.tensor([[x[0]], [x[-1]]], dtype=torch.float64)
    dirac_pts = torch.cat([interior, boundary], dim=0)
    dg = (torch.arange(n_int, dtype=torch.long),
          torch.arange(n_int, dtype=torch.long))
    ordering = build_measurement_ordering(dirac_pts, dg, k_neighbors=k_neighbors)

    # Feature matrices: (n_total, M) where M = n_train × n_steps
    dirac_snaps = torch.cat([data['u'][:, 1:-1, :],
                              data['u'][:, [0, -1], :]], dim=1)   # (N, nx, T)
    dirac_feat  = temporal_feature_matrix(dirac_snaps)             # (nx, N*T)
    ux_feat     = temporal_feature_matrix(data['ux'][:, 1:-1, :])  # (n_int, N*T)
    uxx_feat    = temporal_feature_matrix(data['uxx'][:, 1:-1, :]) # (n_int, N*T)
    M = dirac_feat.shape[1]

    emp_ks    = make_snapshot_kernel_source((dirac_feat, ux_feat, uxx_feat), nugget=NUGGET, scale=1.0 / M)
    mat_asm   = build_burgers_matern_theta(interior, boundary, lengthscale=MATERN_LS, nugget=NUGGET)

    bv = torch.zeros(2, dtype=torch.float64)
    emp_sparse, _ = sparse_precision_factor(
        theta=emp_ks, dirac_points=dirac_pts, derivative_point_groups=dg,
        rho=rho, nugget=NUGGET, ordering=ordering)
    mat_sparse, _ = sparse_precision_factor(
        theta=mat_asm.theta, dirac_points=mat_asm.dirac_points,
        derivative_point_groups=mat_asm.derivative_point_groups,
        rho=rho, nugget=NUGGET, ordering=ordering)

    factors = {
        'empirical_sparse': emp_sparse,
        'matern_sparse':    mat_sparse,
        'boundary_values':  bv,
    }

    if with_dense:
        # Dense factors are expensive (Cholesky of ~6000×6000 matrix) — skip by default.
        # The paper reports sparse variants only.
        emp_theta = build_burgers_empirical_theta(dirac_feat, ux_feat, uxx_feat, nugget=NUGGET) / M
        factors['empirical_dense'] = dense_precision_factor(emp_theta, nugget=NUGGET)
        factors['matern_dense']    = dense_precision_factor(mat_asm.theta, nugget=NUGGET)

    return factors


# ---------------------------------------------------------------------------
# KROM rollout (re-uses Burgers_clean.py's pattern)
# ---------------------------------------------------------------------------
def rollout_krom(
    u0: torch.Tensor, ux0: torch.Tensor, uxx0: torch.Tensor,
    factor, config: BurgersCNConfig,
) -> torch.Tensor:
    n_int   = config.nx - 2
    n_steps = int(round(config.tmax / config.dt)) + 1
    rollout = torch.zeros(n_int, n_steps, dtype=torch.float64)
    bv      = torch.zeros(2, dtype=torch.float64)

    prev_u   = u0[1:-1].clone()
    prev_ux  = ux0[1:-1].clone()
    prev_uxx = uxx0[1:-1].clone()
    state    = torch.cat([prev_u, prev_ux], dim=0)
    rollout[:, 0] = prev_u

    for step in range(n_steps - 1):
        su, sux, suxx = prev_u.clone(), prev_ux.clone(), prev_uxx.clone()
        res = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: burgers_residual_operator(
                z, su, sux, suxx, bv, config.viscosity, config.dt),
            factor=factor,
            max_iter=GN_STEPS,
            damping=GN_DAMPING,
            record_history=False,
            linear_solver='auto',
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: burgers_residual_and_jacobian(
                z, su, sux, suxx, bv, config.viscosity, config.dt),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state   = res.state
        prev_u  = state[:n_int].clone()
        prev_ux = state[n_int:].clone()
        dx = (config.x_end - config.x_start) / (config.nx - 1)
        # Recompute uxx from updated u via FD (pad zeros at boundary)
        u_full = torch.cat([torch.zeros(1), prev_u, torch.zeros(1)])
        prev_uxx = (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / dx**2
        rollout[:, step + 1] = prev_u
    return rollout


def evaluate_factor(
    factor, test_u: torch.Tensor, test_ux: torch.Tensor, test_uxx: torch.Tensor,
    config: BurgersCNConfig,
) -> dict:
    n_int   = config.nx - 2
    n_test  = test_u.shape[0]
    errors, times = [], []
    for i in range(n_test):
        u0  = test_u[i, :, 0]
        ux0 = test_ux[i, :, 0]
        uxx0 = test_uxx[i, :, 0]
        t0 = time.perf_counter()
        pred = rollout_krom(u0, ux0, uxx0, factor, config)
        times.append(time.perf_counter() - t0)
        errors.append(relative_l2(pred[:, -1], test_u[i, 1:-1, -1]))
    return {'mean_rel_l2': sum(errors) / len(errors), 'mean_seconds': sum(times) / len(times)}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def run_n_sweep(
    N_values: Sequence[int] = N_SWEEP,
    rho: float = RHO_DEFAULT,
    config: BurgersCNConfig = PAPER_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 0,
    test_seed: int = 1,
    verbose: bool = True,
) -> dict:
    # Build all 80 training ICs once
    train_ics = generate_training_ics(config, seed=train_seed)
    if verbose:
        print(f'  Generating {len(train_ics)} training FOM trajectories...')
    train_data = generate_dataset(train_ics, config)

    # Test trajectories
    torch.manual_seed(test_seed)
    test_ics_list = [test_ic_sin(config)] + [
        _trig_ic(torch.linspace(-1,1,config.nx,dtype=torch.float64), test_seed*50+i)
        for i in range(n_test - 1)
    ]
    test_data = generate_dataset(test_ics_list, config)

    results = {'N': [], 'empirical_sparse': [], 'matern_sparse': []}
    for N in N_values:
        if verbose:
            print(f'  N={N}...', end=' ', flush=True)
        sub = {k: v[:N] for k, v in train_data.items()}
        factors = build_krom_factors(sub, config, rho)
        t0 = time.perf_counter()
        results['N'].append(N)
        for key in ('empirical_sparse', 'matern_sparse'):
            ev = evaluate_factor(factors[key], test_data['u'], test_data['ux'], test_data['uxx'], config)
            results[key].append(ev['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    return results


def run_rho_sweep(
    rho_values: Sequence[float] = RHO_SWEEP,
    N_fixed: int = N_FIXED_RHO,
    config: BurgersCNConfig = PAPER_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 0,
    test_seed: int = 1,
    verbose: bool = True,
) -> dict:
    train_ics  = generate_training_ics(config, seed=train_seed)
    train_data = generate_dataset(train_ics[:N_fixed], config)
    test_ics_list = [test_ic_sin(config)] + [
        _trig_ic(torch.linspace(-1,1,config.nx,dtype=torch.float64), test_seed*50+i)
        for i in range(n_test - 1)
    ]
    test_data = generate_dataset(test_ics_list, config)

    results = {'rho': [], 'empirical_sparse': [], 'matern_sparse': []}
    for rho in rho_values:
        if verbose:
            print(f'  rho={rho}...', end=' ', flush=True)
        factors = build_krom_factors(train_data, config, rho, with_dense=False)
        results['rho'].append(rho)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(
                evaluate_factor(factors[key], test_data['u'], test_data['ux'], test_data['uxx'], config)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}")
    return results


def plot_sweeps(n_res: dict, rho_res: dict, save_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(n_res['N'], n_res['empirical_sparse'], 'o-', label='KROM (empirical)', color='steelblue')
    ax.semilogy(n_res['N'], n_res['matern_sparse'],    's--', label='KROM (Matérn)',   color='darkorange')
    ax.set_xlabel('N (training trajectories)'); ax.set_ylabel('Relative L2 error (t=T)')
    ax.set_title('Exp 3: Burgers  —  N sweep  (M=2001, ρ=5)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(rho_res['rho'], rho_res['empirical_sparse'], 'o-', label='KROM (empirical)', color='steelblue')
    ax.semilogy(rho_res['rho'], rho_res['matern_sparse'],    's--', label='KROM (Matérn)',   color='darkorange')
    ax.set_xlabel('ρ'); ax.set_ylabel('Relative L2 error (t=T)')
    ax.set_title(f'Exp 3: Burgers  —  ρ sweep  (N={N_FIXED_RHO})'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / 'exp3_burgers_sweeps.png', dpi=150, bbox_inches='tight')
    return fig


# %%
if __name__ == '__main__':
    print('=' * 60)
    print('Experiment 3: Burgers 1D (paper Table 1)')
    print(f'  M=2001, ν=1e-3, Δt=0.04, T=1, N sweep {N_SWEEP}')
    print(f'  80 training ICs (8 base × 10 shifts)')
    print('=' * 60)
    print('\n--- N sweep (ρ=5) ---')
    n_results   = run_n_sweep(verbose=True)
    print('\n--- ρ sweep (N=80) ---')
    rho_results = run_rho_sweep(verbose=True)
    out = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    plot_sweeps(n_results, rho_results, save_dir=out)
    plt.show()
