# %%
"""
Paper Experiment 4 — Allen-Cahn Equation (2D)
===============================================
Matches Section 3.4 / Table 1 of KROM (Bacho, Lee, Owhadi, 2026).

PDE:    ∂u/∂t = ε²Δu − (u³−u)  on Ω=(0,2π)²×[0,5], u=0 on ∂Ω
ε:      0.01
M:      51² = 2601 collocation points (all grid nodes including boundary)
N:      sweep 10,20,...,200 training trajectories
ICs:    u(x,y,0) = Σ_{n=1}^5 Σ_{m=1}^5 a_{nm} sin(nx) sin(my), a_{nm} ~ N(0,1)
Test:   u(x,y,0) = 0.25 sin(3x) sin(3y)
FOM:    Crank-Nicolson (replaces paper's FTCS — more efficient, same structure)
Δt:     0.05, T=5 → 100 steps
ρ:      5 (default), sweep ρ ∈ {2,4,6,8,10}
Kernel: empirical K_N (sparse) and Matérn-5/2 (θ=0.30)
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
from src.krom.factors import dense_precision_factor
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import (
    AllenCahnCNConfig,
    build_allen_cahn_flattened_coordinates,
    solve_allen_cahn_crank_nicolson,
)
from src.krom.sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from src.krom.workflows import (
    allen_cahn_residual_and_jacobian,
    allen_cahn_residual_operator,
    build_allen_cahn_empirical_theta,
    build_elliptic_matern_theta,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Paper parameters (Table 1, Section 3.4)
# ---------------------------------------------------------------------------
PAPER_CONFIG = AllenCahnCNConfig(
    epsilon=0.01,
    domain_length=2.0 * math.pi,   # (0,2π)² — matches AllenCahnCNConfig default
    nx=51, ny=51,                  # M = 51² = 2601 collocation points (paper Table 1)
    dt=0.05, tmax=5.0,
    newton_tol=1e-10, newton_max_iter=25,
)
RHO_DEFAULT  = 5.0
RHO_SWEEP    = [2.0, 4.0, 6.0, 8.0, 10.0]
N_SWEEP      = list(range(10, 201, 10))
N_FIXED_RHO  = 200
N_TRAIN_MAX  = 200
N_TEST       = 5
MATERN_LS    = 0.30
GN_STEPS     = 2
GN_DAMPING   = 1e-8
NUGGET       = 1e-9
AUTO_DIRECT_THRESHOLD = 4096   # Allen-Cahn interior state = 49²=2401 > 2048; must exceed this


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-30))


# ---------------------------------------------------------------------------
# IC generation matching paper Section 3.4
# u(x,y,0) = Σ_{n=1}^5 Σ_{m=1}^5 a_{nm} sin(nx) sin(my)
# Automatically satisfies u=0 at x=0,2π and y=0,2π
# ---------------------------------------------------------------------------
def paper_allen_cahn_ic(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    seed: int,
    max_modes: int = 5,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Random IC on (0,2π)² satisfying Dirichlet BC (sin modes vanish at 0 and 2π)."""
    torch.manual_seed(seed)
    a = torch.randn(max_modes, max_modes, dtype=torch.float64)
    u = torch.zeros_like(x_grid)
    for n in range(1, max_modes + 1):
        for m in range(1, max_modes + 1):
            u += a[n - 1, m - 1] * torch.sin(n * x_grid) * torch.sin(m * y_grid)
    # Rescale so max |u| ≤ amplitude
    scale = u.abs().max().clamp(min=1e-8)
    return amplitude * u / scale


def paper_test_ic(x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
    """Paper test IC: u₀(x,y) = 0.25 sin(3x) sin(3y)."""
    return 0.25 * torch.sin(3.0 * x_grid) * torch.sin(3.0 * y_grid)


# ---------------------------------------------------------------------------
# Build grid (x_grid, y_grid) for IC evaluation
# ---------------------------------------------------------------------------
def _make_grids(config: AllenCahnCNConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, config.domain_length, config.nx, dtype=torch.float64)
    y = torch.linspace(0.0, config.domain_length, config.ny, dtype=torch.float64)
    return torch.meshgrid(x, y, indexing='xy')   # (nx, ny) each


# ---------------------------------------------------------------------------
# Flatten / unflatten (interior-first, boundary last)
# ---------------------------------------------------------------------------
def flatten_with_boundary_last(u_grid: torch.Tensor) -> torch.Tensor:
    interior = u_grid[1:-1, 1:-1].reshape(-1)
    top      = u_grid[0, :]
    bot      = u_grid[-1, :]
    left     = u_grid[1:-1, 0]
    right    = u_grid[1:-1, -1]
    return torch.cat([interior, top, bot, left, right])


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def run_fom(ic_grid: torch.Tensor, config: AllenCahnCNConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Crank-Nicolson rollout; returns (flattened solutions, flattened laplacians)."""
    sol, lap = solve_allen_cahn_crank_nicolson(ic_grid, config)
    n_steps = len(sol)
    coords   = build_allen_cahn_flattened_coordinates(config)
    n_flat   = coords.shape[0]
    sol_flat = torch.zeros(n_flat, n_steps, dtype=torch.float64)
    lap_flat = torch.zeros(n_flat, n_steps, dtype=torch.float64)
    for t in range(n_steps):
        sol_flat[:, t] = flatten_with_boundary_last(sol[t])
        lap_flat[:, t] = flatten_with_boundary_last(lap[t])
    return sol_flat, lap_flat


def generate_dataset(config: AllenCahnCNConfig, num_samples: int, seed: int) -> dict:
    x_g, y_g = _make_grids(config)
    n_steps = int(round(config.tmax / config.dt)) + 1
    coords  = build_allen_cahn_flattened_coordinates(config)
    n_flat  = coords.shape[0]
    sol_all = torch.zeros(num_samples, n_flat, n_steps, dtype=torch.float64)
    lap_all = torch.zeros_like(sol_all)
    for i in range(num_samples):
        ic = paper_allen_cahn_ic(x_g, y_g, seed=seed * 1000 + i)
        s, l = run_fom(ic, config)
        sol_all[i] = s; lap_all[i] = l
    return {'solutions': sol_all, 'laplacians': lap_all}


def generate_test_dataset(config: AllenCahnCNConfig, n_test: int, seed: int) -> dict:
    """First test case = paper's canonical IC 0.25sin(3x)sin(3y); rest are random."""
    x_g, y_g = _make_grids(config)
    n_steps = int(round(config.tmax / config.dt)) + 1
    coords  = build_allen_cahn_flattened_coordinates(config)
    n_flat  = coords.shape[0]
    sol_all = torch.zeros(n_test, n_flat, n_steps, dtype=torch.float64)
    lap_all = torch.zeros_like(sol_all)
    # First test: paper's canonical IC
    ic0 = paper_test_ic(x_g, y_g)
    s, l = run_fom(ic0, config)
    sol_all[0] = s; lap_all[0] = l
    # Additional random tests
    for i in range(1, n_test):
        ic = paper_allen_cahn_ic(x_g, y_g, seed=seed * 1000 + i)
        s, l = run_fom(ic, config)
        sol_all[i] = s; lap_all[i] = l
    return {'solutions': sol_all, 'laplacians': lap_all}


# ---------------------------------------------------------------------------
# KROM factor construction
# ---------------------------------------------------------------------------
def build_krom_factors(
    train: dict,
    config: AllenCahnCNConfig,
    rho: float,
    k_neighbors: int = 3,
    with_dense: bool = False,
) -> dict:
    coords  = build_allen_cahn_flattened_coordinates(config)
    n_int   = (config.nx - 2) * (config.ny - 2)
    dg      = (torch.arange(n_int, dtype=torch.long),)
    bv      = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)
    ordering = build_measurement_ordering(coords, dg, k_neighbors=k_neighbors)

    sol_feat = temporal_feature_matrix(train['solutions'])           # (n_flat, N*T)
    lap_feat = temporal_feature_matrix(train['laplacians'][:, :n_int, :])  # (n_int, N*T)
    M_feat   = sol_feat.shape[1]

    emp_ks    = make_snapshot_kernel_source((sol_feat, lap_feat), nugget=NUGGET, scale=1.0 / M_feat)
    mat_asm   = build_elliptic_matern_theta(
        coords[:n_int], coords[n_int:], lengthscale=MATERN_LS, nugget=NUGGET)

    emp_sparse, _ = sparse_precision_factor(
        theta=emp_ks, dirac_points=coords, derivative_point_groups=dg,
        rho=rho, nugget=NUGGET, ordering=ordering)
    mat_sparse, _ = sparse_precision_factor(
        theta=mat_asm.theta, dirac_points=mat_asm.dirac_points,
        derivative_point_groups=mat_asm.derivative_point_groups,
        rho=rho, nugget=NUGGET, ordering=ordering)

    factors = {
        'empirical_sparse': emp_sparse,
        'matern_sparse':    mat_sparse,
        'boundary_values': bv, 'n_int': n_int,
    }
    if with_dense:
        emp_theta = build_allen_cahn_empirical_theta(sol_feat, lap_feat, nugget=NUGGET) / M_feat
        factors['empirical_dense'] = dense_precision_factor(emp_theta, nugget=NUGGET)
        factors['matern_dense']    = dense_precision_factor(mat_asm.theta, nugget=NUGGET)
    return factors


# ---------------------------------------------------------------------------
# KROM rollout
# ---------------------------------------------------------------------------
def rollout_krom(
    sol0: torch.Tensor, lap0: torch.Tensor,
    factor, config: AllenCahnCNConfig,
    bv: torch.Tensor, n_int: int,
) -> torch.Tensor:
    n_steps = int(round(config.tmax / config.dt)) + 1
    rollout = torch.zeros(n_int, n_steps, dtype=torch.float64)
    prev_u  = sol0[:n_int].clone()
    prev_l  = lap0[:n_int].clone()
    state   = prev_u.clone()
    rollout[:, 0] = prev_u

    for step in range(n_steps - 1):
        su, sl = prev_u.clone(), prev_l.clone()
        res = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: allen_cahn_residual_operator(
                z, su, sl, bv, config.epsilon, config.dt),
            factor=factor,
            max_iter=GN_STEPS,
            damping=GN_DAMPING,
            record_history=False,
            linear_solver='auto',
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: allen_cahn_residual_and_jacobian(
                z, su, sl, bv, config.epsilon, config.dt),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state  = res.state
        prev_u = state.clone()
        # Update Laplacian from the Allen-Cahn equation residual definition
        prev_l = (2.0 / config.epsilon**2 * (
            (prev_u - su) / config.dt
            - 0.5 * ((su - su.pow(3)) + (prev_u - prev_u.pow(3)))
        ) - sl)
        rollout[:, step + 1] = prev_u
    return rollout


def evaluate_factor(factor, test: dict, bv: torch.Tensor, n_int: int, config: AllenCahnCNConfig) -> dict:
    errors, times = [], []
    for i in range(test['solutions'].shape[0]):
        t0 = time.perf_counter()
        pred = rollout_krom(test['solutions'][i, :, 0], test['laplacians'][i, :, 0],
                            factor, config, bv, n_int)
        times.append(time.perf_counter() - t0)
        errors.append(relative_l2(pred[:, -1], test['solutions'][i, :n_int, -1]))
    return {'mean_rel_l2': sum(errors) / len(errors), 'mean_seconds': sum(times) / len(times)}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def run_n_sweep(
    N_values: Sequence[int] = N_SWEEP,
    rho: float = RHO_DEFAULT,
    config: AllenCahnCNConfig = PAPER_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 0,
    test_seed: int = 1,
    verbose: bool = True,
) -> dict:
    N_max = max(N_values)
    if verbose:
        print(f'  Generating {N_max} training trajectories...')
    train_all = generate_dataset(config, N_max, train_seed)
    if verbose:
        print(f'  Generating {n_test} test trajectories...')
    test = generate_test_dataset(config, n_test, test_seed)

    results = {'N': [], 'empirical_sparse': [], 'matern_sparse': []}
    for N in N_values:
        if verbose:
            print(f'  N={N}...', end=' ', flush=True)
        sub = {k: v[:N] for k, v in train_all.items()}
        factors = build_krom_factors(sub, config, rho)
        bv, n_int = factors['boundary_values'], factors['n_int']
        t0 = time.perf_counter()
        results['N'].append(N)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, bv, n_int, config)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    return results


def run_rho_sweep(
    rho_values: Sequence[float] = RHO_SWEEP,
    N_fixed: int = N_FIXED_RHO,
    config: AllenCahnCNConfig = PAPER_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 0,
    test_seed: int = 1,
    verbose: bool = True,
) -> dict:
    train_all = generate_dataset(config, N_fixed, train_seed)
    test      = generate_test_dataset(config, n_test, test_seed)

    results = {'rho': [], 'empirical_sparse': [], 'matern_sparse': []}
    for rho in rho_values:
        if verbose:
            print(f'  rho={rho}...', end=' ', flush=True)
        factors = build_krom_factors(train_all, config, rho)
        bv, n_int = factors['boundary_values'], factors['n_int']
        results['rho'].append(rho)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, bv, n_int, config)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}")
    return results


def plot_sweeps(n_res: dict, rho_res: dict, save_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(n_res['N'], n_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(n_res['N'], n_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('N'); ax.set_ylabel('Relative L2 error (t=T)')
    n_int = (PAPER_CONFIG.nx-2)**2
    ax.set_title(f'Exp 4: Allen-Cahn  —  N sweep  (M≈{n_int}, ρ=5)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(rho_res['rho'], rho_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(rho_res['rho'], rho_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('ρ'); ax.set_ylabel('Relative L2 error (t=T)')
    ax.set_title(f'Exp 4: Allen-Cahn  —  ρ sweep  (N={N_FIXED_RHO})'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / 'exp4_allen_cahn_sweeps.png', dpi=150, bbox_inches='tight')
    return fig


# %%
if __name__ == '__main__':
    n_int = (PAPER_CONFIG.nx - 2) ** 2
    print('=' * 60)
    print('Experiment 4: Allen-Cahn 2D (paper Table 1)')
    print(f'  M≈{n_int}, ε=0.01, Δt=0.05, T=5, domain (0,2π)²')
    print(f'  Test IC: 0.25 sin(3x)sin(3y)')
    print('=' * 60)
    print('\n--- N sweep (ρ=5) ---')
    n_results   = run_n_sweep(verbose=True)
    print('\n--- ρ sweep (N=200) ---')
    rho_results = run_rho_sweep(verbose=True)
    out = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    plot_sweeps(n_results, rho_results, save_dir=out)
    plt.show()
