# %%
"""
Paper Experiment 2 — Darcy Flow
================================
Matches Section 3.2 / Table 1 of KROM (Bacho, Lee, Owhadi, 2026).

PDE:    −∇·(k(x)∇u) + u³ = f  on Ω=(0,1)²,  u=0 on ∂Ω
k(x):  checkerboard  k = 101/2 − (99/2)(−1)^{⌊8x₁⌋+⌊8x₂⌋}
        (alternates between 100 and 1 on 8×8 sub-squares)
M:     32² = 1024 collocation points  (34×34 FD grid → 32×32 interior)
ρ:     4 (default), sweep ρ ∈ {2,4,6,8,10}
N:     sweep 10,20,...,200 training solutions
f_train: i.i.d. GP(0, K_gauss), K_gauss(x,y)=exp(−|x−y|²/(2×0.2²))
f_test:  constant f=1
Kernel:  empirical K_N (sparse + dense) and Matérn-5/2 (θ=0.30)
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

from src.krom.factors import dense_precision_factor
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import AllenCahnCNConfig, build_allen_cahn_flattened_coordinates
from src.krom.sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from src.krom.workflows import (
    build_elliptic_empirical_theta,
    build_elliptic_matern_theta,
    darcy_residual_and_jacobian,
    darcy_residual_operator,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Paper parameters (Table 1, Section 3.2)
# ---------------------------------------------------------------------------
GRID_SIZE       = 32          # nx=ny=32 → M = 32²=1024 total (paper), M_Ω = 30²=900 interior
RHO_DEFAULT     = 4.0
RHO_SWEEP       = [2.0, 4.0, 6.0, 8.0, 10.0]
N_SWEEP         = list(range(10, 201, 10))
N_FIXED_RHO     = 200
GP_LENGTHSCALE  = 0.20
MATERN_LS       = 0.30
GN_STEPS        = 2
GN_DAMPING      = 1e-8
NUGGET          = 1e-10
N_TEST          = 10          # number of test queries
AUTO_DIRECT_THRESHOLD = 2048
GP_FOURIER_MODES = 20


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Checkerboard diffusion coefficient (paper Section 3.2)
# ---------------------------------------------------------------------------
def k_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """k(x₁,x₂) = 101/2 − (99/2)(−1)^{⌊8x₁⌋+⌊8x₂⌋}."""
    sign = (-1.0) ** ((8.0 * x).floor().long() + (8.0 * y).floor().long())
    return 0.5 * 101.0 - 0.5 * 99.0 * sign.double()


# ---------------------------------------------------------------------------
# GP forcing sampler — full cosine basis on [0,1]²
# ---------------------------------------------------------------------------
# Unlike the solution u (which must vanish on ∂Ω), the forcing f is any
# L²([0,1]²) function, including non-zero-mean ones.  Using only sine modes
# (k,l ≥ 1) produces strictly zero-mean forcings, so the training distribution
# has no DC component and KROM cannot recover u for the constant test f=1.
# The full cosine basis {cos(kπx)cos(lπy) : k,l=0,…,K} spans constant and
# low-frequency functions and cures the problem.
# Normalisation: ‖1‖²=1, ‖√2 cos(kπ·)‖²=1, ‖2 cos(kπ·)cos(lπ·)‖²=1 on [0,1]².
def sample_gp_fourier(
    interior_coords: torch.Tensor,
    lengthscale: float,
    num_samples: int,
    seed: int,
    K: int = GP_FOURIER_MODES,
) -> torch.Tensor:
    torch.manual_seed(seed)
    x, y = interior_coords[:, 0], interior_coords[:, 1]
    n = x.shape[0]
    f = torch.zeros(n, num_samples, dtype=torch.float64)
    for k in range(0, K + 1):
        for l in range(0, K + 1):
            var = math.exp(-(k**2 + l**2) * (math.pi * lengthscale) ** 2 / 2.0)
            if var < 1e-14:
                continue
            # Orthonormal basis on [0,1]²
            norm = (1.0 if k == 0 else math.sqrt(2.0)) * (1.0 if l == 0 else math.sqrt(2.0))
            ck = torch.ones_like(x) if k == 0 else torch.cos(k * math.pi * x)
            cl = torch.ones_like(y) if l == 0 else torch.cos(l * math.pi * y)
            basis = norm * ck * cl
            coeffs = torch.randn(num_samples, dtype=torch.float64) * math.sqrt(var)
            f += basis.unsqueeze(1) * coeffs.unsqueeze(0)
    return f


# ---------------------------------------------------------------------------
# FD problem: build Darcy stiffness matrix with checkerboard k
# ---------------------------------------------------------------------------
def build_problem(grid_size: int = GRID_SIZE) -> dict:
    cfg = AllenCahnCNConfig(nx=grid_size, ny=grid_size, dt=1.0, tmax=1.0,
                            domain_length=1.0)
    coords = build_allen_cahn_flattened_coordinates(cfg)
    n_int  = (grid_size - 2) ** 2
    interior = coords[:n_int]
    boundary = coords[n_int:]
    bv = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)

    x = torch.linspace(0.0, 1.0, grid_size)
    y = torch.linspace(0.0, 1.0, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='xy')   # (nx, ny)
    k_grid = k_coefficient(X, Y)                  # (nx, ny)

    dx = x[1] - x[0]
    ni = grid_size - 2
    m  = n_int
    A  = torch.zeros(m, m, dtype=torch.float64)

    for i in range(ni):
        for j in range(ni):
            ii, jj = i + 1, j + 1    # physical grid indices (1-indexed)
            r = i * ni + j
            # Harmonic mean face conductivities
            k_e = 2.0 * k_grid[ii, jj] * k_grid[ii + 1, jj] / (k_grid[ii, jj] + k_grid[ii + 1, jj] + 1e-30)
            k_w = 2.0 * k_grid[ii, jj] * k_grid[ii - 1, jj] / (k_grid[ii, jj] + k_grid[ii - 1, jj] + 1e-30)
            k_n = 2.0 * k_grid[ii, jj] * k_grid[ii, jj + 1] / (k_grid[ii, jj] + k_grid[ii, jj + 1] + 1e-30)
            k_s = 2.0 * k_grid[ii, jj] * k_grid[ii, jj - 1] / (k_grid[ii, jj] + k_grid[ii, jj - 1] + 1e-30)
            A[r, r] = (k_e + k_w + k_n + k_s) / dx**2
            if i > 0:      A[r, r - ni] = -k_w / dx**2
            if i < ni - 1: A[r, r + ni] = -k_e / dx**2
            if j > 0:      A[r, r - 1]  = -k_s / dx**2
            if j < ni - 1: A[r, r + 1]  = -k_n / dx**2

    return {
        'coords': coords, 'n_int': n_int,
        'interior': interior, 'boundary': boundary,
        'boundary_values': bv, 'matrix': A,
        'grid_size': grid_size,
    }


def solve_fom(A: torch.Tensor, rhs: torch.Tensor, max_iter: int = 40) -> torch.Tensor:
    """Newton solve for Au + u³ = rhs."""
    u = torch.zeros(rhs.shape[0], dtype=torch.float64)
    eye = torch.eye(u.shape[0], dtype=torch.float64)
    for _ in range(max_iter):
        r = A @ u + u.pow(3) - rhs
        J = A + 3.0 * torch.diag(u.pow(2))
        d = torch.linalg.solve(J + NUGGET * eye, r)
        u = u - d
        if torch.linalg.norm(d) < 1e-10 * max(1.0, float(torch.linalg.norm(u))):
            break
    return u


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_training_data(problem: dict, num_samples: int, seed: int) -> dict:
    f_mat     = sample_gp_fourier(problem['interior'], GP_LENGTHSCALE, num_samples, seed)
    solutions = torch.zeros(problem['n_int'], num_samples, dtype=torch.float64)
    for i in range(num_samples):
        solutions[:, i] = solve_fom(problem['matrix'], f_mat[:, i])
    nl_bank  = f_mat - solutions.pow(3)
    sol_bank = torch.cat([solutions, torch.zeros(
        problem['coords'].shape[0] - problem['n_int'], num_samples)], dim=0)
    return {'solutions': sol_bank, 'nl': nl_bank}


def generate_test_data(problem: dict, n_test: int) -> dict:
    """Paper test: constant forcing f=1, solution solved via Newton."""
    rhs_const = torch.ones(problem['n_int'], dtype=torch.float64)
    u_true    = solve_fom(problem['matrix'], rhs_const)
    # Replicate for n_test identical queries (single test condition)
    solutions = u_true.unsqueeze(1).expand(-1, n_test).clone()
    rhs_mat   = rhs_const.unsqueeze(1).expand(-1, n_test).clone()
    return {'solutions': solutions, 'rhs': rhs_mat}


# ---------------------------------------------------------------------------
# KROM factor construction
# ---------------------------------------------------------------------------
def build_krom_factors(problem: dict, train: dict, rho: float, k_neighbors: int = 3,
                       with_dense: bool = False) -> dict:
    coords  = problem['coords']
    n_int   = problem['n_int']
    dg      = (torch.arange(n_int, dtype=torch.long),)
    ordering = build_measurement_ordering(coords, dg, k_neighbors=k_neighbors)

    sol_feat  = train['solutions']
    nl_feat   = train['nl']
    num_train = sol_feat.shape[1]

    emp_ks  = make_snapshot_kernel_source((sol_feat, nl_feat), nugget=NUGGET, scale=1.0 / num_train)
    mat_asm = build_elliptic_matern_theta(problem['interior'], problem['boundary'],
                                          lengthscale=MATERN_LS, nugget=NUGGET)

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
        'boundary_values':  problem['boundary_values'],
        'n_int': n_int,
    }
    if with_dense:
        emp_theta = build_elliptic_empirical_theta(sol_feat, nl_feat, nugget=NUGGET) / num_train
        factors['empirical_dense'] = dense_precision_factor(emp_theta, nugget=NUGGET)
        factors['matern_dense']    = dense_precision_factor(mat_asm.theta, nugget=NUGGET)
    return factors


def solve_krom(rhs: torch.Tensor, factor, bv: torch.Tensor, n_int: int) -> torch.Tensor:
    res = solve_gauss_newton(
        initial_state=torch.zeros(n_int, dtype=torch.float64),
        residual_and_jacobian=lambda z: darcy_residual_operator(z, rhs, bv),
        factor=factor,
        max_iter=GN_STEPS,
        damping=GN_DAMPING,
        record_history=False,
        linear_solver='auto',
        preconditioner='jacobi',
        direct_residual_and_jacobian=lambda z: darcy_residual_and_jacobian(z, rhs, bv),
        direct_threshold=AUTO_DIRECT_THRESHOLD,
    )
    return res.state


def evaluate_factor(factor, test: dict, bv: torch.Tensor, n_int: int) -> dict:
    errors, times = [], []
    for i in range(test['rhs'].shape[1]):
        t0 = time.perf_counter()
        pred = solve_krom(test['rhs'][:, i], factor, bv, n_int)
        times.append(time.perf_counter() - t0)
        errors.append(relative_l2(pred, test['solutions'][:, i]))
    return {'mean_rel_l2': sum(errors) / len(errors), 'mean_seconds': sum(times) / len(times)}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def run_n_sweep(
    N_values: Sequence[int] = N_SWEEP,
    rho: float = RHO_DEFAULT,
    grid_size: int = GRID_SIZE,
    n_test: int = N_TEST,
    train_seed: int = 0,
    verbose: bool = True,
) -> dict:
    problem   = build_problem(grid_size)
    N_max     = max(N_values)
    train_all = generate_training_data(problem, N_max, train_seed)
    test      = generate_test_data(problem, n_test)

    results = {'N': [], 'empirical_sparse': [], 'matern_sparse': []}
    for N in N_values:
        if verbose:
            print(f"  N={N}...", end=' ', flush=True)
        sub = {'solutions': train_all['solutions'][:, :N], 'nl': train_all['nl'][:, :N]}
        factors = build_krom_factors(problem, sub, rho)
        bv, n_int = factors['boundary_values'], factors['n_int']
        t0 = time.perf_counter()
        results['N'].append(N)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, bv, n_int)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    return results


def run_rho_sweep(
    rho_values: Sequence[float] = RHO_SWEEP,
    N_fixed: int = N_FIXED_RHO,
    grid_size: int = GRID_SIZE,
    n_test: int = N_TEST,
    train_seed: int = 0,
    verbose: bool = True,
) -> dict:
    problem   = build_problem(grid_size)
    train_all = generate_training_data(problem, N_fixed, train_seed)
    test      = generate_test_data(problem, n_test)

    results = {'rho': [], 'empirical_sparse': [], 'matern_sparse': []}
    for rho in rho_values:
        if verbose:
            print(f"  rho={rho}...", end=' ', flush=True)
        factors = build_krom_factors(problem, train_all, rho)
        bv, n_int = factors['boundary_values'], factors['n_int']
        results['rho'].append(rho)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, bv, n_int)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}")
    return results


def plot_sweeps(n_res: dict, rho_res: dict, save_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(n_res['N'], n_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(n_res['N'], n_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('N (training solutions)'); ax.set_ylabel('Relative L2 error')
    ax.set_title('Exp 2: Darcy Flow  —  N sweep  (M=32², ρ=4)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(rho_res['rho'], rho_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(rho_res['rho'], rho_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('ρ'); ax.set_ylabel('Relative L2 error')
    ax.set_title(f'Exp 2: Darcy Flow  —  ρ sweep  (N={N_FIXED_RHO})'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / 'exp2_darcy_sweeps.png', dpi=150, bbox_inches='tight')
    return fig


# %%
if __name__ == '__main__':
    print('=' * 60)
    print('Experiment 2: Darcy Flow (paper Table 1)')
    print(f'  M={GRID_SIZE-2}² = {(GRID_SIZE-2)**2}, checkerboard k, f_test=1')
    print('=' * 60)
    print('\n--- N sweep (ρ=4) ---')
    n_results   = run_n_sweep(verbose=True)
    print('\n--- ρ sweep (N=200) ---')
    rho_results = run_rho_sweep(verbose=True)
    out = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    plot_sweeps(n_results, rho_results, save_dir=out)
    plt.show()
