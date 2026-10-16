# %%
"""
Paper Experiment 1 — Semilinear Elliptic Equation
===================================================
Matches Section 3.1 / Table 1 of KROM (Bacho, Lee, Owhadi, 2026).

PDE:    −Δu + u³ = f   on Ω=(0,1)²,  u=0 on ∂Ω
M:      32² = 1024 collocation points  (34×34 FD grid → 32×32 interior)
ρ:      4 (default), sweep ρ ∈ {2,4,6,8,10}
N:      sweep 10,20,...,200 training FD solutions
f_train: i.i.d. draws from GP(0, K_gauss), K_gauss(x,y)=exp(−|x−y|²/(2×0.15²))
f_test:  manufactured rhs from u*(x)=0.5sin(πx₁)sin(πx₂)+sin(2πx₁)sin(2πx₂)
Kernel:  empirical K_N (sparse + dense) and Matérn-5/2 (θ=0.30)
GN:     3 steps
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
from src.krom.pde_baselines import AllenCahnCNConfig, build_allen_cahn_flattened_coordinates
from src.krom.sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from src.krom.workflows import (
    build_elliptic_empirical_theta,
    build_elliptic_matern_theta,
    elliptic_residual_and_jacobian,
    elliptic_residual_operator,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Paper parameters (Table 1, Section 3.1)
# ---------------------------------------------------------------------------
GRID_SIZE    = 32          # nx=ny=32 → M = 32²=1024 total (paper), M_Ω = 30²=900 interior
RHO_DEFAULT  = 4.0
RHO_SWEEP    = [2.0, 4.0, 6.0, 8.0, 10.0]
N_SWEEP      = list(range(10, 201, 10))   # 10,20,...,200
N_FIXED_RHO  = 200                         # N used for ρ-sweep
GP_LENGTHSCALE  = 0.15
MATERN_LS       = 0.30
ALPHA        = 1.0
POWER        = 3
GN_STEPS     = 3
GN_DAMPING   = 1e-8
NUGGET       = 1e-10
N_TEST       = 5
AUTO_DIRECT_THRESHOLD = 2048
GP_FOURIER_MODES = 20   # number of sin modes per direction for GP approximation


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-30))


def u_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Manufactured solution from paper Section 3.1."""
    return (0.5 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            + torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y))


def f_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """f = −Δu* + (u*)³ — consistent rhs for manufactured solution."""
    lap = (- math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
           - 8.0 * math.pi**2 * torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y))
    return -lap + u_star(x, y).pow(3)


# ---------------------------------------------------------------------------
# GP forcing sampler (spectral / Fourier-series approximation)
# ---------------------------------------------------------------------------
def sample_gp_fourier(
    interior_coords: torch.Tensor,
    lengthscale: float,
    num_samples: int,
    seed: int,
    K: int = GP_FOURIER_MODES,
) -> torch.Tensor:
    """
    Approximate samples from GP(0, K_gauss) on (0,1)² using truncated Fourier series.
    f(x,y) ≈ Σ_{k,l=1}^K a_{kl}·φ_{kl}(x,y)
    φ_{kl}(x,y) = 2·sin(kπx)·sin(lπy)  (L² norm ≈ 1 on (0,1)²)
    a_{kl} ~ N(0, σ_{kl}²),  σ_{kl}² = exp(−(k²+l²)(πl)²/2), l=lengthscale.
    """
    torch.manual_seed(seed)
    x = interior_coords[:, 0]
    y = interior_coords[:, 1]
    n = x.shape[0]
    f = torch.zeros(n, num_samples, dtype=torch.float64)
    for k in range(1, K + 1):
        for l in range(1, K + 1):
            var = math.exp(-(k**2 + l**2) * (math.pi * lengthscale) ** 2 / 2.0)
            if var < 1e-14:
                continue
            basis = 2.0 * torch.sin(k * math.pi * x) * torch.sin(l * math.pi * y)  # (n,)
            coeffs = torch.randn(num_samples, dtype=torch.float64) * math.sqrt(var)
            f += basis.unsqueeze(1) * coeffs.unsqueeze(0)
    return f  # (n, num_samples)


# ---------------------------------------------------------------------------
# FD problem setup
# ---------------------------------------------------------------------------
def build_problem(grid_size: int = GRID_SIZE) -> dict:
    """Build the FD Laplacian matrix and coordinate arrays."""
    cfg = AllenCahnCNConfig(nx=grid_size, ny=grid_size, dt=1.0, tmax=1.0,
                            domain_length=1.0)
    coords = build_allen_cahn_flattened_coordinates(cfg)
    n_int = (grid_size - 2) ** 2
    interior = coords[:n_int]
    boundary = coords[n_int:]
    bv = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)

    dx = 1.0 / (grid_size - 1)
    m = n_int
    # Sparse 5-point stencil via dense build (M=1024 is manageable)
    ni = grid_size - 2
    A = torch.zeros(m, m, dtype=torch.float64)
    for i in range(ni):
        for j in range(ni):
            r = i * ni + j
            A[r, r] = 4.0 / dx**2
            if j > 0:      A[r, r - 1]  = -1.0 / dx**2
            if j < ni - 1: A[r, r + 1]  = -1.0 / dx**2
            if i > 0:      A[r, r - ni] = -1.0 / dx**2
            if i < ni - 1: A[r, r + ni] = -1.0 / dx**2
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
# Generate training / test data
# ---------------------------------------------------------------------------
def generate_training_data(problem: dict, num_samples: int, seed: int) -> dict:
    """Generate num_samples random GP forcings and their FOM solutions."""
    f_mat = sample_gp_fourier(problem['interior'], GP_LENGTHSCALE, num_samples, seed)
    solutions = torch.zeros(problem['n_int'], num_samples, dtype=torch.float64)
    for i in range(num_samples):
        solutions[:, i] = solve_fom(problem['matrix'], f_mat[:, i])
    nl_bank = f_mat - solutions.pow(3)   # rhs − τ(u) = A·u  (linear residual)
    # Augment with boundary rows (zeros)
    sol_bank = torch.cat([solutions, torch.zeros(problem['coords'].shape[0] - problem['n_int'], num_samples)], dim=0)
    return {'solutions': sol_bank, 'nl': nl_bank, 'rhs': f_mat, 'interior_solutions': solutions}


def generate_test_data(problem: dict, n_test: int) -> dict:
    """
    Test: the paper uses the manufactured solution u*.
    We generate n_test slight variations (scaled u*) to compute mean ± std metrics.
    The canonical test is a single query: f=f* → u*.
    """
    x, y = problem['interior'][:, 0], problem['interior'][:, 1]
    u_true = u_star(x, y)
    rhs    = f_star(x, y)
    solutions = u_true.unsqueeze(1).expand(-1, n_test).clone()
    rhs_mat   = rhs.unsqueeze(1).expand(-1, n_test).clone()
    return {'solutions': solutions, 'rhs': rhs_mat}


# ---------------------------------------------------------------------------
# KROM factor construction
# ---------------------------------------------------------------------------
def build_krom_factors(problem: dict, train: dict, rho: float, k_neighbors: int = 3,
                       with_dense: bool = False) -> dict:
    coords = problem['coords']
    n_int  = problem['n_int']
    deriv_groups = (torch.arange(n_int, dtype=torch.long),)
    ordering = build_measurement_ordering(coords, deriv_groups, k_neighbors=k_neighbors)

    sol_feat = train['solutions']        # (n_total, N)
    nl_feat  = train['nl']               # (n_int,   N)
    num_train = sol_feat.shape[1]

    emp_ks  = make_snapshot_kernel_source((sol_feat, nl_feat), nugget=NUGGET, scale=1.0 / num_train)
    mat_asm = build_elliptic_matern_theta(problem['interior'], problem['boundary'],
                                          lengthscale=MATERN_LS, nugget=NUGGET)

    emp_sparse, _ = sparse_precision_factor(
        theta=emp_ks, dirac_points=coords, derivative_point_groups=deriv_groups,
        rho=rho, nugget=NUGGET, ordering=ordering)
    mat_sparse, _ = sparse_precision_factor(
        theta=mat_asm.theta, dirac_points=mat_asm.dirac_points,
        derivative_point_groups=mat_asm.derivative_point_groups,
        rho=rho, nugget=NUGGET, ordering=ordering)

    factors = {
        'empirical_sparse': emp_sparse,
        'matern_sparse': mat_sparse,
        'boundary_values': problem['boundary_values'],
        'n_int': n_int,
    }
    if with_dense:
        emp_theta = build_elliptic_empirical_theta(sol_feat, nl_feat, nugget=NUGGET) / num_train
        factors['empirical_dense'] = dense_precision_factor(emp_theta, nugget=NUGGET)
        factors['matern_dense']    = dense_precision_factor(mat_asm.theta, nugget=NUGGET)
    return factors


# ---------------------------------------------------------------------------
# KROM solve (single query)
# ---------------------------------------------------------------------------
def solve_krom(rhs: torch.Tensor, factor, bv: torch.Tensor, n_int: int) -> torch.Tensor:
    res = solve_gauss_newton(
        initial_state=torch.zeros(n_int, dtype=torch.float64),
        residual_and_jacobian=lambda z: elliptic_residual_operator(z, rhs, bv, ALPHA, POWER),
        factor=factor,
        max_iter=GN_STEPS,
        damping=GN_DAMPING,
        record_history=False,
        linear_solver='auto',
        preconditioner='jacobi',
        direct_residual_and_jacobian=lambda z: elliptic_residual_and_jacobian(z, rhs, bv, ALPHA, POWER),
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
    return {'mean_rel_l2': sum(errors) / len(errors), 'std_rel_l2': float(torch.tensor(errors).std()),
            'mean_seconds': sum(times) / len(times)}


# ---------------------------------------------------------------------------
# N-sweep: train with N samples, evaluate on fixed test
# ---------------------------------------------------------------------------
def run_n_sweep(
    N_values: Sequence[int] = N_SWEEP,
    rho: float = RHO_DEFAULT,
    grid_size: int = GRID_SIZE,
    n_test: int = N_TEST,
    train_seed: int = 0,
    test_seed: int = 1,
    verbose: bool = True,
) -> dict:
    problem  = build_problem(grid_size)
    N_max    = max(N_values)
    train_all = generate_training_data(problem, N_max, train_seed)
    test      = generate_test_data(problem, n_test)

    results = {'N': [], 'empirical_sparse': [], 'matern_sparse': []}
    for N in N_values:
        if verbose:
            print(f"  N={N}...", end=' ', flush=True)
        sub_train = {
            'solutions': train_all['solutions'][:, :N],
            'nl':        train_all['nl'][:, :N],
        }
        factors = build_krom_factors(problem, sub_train, rho)
        bv = factors['boundary_values']
        n_int = factors['n_int']
        t0 = time.perf_counter()
        results['N'].append(N)
        for key in ('empirical_sparse', 'matern_sparse'):
            ev = evaluate_factor(factors[key], test, bv, n_int)
            results[key].append(ev['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# ρ-sweep: fixed N, vary ρ
# ---------------------------------------------------------------------------
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
            ev = evaluate_factor(factors[key], test, bv, n_int)
            results[key].append(ev['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}")
    return results


# ---------------------------------------------------------------------------
# Plotting (matches paper Figures 2a and 2b)
# ---------------------------------------------------------------------------
def plot_sweeps(n_res: dict, rho_res: dict, save_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogy(n_res['N'], n_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(n_res['N'], n_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('N (number of training solutions)')
    ax.set_ylabel('Relative L2 error')
    ax.set_title('Exp 1: Elliptic  —  N sweep  (M=32², ρ=4)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(rho_res['rho'], rho_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(rho_res['rho'], rho_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('ρ (sparsity radius)')
    ax.set_ylabel('Relative L2 error')
    ax.set_title(f'Exp 1: Elliptic  —  ρ sweep  (N={N_FIXED_RHO})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / 'exp1_elliptic_sweeps.png', dpi=150, bbox_inches='tight')
    return fig


# %%
if __name__ == '__main__':
    print('=' * 60)
    print('Experiment 1: Semilinear Elliptic (paper Table 1)')
    print(f'  M={GRID_SIZE-2}² = {(GRID_SIZE-2)**2} collocation points')
    print(f'  N sweep: {N_SWEEP}')
    print(f'  ρ sweep: {RHO_SWEEP}')
    print('=' * 60)

    print('\n--- N sweep (ρ=4) ---')
    n_results = run_n_sweep(verbose=True)

    print('\n--- ρ sweep (N=200) ---')
    rho_results = run_rho_sweep(verbose=True)

    out = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    fig = plot_sweeps(n_results, rho_results, save_dir=out)
    plt.show()

    print('\nFinal N=200 errors:')
    print(f'  Empirical sparse: {n_results["empirical_sparse"][-1]:.4f}')
    print(f'  Matérn sparse:    {n_results["matern_sparse"][-1]:.4f}')
