# %%
"""
Paper Experiment 5 — Navier-Stokes 2D (periodic vorticity form)
=================================================================
Matches Section 3.5 / Table 1 of KROM (Bacho, Lee, Owhadi, 2026).

PDE:    ∂ω/∂t + u·∇ω = νΔω  on (0,2π)² (periodic), T=5
        −Δψ = ω,  u = ∇⊥ψ
ν:      1e-3
M:      32² = 1024 collocation points
N:      sweep 10,20,...,200 training trajectories
ICs:    Re[Σ_{|k₁|,|k₂|≤8} Z_{k₁k₂}/(1+k₁²+k₂²) e^{i(k₁x+k₂y)}]
        Z_{kl} = ξ·exp(iφ), ξ~N(0,1), φ~Unif(0,π)
FOM:    Crank-Nicolson + spectral FFT (replaces paper's Forward Euler; more stable)
Δt:     0.01, T=5 → 500 steps
ρ:      5 (default), sweep ρ ∈ {2,4,6,8,10}
Kernel: empirical K_N (sparse) and Matérn-5/2 (θ=0.30)
GN:     2 steps  (paper uses standard KROM GN)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.empirical import temporal_feature_matrix
from src.krom.factors import dense_precision_factor
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.navier_stokes import (
    AUTO_DIRECT_THRESHOLD,
    rollout_navier_stokes_krom,
    solve_streamfunction_with_fft,
)
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import (
    NSVorticityConfig,
    _ns_spectral_operators,
    compute_velocity_from_vorticity,
    ns_vorticity_grid,
)
from src.krom.sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from src.krom.workflows import build_navier_stokes_matern_theta

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Paper parameters (Table 1, Section 3.5)
# ---------------------------------------------------------------------------
PAPER_NS_CONFIG = NSVorticityConfig(
    nx=32,
    domain_length=2.0 * math.pi,
    dt=0.01,
    tmax=5.0,
    viscosity=1e-3,
    modes=8,            # IC uses |k₁|,|k₂| ≤ 8
    snapshot_stride=1,
)
RHO_DEFAULT  = 5.0
RHO_SWEEP    = [2.0, 4.0, 6.0, 8.0, 10.0]
N_SWEEP      = list(range(10, 201, 10))
N_FIXED_RHO  = 200
N_TEST       = 5
MATERN_LS    = 0.30
GN_STEPS     = 2
GN_DAMPING   = 1e-8
NUGGET       = 1e-9
K_NEIGHBORS  = 3


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-30))


# ---------------------------------------------------------------------------
# IC generation matching paper Section 3.5
# ω₀ = Re[Σ_{|k₁|,|k₂|≤8} Z_{k₁k₂}/(1+k₁²+k₂²) · e^{i(k₁x+k₂y)}]
# Z_{kl} = ξ·exp(iφ), ξ ~ N(0,1), φ ~ Unif(0,π)
# ---------------------------------------------------------------------------
def paper_ns_ic(config: NSVorticityConfig, seed: int) -> torch.Tensor:
    """Band-limited random vorticity IC with Kolmogorov-like energy spectrum."""
    torch.manual_seed(seed)
    nx = config.nx
    k_max = config.modes   # 8

    # Wavenumber arrays (fftfreq-style)
    kx_1d = torch.fft.fftfreq(nx, d=1.0 / nx).long()
    ky_1d = torch.fft.fftfreq(nx, d=1.0 / nx).long()
    KX, KY = torch.meshgrid(kx_1d, ky_1d, indexing='ij')

    omega_hat = torch.zeros(nx, nx, dtype=torch.complex128)
    amp = torch.randn(nx, nx, dtype=torch.float64)       # ξ
    phase = torch.rand(nx, nx, dtype=torch.float64) * math.pi  # φ ~ Unif(0,π)
    Z = amp * torch.exp(1j * phase.to(torch.complex128))

    mask = (KX.abs() <= k_max) & (KY.abs() <= k_max)
    denom = 1.0 + KX.double().pow(2) + KY.double().pow(2)
    omega_hat[mask] = (Z / denom.to(torch.complex128))[mask]
    # Real-valued vorticity → enforce Hermitian symmetry
    omega_hat[0, 0] = 0.0   # zero mean

    omega = torch.fft.ifft2(omega_hat).real
    # Normalize so max |ω₀| = 1
    scale = omega.abs().max().clamp(min=1e-8)
    return omega / scale


# ---------------------------------------------------------------------------
# Crank-Nicolson FOM for unforced NS vorticity equation
# ∂ω/∂t + u·∇ω = νΔω,   solved spectrally
# ---------------------------------------------------------------------------
def ns_cn_rollout(
    omega0: torch.Tensor,
    config: NSVorticityConfig,
) -> dict[str, torch.Tensor]:
    """
    Crank-Nicolson spectral rollout for UNFORCED NS (no body force).
    Implicit diffusion, explicit convection (IMEX).
    """
    dtype, device = omega0.dtype, omega0.device
    nx = config.nx
    kx, ky, k2 = _ns_spectral_operators(config, dtype=dtype, device=device)

    total_steps = int(round(config.tmax / config.dt))
    num_snaps   = total_steps // config.snapshot_stride + 1
    n_points    = nx * nx

    solutions  = torch.zeros(n_points, num_snaps, dtype=dtype, device=device)
    grad_x     = torch.zeros_like(solutions)
    grad_y     = torch.zeros_like(solutions)
    laplacians = torch.zeros_like(solutions)

    omega = omega0.clone()
    snap_idx = 0

    for step in range(total_steps + 1):
        u, v, lap, omega_hat, _ = compute_velocity_from_vorticity(omega, config)
        dox = torch.fft.ifft2(1j * kx * omega_hat).real
        doy = torch.fft.ifft2(1j * ky * omega_hat).real

        if step % config.snapshot_stride == 0:
            solutions[:, snap_idx] = omega.reshape(-1)
            grad_x[:,   snap_idx]  = dox.reshape(-1)
            grad_y[:,   snap_idx]  = doy.reshape(-1)
            laplacians[:, snap_idx] = lap.reshape(-1)
            snap_idx += 1

        if step == total_steps:
            break

        conv_hat = torch.fft.fft2(u * dox + v * doy)
        # IMEX CN: (1 + νk²Δt/2)ω̂_new = (1 − νk²Δt/2)ω̂ − Δt·conv_hat
        num = (1.0 - 0.5 * config.dt * config.viscosity * k2) * omega_hat - config.dt * conv_hat
        omega_hat_new = num / (1.0 + 0.5 * config.dt * config.viscosity * k2)
        omega_hat_new[0, 0] = 0.0
        omega = torch.fft.ifft2(omega_hat_new).real

    return {
        'solutions': solutions, 'gradients_x': grad_x,
        'gradients_y': grad_y, 'laplacians': laplacians,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(config: NSVorticityConfig, n_samples: int, seed: int) -> dict:
    n_points = config.nx ** 2
    n_snaps  = int(round(config.tmax / config.dt)) // config.snapshot_stride + 1
    sol_all  = torch.zeros(n_samples, n_points, n_snaps, dtype=torch.float64)
    dx_all   = torch.zeros_like(sol_all)
    dy_all   = torch.zeros_like(sol_all)
    lap_all  = torch.zeros_like(sol_all)
    ics      = []
    for i in range(n_samples):
        omega0 = paper_ns_ic(config, seed=seed * 10000 + i)
        out    = ns_cn_rollout(omega0, config)
        sol_all[i] = out['solutions']; dx_all[i] = out['gradients_x']
        dy_all[i]  = out['gradients_y']; lap_all[i] = out['laplacians']
        ics.append(omega0)
    points, _ = ns_vorticity_grid(config, dtype=torch.float64, device=torch.device('cpu'))
    return {
        'solutions': sol_all, 'gradients_x': dx_all,
        'gradients_y': dy_all, 'laplacians': lap_all,
        'initial_conditions': torch.stack(ics), 'points': points,
    }


# ---------------------------------------------------------------------------
# KROM factor construction
# ---------------------------------------------------------------------------
def build_krom_factors(train: dict, config: NSVorticityConfig, rho: float) -> dict:
    points = train['points']
    n_pts  = points.shape[0]
    dg     = (
        torch.arange(n_pts, dtype=torch.long),
        torch.arange(n_pts, dtype=torch.long),
        torch.arange(n_pts, dtype=torch.long),
    )
    ordering = build_measurement_ordering(points, dg, k_neighbors=K_NEIGHBORS)

    sol_feat = temporal_feature_matrix(train['solutions'])
    dx_feat  = temporal_feature_matrix(train['gradients_x'])
    dy_feat  = temporal_feature_matrix(train['gradients_y'])
    lap_feat = temporal_feature_matrix(train['laplacians'])
    M_feat   = sol_feat.shape[1]

    emp_ks = make_snapshot_kernel_source(
        (sol_feat, dx_feat, dy_feat, lap_feat), nugget=NUGGET, scale=1.0 / M_feat)
    mat_asm = build_navier_stokes_matern_theta(points, lengthscale=MATERN_LS, nugget=NUGGET)

    # From build_navier_stokes_factors (empirical kernel only, Matérn separate)
    emp_sparse, _ = sparse_precision_factor(
        theta=emp_ks, dirac_points=points, derivative_point_groups=dg,
        rho=rho, nugget=NUGGET, ordering=ordering)
    mat_sparse, _ = sparse_precision_factor(
        theta=mat_asm.theta, dirac_points=mat_asm.dirac_points,
        derivative_point_groups=mat_asm.derivative_point_groups,
        rho=rho, nugget=NUGGET, ordering=ordering)

    return {
        'empirical_sparse': emp_sparse,
        'matern_sparse':    mat_sparse,
        'points': points,
    }


# ---------------------------------------------------------------------------
# KROM rollout (unforced, delegates to existing rollout_navier_stokes_krom)
# ---------------------------------------------------------------------------
def rollout_krom(
    omega0: torch.Tensor, dx0: torch.Tensor, dy0: torch.Tensor, lap0: torch.Tensor,
    factor, config: NSVorticityConfig,
    gn_steps: int = GN_STEPS,
) -> torch.Tensor:
    """Uses existing rollout_navier_stokes_krom for the unforced equation."""
    out = rollout_navier_stokes_krom(
        initial_omega=omega0,
        initial_dx=dx0,
        initial_dy=dy0,
        initial_laplace=lap0,
        vorticity_factor=factor,
        config=config,
        gn_steps=gn_steps,
        gn_damping=GN_DAMPING,
    )
    return out['solutions']   # (n_pts, n_snaps)


def evaluate_factor(factor, test: dict, config: NSVorticityConfig) -> dict:
    n_pts = config.nx ** 2
    kx, ky, _ = _ns_spectral_operators(config, dtype=torch.float64, device=torch.device('cpu'))
    errors, times = [], []
    for i in range(test['solutions'].shape[0]):
        omega0 = test['initial_conditions'][i]
        truth  = test['solutions'][i]                 # (n_pts, n_snaps)
        _, _, lap0, oh, _ = compute_velocity_from_vorticity(omega0, config)
        dx0 = torch.fft.ifft2(1j * kx * oh).real.reshape(-1)
        dy0 = torch.fft.ifft2(1j * ky * oh).real.reshape(-1)
        t0 = time.perf_counter()
        pred = rollout_krom(omega0.reshape(-1), dx0, dy0, lap0.reshape(-1), factor, config)
        times.append(time.perf_counter() - t0)
        errors.append(relative_l2(pred[:, -1], truth[:, -1]))
    return {'mean_rel_l2': sum(errors) / len(errors), 'mean_seconds': sum(times) / len(times)}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def run_n_sweep(
    N_values: Sequence[int] = N_SWEEP,
    rho: float = RHO_DEFAULT,
    config: NSVorticityConfig = PAPER_NS_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 42,
    test_seed: int = 99,
    verbose: bool = True,
) -> dict:
    N_max = max(N_values)
    if verbose:
        print(f'  Generating {N_max} training trajectories (this may take a while)...')
    train_all = generate_dataset(config, N_max, train_seed)
    if verbose:
        print(f'  Generating {n_test} test trajectories...')
    test = generate_dataset(config, n_test, test_seed)

    results = {'N': [], 'empirical_sparse': [], 'matern_sparse': []}
    for N in N_values:
        if verbose:
            print(f'  N={N}...', end=' ', flush=True)
        sub = {k: v[:N] if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == N_max
               else v for k, v in train_all.items()}
        # Correctly slice only the trajectory arrays
        sub = {
            'solutions':     train_all['solutions'][:N],
            'gradients_x':   train_all['gradients_x'][:N],
            'gradients_y':   train_all['gradients_y'][:N],
            'laplacians':    train_all['laplacians'][:N],
            'points':        train_all['points'],
        }
        factors = build_krom_factors(sub, config, rho)
        t0 = time.perf_counter()
        results['N'].append(N)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, config)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    return results


def run_rho_sweep(
    rho_values: Sequence[float] = RHO_SWEEP,
    N_fixed: int = N_FIXED_RHO,
    config: NSVorticityConfig = PAPER_NS_CONFIG,
    n_test: int = N_TEST,
    train_seed: int = 42,
    test_seed: int = 99,
    verbose: bool = True,
) -> dict:
    train_all = generate_dataset(config, N_fixed, train_seed)
    test      = generate_dataset(config, n_test,  test_seed)

    results = {'rho': [], 'empirical_sparse': [], 'matern_sparse': []}
    for rho in rho_values:
        if verbose:
            print(f'  rho={rho}...', end=' ', flush=True)
        factors = build_krom_factors(train_all, config, rho)
        results['rho'].append(rho)
        for key in ('empirical_sparse', 'matern_sparse'):
            results[key].append(evaluate_factor(factors[key], test, config)['mean_rel_l2'])
        if verbose:
            print(f"emp={results['empirical_sparse'][-1]:.4f}  mat={results['matern_sparse'][-1]:.4f}")
    return results


def plot_sweeps(n_res: dict, rho_res: dict, save_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(n_res['N'], n_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(n_res['N'], n_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('N'); ax.set_ylabel('Relative L2 error (t=T)')
    ax.set_title('Exp 5: NS 2D  —  N sweep  (M=32², ρ=5)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(rho_res['rho'], rho_res['empirical_sparse'], 'o-', label='Empirical sparse', color='steelblue')
    ax.semilogy(rho_res['rho'], rho_res['matern_sparse'],    's--', label='Matérn sparse',   color='darkorange')
    ax.set_xlabel('ρ'); ax.set_ylabel('Relative L2 error (t=T)')
    ax.set_title(f'Exp 5: NS 2D  —  ρ sweep  (N={N_FIXED_RHO})'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / 'exp5_ns_sweeps.png', dpi=150, bbox_inches='tight')
    return fig


# %%
if __name__ == '__main__':
    cfg = PAPER_NS_CONFIG
    print('=' * 60)
    print('Experiment 5: Navier-Stokes 2D (paper Table 1)')
    print(f'  M=32²={cfg.nx**2}, ν={cfg.viscosity}, Δt={cfg.dt}, T={cfg.tmax}')
    print(f'  IC: band-limited Fourier |k|≤{cfg.modes}')
    print('=' * 60)
    print('\n--- N sweep (ρ=5) ---')
    n_results   = run_n_sweep(verbose=True)
    print('\n--- ρ sweep (N=200) ---')
    rho_results = run_rho_sweep(verbose=True)
    out = PROJECT_ROOT / 'experiments' / 'paper' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    plot_sweeps(n_results, rho_results, save_dir=out)
    plt.show()
