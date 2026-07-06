# %%
"""
Kolmogorov Flow ROM Comparison
==============================
Compares three reduced-order models on 2D Kolmogorov flow (forced periodic NS):

  1. KROM         -- sparse Cholesky + matrix-free Gauss-Newton (empirical kernel)
  2. POD-Galerkin -- truncated SVD basis + spectral Galerkin + Smagorinsky closure
  3. POD-DEIM     -- same POD basis + DEIM hyper-reduction for nonlinear convection

Domain  : [0, 2π]² periodic, uniform nx×nx grid
Forcing : f(y) = A · sin(k_f · y)  added to vorticity equation
State   : vorticity ω(x, y, t)

At nx=64  →  n_int = 4096,  n_s(KROM) = 3·4096 = 12288 > 2048  →  CG path active
At nx=32  →  n_int = 1024,  n_s(KROM) = 3072                    →  CG path active (minimal)

Running the file:
    python experiments/Kolmogorov_ROM_Comparison.py

To use as a notebook, open in VS Code / Jupyter and run cell-by-cell (# %% markers).
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.empirical import temporal_feature_matrix
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.navier_stokes import (
    AUTO_DIRECT_THRESHOLD,
    build_navier_stokes_factors,
    rollout_navier_stokes_krom,
)
from src.krom.pde_baselines import (
    NSVorticityConfig,
    _ns_spectral_operators,
    compute_energy_spectrum,
    compute_velocity_from_vorticity,
    generate_ns_vorticity_initial_condition,
    ns_vorticity_grid,
)
from src.krom.workflows import (
    build_navier_stokes_empirical_theta,
    navier_stokes_vorticity_residual_operator,
    navier_stokes_vorticity_residual_and_jacobian,
)
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.ordering import build_measurement_ordering

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KolmogorovConfig:
    """Kolmogorov flow parameters.  nx=64 activates KROM CG path (n_s=12288)."""
    nx: int = 32                   # grid points per side (use 64 for CG regime)
    domain_length: float = 2.0 * math.pi
    dt: float = 2e-2
    tmax: float = 4.0              # rollout horizon
    viscosity: float = 0.01        # Re ≈ 1/(ν k_f²) ≈ 25 at ν=0.01, k_f=2
    forcing_amplitude: float = 1.0 # A
    forcing_wavenumber: int = 2    # k_f
    modes: int = 4                 # initial condition Fourier modes
    snapshot_stride: int = 1
    # derived NSVorticityConfig (domain_length matches periodic spectral solver)
    def to_ns_config(self) -> NSVorticityConfig:
        return NSVorticityConfig(
            nx=self.nx,
            domain_length=self.domain_length,
            dt=self.dt,
            tmax=self.tmax,
            viscosity=self.viscosity,
            modes=self.modes,
            snapshot_stride=self.snapshot_stride,
        )


@dataclass
class ExperimentConfig:
    """Hyperparameter sweeps for the comparison."""
    n_train: int = 20
    n_test: int = 8
    train_seed: int = 42
    test_seed: int = 99
    # KROM
    krom_rho_values: list = field(default_factory=lambda: [2.0, 3.0, 4.0])
    krom_gn_steps: int = 3
    krom_damping: float = 1e-8
    krom_cg_tol: float = 1e-6
    # k_neighbors=3 aligns with the official repo's stated default for PDE
    # problems with derivative measurements (same alignment applied to
    # Burgers/Allen-Cahn/Darcy/Elliptic/Moving-Domain-Heat/navier_stokes.py).
    krom_k_neighbors: int = 3
    # POD
    pod_k_values: list = field(default_factory=lambda: [5, 10, 20, 30, 50])
    # DEIM
    deim_m_ratio: float = 1.5      # m = round(k * deim_m_ratio) DEIM points
    # Smagorinsky
    smagorinsky_cs: float = 0.18   # Smagorinsky constant


# ---------------------------------------------------------------------------
# Kolmogorov flow FOM
# ---------------------------------------------------------------------------

def kolmogorov_forcing_hat(
    config: KolmogorovConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Vorticity forcing corresponding to body force f(y) = A·sin(k_f·y) in x-momentum.
    The vorticity equation gains:   ∂ω/∂t + ... = ν·Δω  +  curl(f)
    curl(f) = ∂f_y/∂x - ∂f_x/∂y = -∂(A·sin(k_f·y))/∂y = -A·k_f·cos(k_f·y)

    Returns the Fourier coefficient tensor (nx,nx) of the forcing.
    """
    nx = config.nx
    forcing_hat = torch.zeros((nx, nx), dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=device)
    kf = config.forcing_wavenumber
    A = config.forcing_amplitude
    # cos(k_f·y) has nonzero Fourier coefficients at (i=0, j=±k_f)
    # The (j=k_f) and (j=-k_f) components each contribute A*k_f/2 to the real part
    forcing_hat[0, kf % nx] = -A * kf / 2.0
    forcing_hat[0, (-kf) % nx] = -A * kf / 2.0
    return forcing_hat


def kolmogorov_rollout_fom(
    initial_vorticity: torch.Tensor,
    config: KolmogorovConfig,
) -> dict[str, torch.Tensor]:
    """
    Full-order spectral Crank-Nicolson solver for Kolmogorov flow.
    Extends pde_baselines.navier_stokes_vorticity_rollout with body forcing.
    """
    dtype = initial_vorticity.dtype
    device = initial_vorticity.device
    ns_config = config.to_ns_config()
    kx, ky, k_squared = _ns_spectral_operators(ns_config, dtype=dtype, device=device)
    forcing_hat = kolmogorov_forcing_hat(config, dtype=dtype, device=device)

    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1

    solutions = torch.zeros((config.nx * config.nx, num_snapshots), dtype=dtype, device=device)
    gradients_x = torch.zeros_like(solutions)
    gradients_y = torch.zeros_like(solutions)
    laplacians = torch.zeros_like(solutions)

    omega = initial_vorticity.clone()
    snapshot_index = 0

    for step in range(total_steps + 1):
        u, v, lap, omega_hat, psi_hat = compute_velocity_from_vorticity(omega, ns_config)
        d_omega_dx = torch.fft.ifft2(1j * kx * omega_hat).real
        d_omega_dy = torch.fft.ifft2(1j * ky * omega_hat).real

        if step % config.snapshot_stride == 0:
            solutions[:, snapshot_index] = omega.reshape(-1)
            gradients_x[:, snapshot_index] = d_omega_dx.reshape(-1)
            gradients_y[:, snapshot_index] = d_omega_dy.reshape(-1)
            laplacians[:, snapshot_index] = lap.reshape(-1)
            snapshot_index += 1

        if step == total_steps:
            break

        convection = u * d_omega_dx + v * d_omega_dy
        convection_hat = torch.fft.fft2(convection)
        # Crank-Nicolson with forcing:
        #   (1 + ν·k²·dt/2)·ω̂_new = (1 - ν·k²·dt/2)·ω̂_old - dt·N̂_old + dt·f̂
        numerator = (
            (1.0 - 0.5 * config.dt * config.viscosity * k_squared) * omega_hat
            - config.dt * convection_hat
            + config.dt * forcing_hat
        )
        denominator = 1.0 + 0.5 * config.dt * config.viscosity * k_squared
        omega_hat_next = numerator / denominator
        omega_hat_next[0, 0] = 0.0
        omega = torch.fft.ifft2(omega_hat_next).real

    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=dtype, device=device)
    return {
        'times': times,
        'solutions': solutions,
        'gradients_x': gradients_x,
        'gradients_y': gradients_y,
        'laplacians': laplacians,
    }


def generate_dataset(
    n_samples: int,
    config: KolmogorovConfig,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> dict[str, torch.Tensor]:
    """Generate FOM trajectories for Kolmogorov flow."""
    ns_config = config.to_ns_config()
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1
    n = config.nx * config.nx

    solutions = torch.zeros((n_samples, n, num_snapshots), dtype=dtype, device=device)
    gradients_x = torch.zeros_like(solutions)
    gradients_y = torch.zeros_like(solutions)
    laplacians = torch.zeros_like(solutions)
    initial_conditions = []

    torch.manual_seed(seed)
    for i in range(n_samples):
        omega0 = generate_ns_vorticity_initial_condition(ns_config, dtype=dtype, device=device)
        result = kolmogorov_rollout_fom(omega0, config)
        solutions[i] = result['solutions']
        gradients_x[i] = result['gradients_x']
        gradients_y[i] = result['gradients_y']
        laplacians[i] = result['laplacians']
        initial_conditions.append(omega0)

    points, _ = ns_vorticity_grid(ns_config, dtype=dtype, device=device)
    return {
        'points': points,
        'solutions': solutions,
        'gradients_x': gradients_x,
        'gradients_y': gradients_y,
        'laplacians': laplacians,
        'initial_conditions': torch.stack(initial_conditions, dim=0),
    }


# ---------------------------------------------------------------------------
# POD machinery
# ---------------------------------------------------------------------------

def build_pod_basis(
    snapshots: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute POD basis from snapshot matrix.

    Args:
        snapshots: (n_samples, n_points, n_times) or (n_points, n_total_snaps)
        k: number of retained modes

    Returns:
        Psi: (n_points, k) orthonormal POD modes  (left singular vectors)
        sigma: (k,) singular values
    """
    if snapshots.ndim == 3:
        n_samples, n_points, n_times = snapshots.shape
        S = snapshots.permute(1, 0, 2).reshape(n_points, n_samples * n_times)
    else:
        S = snapshots

    # Economy SVD: U (n x r), sigma (r,), Vt (r x m)
    U, sigma, _ = torch.linalg.svd(S, full_matrices=False)
    k_actual = min(k, sigma.numel())
    return U[:, :k_actual].contiguous(), sigma[:k_actual]


def explained_variance(sigma: torch.Tensor) -> torch.Tensor:
    """Cumulative fraction of squared singular value energy."""
    energy = sigma.square()
    return torch.cumsum(energy, dim=0) / energy.sum()


# ---------------------------------------------------------------------------
# Spectral differentiation helpers (needed for Galerkin projection)
# ---------------------------------------------------------------------------

def spectral_gradient_x(field: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
    """∂field/∂x via FFT (field shape: nx×nx)."""
    return torch.fft.ifft2(1j * kx * torch.fft.fft2(field)).real


def spectral_gradient_y(field: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(1j * ky * torch.fft.fft2(field)).real


def spectral_laplacian(field: torch.Tensor, k_squared: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(-k_squared * torch.fft.fft2(field)).real


# ---------------------------------------------------------------------------
# POD-Galerkin precomputation
# ---------------------------------------------------------------------------

def build_pod_galerkin_operators(
    Psi: torch.Tensor,
    config: KolmogorovConfig,
) -> dict[str, torch.Tensor]:
    """
    Precompute all operators needed for POD-Galerkin on vorticity NS.

    The Galerkin-projected vorticity equation is:
        ȧ = L·a  -  C(a,a)  +  f_pod

    where:
        L_{ij}    = ν·⟨ψ_i, Δψ_j⟩          (diffusion, k×k)
        C_{ijl}   = ⟨ψ_i, (u_j·∇)ω_l⟩      (convection tensor, k×k×k)
        f_pod_i   = ⟨ψ_i, forcing⟩          (forcing, k)
        ⟨·,·⟩     = (dx·dy) / n  inner product (uniform grid)

    All modes ψ_j are stored as flattened nx² vectors; we reshape as needed.

    Returns dict of precomputed tensors and metadata.
    """
    ns_config = config.to_ns_config()
    k, nx = Psi.shape[1], config.nx
    dtype, device = Psi.dtype, Psi.device
    kx, ky, k_sq = _ns_spectral_operators(ns_config, dtype=dtype, device=device)
    dx_dy = (config.domain_length / nx) ** 2   # uniform cell area

    # --- diffusion matrix L (k×k) -----------------------------------
    L = torch.zeros(k, k, dtype=dtype, device=device)
    for j in range(k):
        phi_j = Psi[:, j].reshape(nx, nx)
        lap_phi_j = spectral_laplacian(phi_j, k_sq)
        for i in range(k):
            phi_i = Psi[:, i].reshape(nx, nx)
            L[i, j] = config.viscosity * float((phi_i * lap_phi_j).sum() * dx_dy)

    # --- convection tensor C (k×k×k) --------------------------------
    # C[i,j,l] = ∫ ψ_i · (u_j · ∂_x ω_l  +  v_j · ∂_y ω_l) dx
    # where u_j = ∂_y ψ_j^stream,  v_j = -∂_x ψ_j^stream
    # and  ψ_j^stream is the streamfunction for vorticity ψ_j (Poisson solve)
    print(f"    Computing convection tensor ({k}×{k}×{k})...", flush=True)

    # Precompute streamfunctions and velocities for each mode
    u_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    v_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    domega_dx_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    domega_dy_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)

    for j in range(k):
        phi_j = Psi[:, j].reshape(nx, nx)
        phi_hat = torch.fft.fft2(phi_j)
        psi_hat = -phi_hat / k_sq
        psi_hat[0, 0] = 0.0
        u_modes[j] = torch.fft.ifft2(1j * ky * psi_hat).real   # u = ∂_y ψ
        v_modes[j] = -torch.fft.ifft2(1j * kx * psi_hat).real  # v = -∂_x ψ
        domega_dx_modes[j] = spectral_gradient_x(phi_j, kx)
        domega_dy_modes[j] = spectral_gradient_y(phi_j, ky)

    C = torch.zeros(k, k, k, dtype=dtype, device=device)
    for j in range(k):
        for l in range(k):
            advected = u_modes[j] * domega_dx_modes[l] + v_modes[j] * domega_dy_modes[l]
            # project onto all test modes at once
            for i in range(k):
                C[i, j, l] = float((Psi[:, i].reshape(nx, nx) * advected).sum() * dx_dy)

    # --- forcing vector f_pod (k,) ----------------------------------
    # curl of Kolmogorov forcing: -A·k_f·cos(k_f·y)
    _, x_1d = ns_vorticity_grid(ns_config, dtype=dtype, device=device)
    y_grid = x_1d.unsqueeze(0).expand(nx, nx)   # shape (nx, nx) — y varies along columns
    forcing_field = -config.forcing_amplitude * config.forcing_wavenumber * torch.cos(
        config.forcing_wavenumber * y_grid
    )
    f_pod = torch.zeros(k, dtype=dtype, device=device)
    for i in range(k):
        f_pod[i] = float((Psi[:, i].reshape(nx, nx) * forcing_field).sum() * dx_dy)

    return {
        'L': L,               # (k, k)
        'C': C,               # (k, k, k)
        'f_pod': f_pod,       # (k,)
        'Psi': Psi,           # (n, k)
        'dx_dy': dx_dy,
        'k': k,
    }


# ---------------------------------------------------------------------------
# Smagorinsky eddy-viscosity closure
# ---------------------------------------------------------------------------

def smagorinsky_eddy_viscosity(
    a: torch.Tensor,
    Psi: torch.Tensor,
    config: KolmogorovConfig,
    u_modes: torch.Tensor,
    v_modes: torch.Tensor,
    Cs: float = 0.18,
) -> torch.Tensor:
    """
    Smagorinsky SGS closure projected onto POD basis.

    ν_T(x) = (Cs·Δ)² |S̄|   where |S̄|² = 2(S₁₁² + S₁₂² + S₂₂²)
    Δ = domain_length / nx  (grid spacing)
    SGS term projected: τ_i = ν_T_mean · ∫ ψ_i · Δω dx  (simplified isotropic)

    Returns correction vector (k,) to subtract from ȧ.
    """
    ns_config = config.to_ns_config()
    nx = config.nx
    dtype, device = a.dtype, a.device
    kx, ky, k_sq = _ns_spectral_operators(ns_config, dtype=dtype, device=device)
    dx_dy = (config.domain_length / nx) ** 2
    delta = config.domain_length / nx

    # Reconstruct velocity field from POD coefficients
    u_field = torch.einsum('j,jxy->xy', a, u_modes)  # (nx, nx)
    v_field = torch.einsum('j,jxy->xy', a, v_modes)

    # Strain-rate tensor components
    du_dx = spectral_gradient_x(u_field, kx)
    du_dy = spectral_gradient_y(u_field, ky)
    dv_dx = spectral_gradient_x(v_field, kx)

    S11 = du_dx
    S12 = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (S11.square() + 2.0 * S12.square()).clamp(min=0.0))

    nu_t = (Cs * delta) ** 2 * S_mag  # (nx, nx) eddy viscosity field

    # Reconstruct current vorticity field
    omega_field = torch.einsum('j,nj->n', a, Psi).reshape(nx, nx)
    lap_omega = spectral_laplacian(omega_field, k_sq)

    # SGS vorticity tendency: ∇·(ν_T ∇ω) ≈ ν_T_mean · Δω  (leading order)
    # (full tensor would require ν_T-weighted Laplacian; this is the standard approximation)
    sgs_field = nu_t * lap_omega

    # Project onto POD basis
    tau = torch.zeros_like(a)
    for i in range(a.numel()):
        tau[i] = float((Psi[:, i].reshape(nx, nx) * sgs_field).sum() * dx_dy)
    return tau


# ---------------------------------------------------------------------------
# POD-Galerkin + Smagorinsky rollout (Crank-Nicolson in reduced space)
# ---------------------------------------------------------------------------

def rollout_pod_galerkin(
    omega0: torch.Tensor,
    ops: dict,
    config: KolmogorovConfig,
    cs: float = 0.18,
    use_smagorinsky: bool = True,
    blow_up_threshold: float = 1e4,
) -> dict[str, torch.Tensor]:
    """
    POD-Galerkin Crank-Nicolson rollout with Smagorinsky closure.

    Timestep: implicit diffusion, explicit convection + forcing + SGS.
    Scheme:
        (I/dt - L/2) a_new = (I/dt + L/2) a_old - C(a_old,a_old) + f + τ_sgs(a_old)

    Returns dict with 'solutions' (n_points, n_snaps) and blow_up flag.
    """
    ns_config = config.to_ns_config()
    nx = config.nx
    dtype, device = omega0.dtype, omega0.device
    L, C, f_pod, Psi = ops['L'], ops['C'], ops['f_pod'], ops['Psi']
    k = ops['k']
    kx, ky, k_sq = _ns_spectral_operators(ns_config, dtype=dtype, device=device)

    # Precompute velocity modes (needed for Smagorinsky)
    u_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    v_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    for j in range(k):
        phi_hat = torch.fft.fft2(Psi[:, j].reshape(nx, nx))
        psi_hat = -phi_hat / k_sq
        psi_hat[0, 0] = 0.0
        u_modes[j] = torch.fft.ifft2(1j * ky * psi_hat).real
        v_modes[j] = -torch.fft.ifft2(1j * kx * psi_hat).real

    # Project initial condition onto POD basis
    dx_dy = (config.domain_length / nx) ** 2
    a = torch.einsum('ni,n->i', Psi, omega0.reshape(-1)) * dx_dy  # (k,)

    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1
    n_points = nx * nx

    solutions = torch.zeros((n_points, num_snapshots), dtype=dtype, device=device)
    reduced_coords = torch.zeros((k, num_snapshots), dtype=dtype, device=device)
    snapshot_index = 0
    blow_up = False
    dt = config.dt

    # LHS matrix for implicit diffusion step (stays constant)
    eye_k = torch.eye(k, dtype=dtype, device=device)
    A_lhs = eye_k / dt - 0.5 * L  # (I/dt - L/2)
    A_lhs_lu = torch.linalg.lu_factor(A_lhs)

    for step in range(total_steps + 1):
        if step % config.snapshot_stride == 0:
            omega_reconstructed = Psi.matmul(a)  # (n,)
            solutions[:, snapshot_index] = omega_reconstructed
            reduced_coords[:, snapshot_index] = a
            snapshot_index += 1

        if step == total_steps:
            break

        # Nonlinear convection: C(a,a) = C_{ijl} a_j a_l  →  shape (k,)
        # conv[i] = sum_{j,l} C[i,j,l] * a[j] * a[l]
        Caa = torch.einsum('ijl,j,l->i', C, a, a)

        # SGS closure
        tau = smagorinsky_eddy_viscosity(a, Psi, config, u_modes, v_modes, Cs=cs) if use_smagorinsky else torch.zeros_like(a)

        # RHS: (I/dt + L/2) a_old - Caa + f + tau
        rhs = (eye_k / dt + 0.5 * L).matmul(a) - Caa + f_pod + tau

        a_new = torch.linalg.lu_solve(*A_lhs_lu, rhs.unsqueeze(-1)).squeeze(-1)

        if torch.any(torch.isnan(a_new)) or float(a_new.abs().max()) > blow_up_threshold:
            blow_up = True
            # Fill remaining snapshots with last good state
            for idx in range(snapshot_index, num_snapshots):
                solutions[:, idx] = solutions[:, max(0, snapshot_index - 1)]
            break

        a = a_new

    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=dtype, device=device)
    return {
        'times': times,
        'solutions': solutions,
        'reduced_coords': reduced_coords,
        'blow_up': blow_up,
        'blow_up_step': step if blow_up else total_steps,
    }


# ---------------------------------------------------------------------------
# DEIM point selection
# ---------------------------------------------------------------------------

def deim_select_points(F: torch.Tensor, m: int) -> torch.Tensor:
    """
    Greedy DEIM algorithm (Chaturantabut & Sorensen 2010).

    Args:
        F: (n, m_modes) matrix of nonlinear snapshot modes (POD of the nonlinear term)
        m: number of interpolation points to select

    Returns:
        indices: (m,) integer tensor of selected point indices
    """
    n, n_modes = F.shape
    m = min(m, n_modes, n)
    indices = torch.zeros(m, dtype=torch.long, device=F.device)

    # First point: argmax of abs of first mode
    indices[0] = F[:, 0].abs().argmax()

    for j in range(1, m):
        # Solve P^T F[:, :j] c = P^T f_j  for c, then compute residual
        Fj = F[:, :j]
        p_idx = indices[:j]
        PtF = Fj[p_idx, :]          # (j, j)
        Ptf = F[p_idx, j]           # (j,)
        try:
            c = torch.linalg.solve(PtF, Ptf.unsqueeze(-1)).squeeze(-1)
        except torch.linalg.LinAlgError:
            c = torch.zeros(j, dtype=F.dtype, device=F.device)
        residual = F[:, j] - Fj.matmul(c)
        indices[j] = residual.abs().argmax()

    return indices


def build_deim_operators(
    Psi: torch.Tensor,
    nonlinear_snapshots: torch.Tensor,
    m: int,
    config: KolmogorovConfig,
) -> dict:
    """
    Build DEIM reconstruction operator for the nonlinear convection term.

    The nonlinear term N(ω) = u·∂_x ω + v·∂_y ω  is first represented
    in a low-dim basis U_N (m_modes modes from SVD of N snapshots).
    DEIM selects m interpolation points and builds the reconstruction matrix P_deim.

    Online:  projected convection = Ψᵀ · U_N · (P^T U_N)^{-1} · N(ω)[indices]
                                  = P_deim_full · N_sampled   (k×m)

    Args:
        Psi: (n, k) POD basis
        nonlinear_snapshots: (n, n_total_snaps) snapshots of the nonlinear convection term
        m: number of DEIM interpolation points
        config: KolmogorovConfig
    """
    ns_config = config.to_ns_config()
    nx = config.nx
    k = Psi.shape[1]
    dtype, device = Psi.dtype, Psi.device
    kx, ky, k_sq = _ns_spectral_operators(ns_config, dtype=dtype, device=device)
    dx_dy = (config.domain_length / nx) ** 2

    # SVD of nonlinear snapshots to get nonlinear basis U_N
    n_nl_modes = min(m, nonlinear_snapshots.shape[1])
    U_N, _, _ = torch.linalg.svd(nonlinear_snapshots, full_matrices=False)
    U_N = U_N[:, :n_nl_modes]   # (n, n_nl_modes)

    # DEIM point selection
    deim_idx = deim_select_points(U_N, m)  # (m,)

    # Reconstruction: P^T U_N  (m × n_nl_modes) — the "interpolation matrix"
    PtUN = U_N[deim_idx, :]      # (m, n_nl_modes)
    # Projected basis: Ψᵀ U_N  (k × n_nl_modes)
    PsiTUN = Psi.t().matmul(U_N) * dx_dy  # (k, n_nl_modes)

    # Online operator: (k × n_nl_modes) @ (n_nl_modes × m)^{-1}  →  (k × m)
    # PsiTUN @ pinv(PtUN) : cheaply computable as PsiTUN @ lstsq(PtUN, I)
    PtUN_pinv = torch.linalg.pinv(PtUN)  # (n_nl_modes, m)
    P_proj = PsiTUN.matmul(PtUN_pinv)    # (k, m)

    # Precompute velocity modes for the Psi basis (for evaluating N at sample points)
    u_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    v_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    domega_dx_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    domega_dy_modes = torch.zeros(k, nx, nx, dtype=dtype, device=device)
    for j in range(k):
        phi_hat = torch.fft.fft2(Psi[:, j].reshape(nx, nx))
        psi_hat = -phi_hat / k_sq
        psi_hat[0, 0] = 0.0
        u_modes[j] = torch.fft.ifft2(1j * ky * psi_hat).real
        v_modes[j] = -torch.fft.ifft2(1j * kx * psi_hat).real
        domega_dx_modes[j] = spectral_gradient_x(Psi[:, j].reshape(nx, nx), kx)
        domega_dy_modes[j] = spectral_gradient_y(Psi[:, j].reshape(nx, nx), ky)

    return {
        'deim_idx': deim_idx,         # (m,)
        'P_proj': P_proj,             # (k, m)  online projection
        'U_N': U_N,                   # (n, n_nl_modes)
        'Psi': Psi,
        'u_modes': u_modes,           # (k, nx, nx)
        'v_modes': v_modes,
        'domega_dx_modes': domega_dx_modes,
        'domega_dy_modes': domega_dy_modes,
        'm': m,
        'k': k,
    }


def deim_project_nonlinear(a: torch.Tensor, deim_ops: dict, config: KolmogorovConfig) -> torch.Tensor:
    """
    Compute the DEIM-approximated projected nonlinear term.

    1. Evaluate u·∂_x ω + v·∂_y ω only at the m DEIM sample points.
    2. Apply precomputed P_proj to get k-vector.

    Cost: O(k·m) instead of O(k·n) for full Galerkin.
    """
    nx = config.nx
    deim_idx = deim_ops['deim_idx']
    P_proj = deim_ops['P_proj']
    u_modes = deim_ops['u_modes']
    v_modes = deim_ops['v_modes']
    domega_dx_modes = deim_ops['domega_dx_modes']
    domega_dy_modes = deim_ops['domega_dy_modes']

    # Evaluate u, v, ∂_x ω, ∂_y ω at DEIM sample points only
    # Shape trick: (k, n) @ a = (n,)  →  index into it
    u_at_pts = torch.einsum('j,jxy->xy', a, u_modes).reshape(-1)[deim_idx]
    v_at_pts = torch.einsum('j,jxy->xy', a, v_modes).reshape(-1)[deim_idx]
    domx_at_pts = torch.einsum('j,jxy->xy', a, domega_dx_modes).reshape(-1)[deim_idx]
    domy_at_pts = torch.einsum('j,jxy->xy', a, domega_dy_modes).reshape(-1)[deim_idx]

    N_sampled = u_at_pts * domx_at_pts + v_at_pts * domy_at_pts  # (m,)
    return P_proj.matmul(N_sampled)  # (k,)


# ---------------------------------------------------------------------------
# POD-DEIM rollout
# ---------------------------------------------------------------------------

def rollout_pod_deim(
    omega0: torch.Tensor,
    pod_ops: dict,
    deim_ops: dict,
    config: KolmogorovConfig,
    blow_up_threshold: float = 1e4,
) -> dict[str, torch.Tensor]:
    """
    POD-DEIM rollout with DEIM hyper-reduction for the nonlinear term.
    Linear + forcing terms are treated identically to POD-Galerkin.
    DEIM replaces the O(k²n) convection tensor contraction with O(k·m).
    """
    nx = config.nx
    dtype, device = omega0.dtype, omega0.device
    L = pod_ops['L']
    f_pod = pod_ops['f_pod']
    Psi = pod_ops['Psi']
    k = pod_ops['k']
    dx_dy = (config.domain_length / nx) ** 2

    # Project initial condition
    a = torch.einsum('ni,n->i', Psi, omega0.reshape(-1)) * dx_dy  # (k,)

    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1
    n_points = nx * nx

    solutions = torch.zeros((n_points, num_snapshots), dtype=dtype, device=device)
    snapshot_index = 0
    blow_up = False
    dt = config.dt

    eye_k = torch.eye(k, dtype=dtype, device=device)
    A_lhs = eye_k / dt - 0.5 * L
    A_lhs_lu = torch.linalg.lu_factor(A_lhs)

    for step in range(total_steps + 1):
        if step % config.snapshot_stride == 0:
            solutions[:, snapshot_index] = Psi.matmul(a)
            snapshot_index += 1

        if step == total_steps:
            break

        Caa = deim_project_nonlinear(a, deim_ops, config)
        rhs = (eye_k / dt + 0.5 * L).matmul(a) - Caa + f_pod

        a_new = torch.linalg.lu_solve(*A_lhs_lu, rhs.unsqueeze(-1)).squeeze(-1)

        if torch.any(torch.isnan(a_new)) or float(a_new.abs().max()) > blow_up_threshold:
            blow_up = True
            for idx in range(snapshot_index, num_snapshots):
                solutions[:, idx] = solutions[:, max(0, snapshot_index - 1)]
            break

        a = a_new

    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=dtype, device=device)
    return {
        'times': times,
        'solutions': solutions,
        'blow_up': blow_up,
        'blow_up_step': step if blow_up else total_steps,
    }


# ---------------------------------------------------------------------------
# KROM Kolmogorov rollout (wraps existing infrastructure + forcing)
# ---------------------------------------------------------------------------

def _build_kolmogorov_forcing_term(
    config: KolmogorovConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    The Kolmogorov forcing enters the vorticity residual as an additive RHS term.
    curl(f) = -A·k_f·cos(k_f·y) evaluated at all grid points (flattened).
    """
    ns_config = config.to_ns_config()
    _, x_1d = ns_vorticity_grid(ns_config, dtype=dtype, device=device)
    nx = config.nx
    # y-coordinate grid: meshgrid with indexing='xy' → y is second axis
    y_grid = x_1d.unsqueeze(0).expand(nx, nx)
    forcing = -config.forcing_amplitude * config.forcing_wavenumber * torch.cos(
        config.forcing_wavenumber * y_grid
    )
    return forcing.reshape(-1)   # (n_points,)


def _kolmogorov_residual_and_jacobian(
    state: torch.Tensor,
    previous_w: torch.Tensor,
    previous_wx: torch.Tensor,
    previous_wy: torch.Tensor,
    previous_lap: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    forcing: torch.Tensor,
    viscosity: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Explicit (residual, Jacobian) for Kolmogorov flow.  Used by the direct solver
    path (n_s ≤ AUTO_DIRECT_THRESHOLD).  Jacobian is identical to the unforced case
    because forcing is constant (independent of state).
    """
    n_points = previous_w.numel()
    w = state[:n_points]
    wx = state[n_points: 2 * n_points]
    wy = state[2 * n_points: 3 * n_points]
    lap = (
        2.0 / viscosity * (
            (w - previous_w) / dt
            + 0.5 * (velocity_u * (wx + previous_wx) + velocity_v * (wy + previous_wy))
        ) - previous_lap
        - (2.0 / viscosity) * forcing
    )
    residual = torch.cat([w, wx, wy, lap], dim=0)
    identity = torch.eye(n_points, dtype=state.dtype, device=state.device)
    jacobian = torch.zeros((4 * n_points, 3 * n_points), dtype=state.dtype, device=state.device)
    jacobian[:n_points, :n_points] = identity
    jacobian[n_points: 2 * n_points, n_points: 2 * n_points] = identity
    jacobian[2 * n_points: 3 * n_points, 2 * n_points: 3 * n_points] = identity
    jacobian[3 * n_points:, :n_points] = (2.0 / viscosity) * identity / dt
    jacobian[3 * n_points:, n_points: 2 * n_points] = (1.0 / viscosity) * torch.diag(velocity_u)
    jacobian[3 * n_points:, 2 * n_points: 3 * n_points] = (1.0 / viscosity) * torch.diag(velocity_v)
    return residual, jacobian


def _kolmogorov_residual_operator(
    state: torch.Tensor,
    previous_w: torch.Tensor,
    previous_wx: torch.Tensor,
    previous_wy: torch.Tensor,
    previous_lap: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    forcing: torch.Tensor,
    viscosity: float,
    dt: float,
):
    """
    Matrix-free ResidualOperator for Kolmogorov flow.  Used by the CG path
    (n_s > AUTO_DIRECT_THRESHOLD).  Extends navier_stokes_vorticity_residual_operator
    with a constant forcing correction in the lap block.
    """
    from src.krom.gauss_newton import ResidualOperator
    n_points = previous_w.numel()
    w = state[:n_points]
    wx = state[n_points: 2 * n_points]
    wy = state[2 * n_points: 3 * n_points]

    coeff_w = torch.full_like(w, 2.0 / viscosity / dt)
    coeff_wx = (1.0 / viscosity) * velocity_u
    coeff_wy = (1.0 / viscosity) * velocity_v

    lap = (2.0 / viscosity * (
        (w - previous_w) / dt
        + 0.5 * (velocity_u * (wx + previous_wx) + velocity_v * (wy + previous_wy))
    ) - previous_lap
    - (2.0 / viscosity) * forcing)

    residual = torch.cat([w, wx, wy, lap], dim=0)

    def jvp(vector):
        dw = vector[:n_points]
        dwx = vector[n_points: 2 * n_points]
        dwy = vector[2 * n_points:]
        return torch.cat([dw, dwx, dwy, coeff_w * dw + coeff_wx * dwx + coeff_wy * dwy], dim=0)

    def vjp(vector):
        weight_w = vector[:n_points]
        weight_wx = vector[n_points: 2 * n_points]
        weight_wy = vector[2 * n_points: 3 * n_points]
        weight_lap = vector[3 * n_points:]
        return torch.cat([
            weight_w + coeff_w * weight_lap,
            weight_wx + coeff_wx * weight_lap,
            weight_wy + coeff_wy * weight_lap,
        ], dim=0)

    diagonal = torch.cat([1.0 + coeff_w.square(), 1.0 + coeff_wx.square(), 1.0 + coeff_wy.square()], dim=0)
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def rollout_krom_kolmogorov(
    initial_omega: torch.Tensor,
    initial_dx: torch.Tensor,
    initial_dy: torch.Tensor,
    initial_laplace: torch.Tensor,
    vorticity_factor,
    config: KolmogorovConfig,
    gn_steps: int = 3,
    gn_damping: float = 1e-8,
    cg_tol: float = 1e-6,
    cg_max_iter: int = 200,
) -> dict[str, torch.Tensor]:
    """
    KROM rollout for Kolmogorov flow.
    Identical to rollout_navier_stokes_krom but with forcing term in the residual.
    Poisson solve always uses FFT (fast path).
    """
    ns_config = config.to_ns_config()
    n_points = config.nx * config.nx
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1
    dtype, device = initial_omega.dtype, initial_omega.device

    forcing = _build_kolmogorov_forcing_term(config, dtype=dtype, device=device)
    solutions = torch.zeros((n_points, num_snapshots), dtype=dtype, device=device)

    current_w = initial_omega.reshape(-1).clone()
    current_wx = initial_dx.reshape(-1).clone()
    current_wy = initial_dy.reshape(-1).clone()
    current_laplace = initial_laplace.reshape(-1).clone()
    state = torch.cat([current_w, current_wx, current_wy], dim=0)
    snapshot_index = 0

    for step in range(total_steps + 1):
        omega_grid = current_w.reshape(config.nx, config.nx)
        from src.krom.navier_stokes import solve_streamfunction_with_fft
        vel = solve_streamfunction_with_fft(omega_grid, ns_config)
        velocity_u = vel['u'].reshape(-1)
        velocity_v = vel['v'].reshape(-1)

        if step % config.snapshot_stride == 0:
            solutions[:, snapshot_index] = current_w
            snapshot_index += 1

        if step == total_steps:
            break

        prev_w = current_w.clone()
        prev_wx = current_wx.clone()
        prev_wy = current_wy.clone()
        prev_lap = current_laplace.clone()

        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: _kolmogorov_residual_operator(
                z, prev_w, prev_wx, prev_wy, prev_lap,
                velocity_u, velocity_v, forcing,
                config.viscosity, config.dt,
            ),
            factor=vorticity_factor,
            max_iter=gn_steps,
            damping=gn_damping,
            record_history=False,
            linear_solver='auto',
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: _kolmogorov_residual_and_jacobian(
                z, prev_w, prev_wx, prev_wy, prev_lap,
                velocity_u, velocity_v, forcing,
                config.viscosity, config.dt,
            ),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state = result.state
        current_w = state[:n_points].clone()
        current_wx = state[n_points: 2 * n_points].clone()
        current_wy = state[2 * n_points: 3 * n_points].clone()
        current_laplace = (
            2.0 / config.viscosity * (
                (current_w - prev_w) / config.dt
                + 0.5 * (velocity_u * (current_wx + prev_wx) + velocity_v * (current_wy + prev_wy))
            ) - prev_lap
            - (2.0 / config.viscosity) * forcing
        )

    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=dtype, device=device)
    return {'times': times, 'solutions': solutions}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(truth))
    if denom < 1e-30:
        return 0.0
    return float(torch.linalg.norm(pred - truth)) / denom


def trajectory_errors(
    pred_solutions: torch.Tensor,
    truth_solutions: torch.Tensor,
) -> torch.Tensor:
    """Relative L2 error at each snapshot. Returns (n_snaps,) tensor."""
    n_snaps = pred_solutions.shape[1]
    errs = torch.zeros(n_snaps)
    for t in range(n_snaps):
        errs[t] = relative_l2(pred_solutions[:, t], truth_solutions[:, t])
    return errs


def enstrophy(omega: torch.Tensor, dx_dy: float) -> float:
    return float(0.5 * omega.square().sum() * dx_dy)


# ---------------------------------------------------------------------------
# Full comparison experiment
# ---------------------------------------------------------------------------

def run_comparison(
    kol_config: KolmogorovConfig,
    exp_config: ExperimentConfig,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> dict:
    """
    End-to-end comparison of KROM, POD-Galerkin+Smagorinsky, and POD-DEIM.

    Returns a results dict containing timing, accuracy, and Pareto data.
    """
    ns_config = kol_config.to_ns_config()
    n_points = kol_config.nx ** 2
    total_steps = int(round(kol_config.tmax / kol_config.dt))
    num_snaps = total_steps // kol_config.snapshot_stride + 1
    dx_dy = (kol_config.domain_length / kol_config.nx) ** 2

    # ------------------------------------------------------------------
    # 1. Generate datasets
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print(f"Kolmogorov flow  nx={kol_config.nx}  Re≈{1/(kol_config.viscosity*kol_config.forcing_wavenumber**2):.0f}")
        print(f"n_points={n_points}  n_s(KROM)={3*n_points}  CG={'YES' if 3*n_points > AUTO_DIRECT_THRESHOLD else 'NO (direct)'}")
        print("=" * 60)
        print(f"\n[1/4] Generating {exp_config.n_train} train + {exp_config.n_test} test trajectories...")

    t0 = time.perf_counter()
    train_data = generate_dataset(exp_config.n_train, kol_config, exp_config.train_seed, dtype, device)
    train_time = time.perf_counter() - t0
    if verbose:
        print(f"    Train data: {train_time:.1f}s")

    t0 = time.perf_counter()
    test_data = generate_dataset(exp_config.n_test, kol_config, exp_config.test_seed, dtype, device)
    test_time = time.perf_counter() - t0
    if verbose:
        print(f"    Test data:  {test_time:.1f}s")

    # Snapshot matrix for POD: (n_points, n_train * n_snaps)
    snap_matrix = train_data['solutions'].permute(1, 0, 2).reshape(n_points, -1)

    # Nonlinear term snapshots (for DEIM): compute N = u·∂_x ω + v·∂_y ω from training
    if verbose:
        print("    Computing nonlinear term snapshots for DEIM...")
    nl_snaps_list = []
    for i in range(exp_config.n_train):
        for t in range(num_snaps):
            omega_t = train_data['solutions'][i, :, t].reshape(kol_config.nx, kol_config.nx)
            u_t, v_t, _, omega_hat_t, _ = compute_velocity_from_vorticity(omega_t, ns_config)
            kx_, ky_, _ = _ns_spectral_operators(ns_config, dtype=dtype, device=device)
            dx_t = torch.fft.ifft2(1j * kx_ * omega_hat_t).real
            dy_t = torch.fft.ifft2(1j * ky_ * omega_hat_t).real
            nl_snaps_list.append((u_t * dx_t + v_t * dy_t).reshape(-1))
    nl_snap_matrix = torch.stack(nl_snaps_list, dim=1)  # (n_points, n_train*n_snaps)

    # ------------------------------------------------------------------
    # 2. POD basis + Galerkin operators (shared across all k values)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2/4] Building POD bases and operators...")
    k_max = max(exp_config.pod_k_values)

    t0 = time.perf_counter()
    Psi_full, sigma = build_pod_basis(snap_matrix, k_max)
    svd_time = time.perf_counter() - t0
    if verbose:
        cum_var = explained_variance(sigma)
        print(f"    SVD ({n_points}×{snap_matrix.shape[1]}): {svd_time:.2f}s")
        for kk in exp_config.pod_k_values:
            kk = min(kk, sigma.numel())
            print(f"    k={kk:3d}: explained variance = {float(cum_var[kk-1])*100:.1f}%")

    # ------------------------------------------------------------------
    # 3. KROM offline build (sweep over rho)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3/4] Building KROM factors (rho sweep)...")

    points = train_data['points']
    derivative_groups = (
        torch.arange(n_points, dtype=torch.long),
        torch.arange(n_points, dtype=torch.long),
        torch.arange(n_points, dtype=torch.long),
    )
    ordering = build_measurement_ordering(points, derivative_groups, k_neighbors=exp_config.krom_k_neighbors)

    krom_factors = {}
    krom_build_times = {}
    for rho in exp_config.krom_rho_values:
        t0 = time.perf_counter()
        # Build empirical theta
        sol_feat = temporal_feature_matrix(train_data['solutions'])
        dx_feat  = temporal_feature_matrix(train_data['gradients_x'])
        dy_feat  = temporal_feature_matrix(train_data['gradients_y'])
        lap_feat = temporal_feature_matrix(train_data['laplacians'])
        theta = build_navier_stokes_empirical_theta(sol_feat, dx_feat, dy_feat, lap_feat, nugget=1e-9)
        theta = theta / sol_feat.shape[1]
        factor, _ = sparse_precision_factor(
            theta=theta,
            dirac_points=points,
            derivative_point_groups=derivative_groups,
            rho=rho,
            nugget=1e-9,
            ordering=ordering,
        )
        krom_factors[rho] = factor
        krom_build_times[rho] = time.perf_counter() - t0
        if verbose:
            print(f"    rho={rho}: build={krom_build_times[rho]:.2f}s")

    # ------------------------------------------------------------------
    # 4. Rollout + evaluation on test set
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[4/4] Evaluating on {exp_config.n_test} test trajectories...")

    results = {
        'config': kol_config,
        'exp_config': exp_config,
        'sigma': sigma,
        'svd_time': svd_time,
        'train_time': train_time,
        'test_time': test_time,
        'krom_build_times': krom_build_times,
        'pod_galerkin': {},
        'pod_deim': {},
        'krom': {},
        'fom_times': [],
    }

    # FOM timing reference
    for i in range(min(3, exp_config.n_test)):
        t0 = time.perf_counter()
        kolmogorov_rollout_fom(test_data['initial_conditions'][i], kol_config)
        results['fom_times'].append(time.perf_counter() - t0)

    # ---- POD-Galerkin + DEIM (one build per k, shared) ----
    for k in exp_config.pod_k_values:
        k_actual = min(k, sigma.numel())
        if verbose:
            print(f"\n  POD k={k_actual}  (Galerkin + DEIM)...")
        Psi_k = Psi_full[:, :k_actual]

        # Build Galerkin operators once
        t0 = time.perf_counter()
        pod_ops = build_pod_galerkin_operators(Psi_k, kol_config)
        galerkin_build_t = time.perf_counter() - t0

        # Build DEIM operators once
        m = min(round(k_actual * exp_config.deim_m_ratio), n_points)
        t0 = time.perf_counter()
        deim_ops = build_deim_operators(Psi_k, nl_snap_matrix, m, kol_config)
        deim_build_t = time.perf_counter() - t0

        if verbose:
            print(f"    Galerkin build: {galerkin_build_t:.2f}s  |  DEIM build (m={m}): {deim_build_t:.2f}s")

        # Rollout on test set
        galerkin_errs_final, galerkin_errs_traj, galerkin_times, galerkin_blowup = [], [], [], []
        deim_errs_final, deim_errs_traj, deim_times, deim_blowup = [], [], [], []

        for i in range(exp_config.n_test):
            omega0 = test_data['initial_conditions'][i].clone()
            truth = test_data['solutions'][i]  # (n_points, n_snaps)

            # POD-Galerkin
            t0 = time.perf_counter()
            gal_res = rollout_pod_galerkin(omega0, pod_ops, kol_config, cs=exp_config.smagorinsky_cs)
            galerkin_times.append(time.perf_counter() - t0)
            galerkin_blowup.append(gal_res['blow_up'])
            galerkin_errs_final.append(relative_l2(gal_res['solutions'][:, -1], truth[:, -1]))
            galerkin_errs_traj.append(trajectory_errors(gal_res['solutions'], truth))

            # POD-DEIM
            t0 = time.perf_counter()
            deim_res = rollout_pod_deim(omega0, pod_ops, deim_ops, kol_config)
            deim_times.append(time.perf_counter() - t0)
            deim_blowup.append(deim_res['blow_up'])
            deim_errs_final.append(relative_l2(deim_res['solutions'][:, -1], truth[:, -1]))
            deim_errs_traj.append(trajectory_errors(deim_res['solutions'], truth))

        results['pod_galerkin'][k_actual] = {
            'build_time': galerkin_build_t + svd_time,
            'mean_online_time': float(np.mean(galerkin_times)),
            'mean_final_err': float(np.mean(galerkin_errs_final)),
            'mean_traj_err': torch.stack(galerkin_errs_traj).mean(0),
            'blow_up_fraction': float(np.mean(galerkin_blowup)),
        }
        results['pod_deim'][k_actual] = {
            'build_time': galerkin_build_t + deim_build_t + svd_time,
            'mean_online_time': float(np.mean(deim_times)),
            'mean_final_err': float(np.mean(deim_errs_final)),
            'mean_traj_err': torch.stack(deim_errs_traj).mean(0),
            'blow_up_fraction': float(np.mean(deim_blowup)),
        }
        if verbose:
            gk = results['pod_galerkin'][k_actual]
            dk = results['pod_deim'][k_actual]
            print(f"    Galerkin: err={gk['mean_final_err']:.4f}  t={gk['mean_online_time']:.2f}s  blow_up={gk['blow_up_fraction']*100:.0f}%")
            print(f"    DEIM:     err={dk['mean_final_err']:.4f}  t={dk['mean_online_time']:.2f}s  blow_up={dk['blow_up_fraction']*100:.0f}%")

    # ---- KROM ----
    for rho in exp_config.krom_rho_values:
        if verbose:
            print(f"\n  KROM rho={rho}...")
        factor = krom_factors[rho]
        krom_errs_final, krom_times = [], []

        for i in range(exp_config.n_test):
            omega0 = test_data['initial_conditions'][i]
            truth = test_data['solutions'][i]
            omega0_grid = omega0.clone()
            # Compute initial derivatives via FFT
            ns_cfg = kol_config.to_ns_config()
            u0, v0, lap0, om_hat0, _ = compute_velocity_from_vorticity(omega0_grid, ns_cfg)
            kx_, ky_, _ = _ns_spectral_operators(ns_cfg, dtype=dtype, device=device)
            dx0 = torch.fft.ifft2(1j * kx_ * om_hat0).real.reshape(-1)
            dy0 = torch.fft.ifft2(1j * ky_ * om_hat0).real.reshape(-1)

            t0 = time.perf_counter()
            krom_res = rollout_krom_kolmogorov(
                initial_omega=omega0_grid,
                initial_dx=dx0,
                initial_dy=dy0,
                initial_laplace=lap0.reshape(-1),
                vorticity_factor=factor,
                config=kol_config,
                gn_steps=exp_config.krom_gn_steps,
                gn_damping=exp_config.krom_damping,
                cg_tol=exp_config.krom_cg_tol,
            )
            krom_times.append(time.perf_counter() - t0)
            krom_errs_final.append(relative_l2(krom_res['solutions'][:, -1], truth[:, -1]))

        results['krom'][rho] = {
            'build_time': krom_build_times[rho],
            'mean_online_time': float(np.mean(krom_times)),
            'mean_final_err': float(np.mean(krom_errs_final)),
        }
        if verbose:
            kr = results['krom'][rho]
            print(f"    err={kr['mean_final_err']:.4f}  t={kr['mean_online_time']:.2f}s")

    results['test_solutions'] = test_data['solutions']
    results['test_ics'] = test_data['initial_conditions']
    results['times_vec'] = torch.linspace(0, kol_config.tmax, num_snaps)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pareto(results: dict, save_path: str | None = None):
    """
    Pareto frontier: (online time per rollout) vs (final relative L2 error).
    Lower-left is better.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Online time vs accuracy ---
    ax = axes[0]
    k_vals = sorted(results['pod_galerkin'].keys())

    # POD-Galerkin
    gal_times = [results['pod_galerkin'][k]['mean_online_time'] for k in k_vals]
    gal_errs = [results['pod_galerkin'][k]['mean_final_err'] for k in k_vals]
    gal_blow = [results['pod_galerkin'][k]['blow_up_fraction'] for k in k_vals]
    ax.plot(gal_times, gal_errs, 'o-', color='steelblue', label='POD-Galerkin+Smag', zorder=3)
    for k, t, e, b in zip(k_vals, gal_times, gal_errs, gal_blow):
        ax.annotate(f'k={k}{"*" if b>0 else ""}', (t, e), textcoords='offset points', xytext=(4, 4), fontsize=7, color='steelblue')

    # POD-DEIM
    deim_times = [results['pod_deim'][k]['mean_online_time'] for k in k_vals]
    deim_errs = [results['pod_deim'][k]['mean_final_err'] for k in k_vals]
    deim_blow = [results['pod_deim'][k]['blow_up_fraction'] for k in k_vals]
    ax.plot(deim_times, deim_errs, 's-', color='darkorange', label='POD-DEIM', zorder=3)
    for k, t, e, b in zip(k_vals, deim_times, deim_errs, deim_blow):
        ax.annotate(f'k={k}{"*" if b>0 else ""}', (t, e), textcoords='offset points', xytext=(4, -10), fontsize=7, color='darkorange')

    # KROM
    rho_vals = sorted(results['krom'].keys())
    krom_times = [results['krom'][r]['mean_online_time'] for r in rho_vals]
    krom_errs = [results['krom'][r]['mean_final_err'] for r in rho_vals]
    ax.plot(krom_times, krom_errs, '^-', color='crimson', label='KROM (empirical)', zorder=3)
    for r, t, e in zip(rho_vals, krom_times, krom_errs):
        ax.annotate(f'ρ={r}', (t, e), textcoords='offset points', xytext=(4, 4), fontsize=7, color='crimson')

    ax.set_xlabel('Mean online time per rollout (s)')
    ax.set_ylabel('Mean final relative L2 error')
    ax.set_title('Online efficiency Pareto frontier\n(* = some rollouts blew up)')
    ax.legend()
    ax.set_yscale('log')

    # --- Right: Blow-up fraction ---
    ax2 = axes[1]
    x = np.arange(len(k_vals))
    width = 0.35
    bars1 = ax2.bar(x - width/2, [results['pod_galerkin'][k]['blow_up_fraction'] * 100 for k in k_vals], width, label='POD-Galerkin+Smag', color='steelblue')
    bars2 = ax2.bar(x + width/2, [results['pod_deim'][k]['blow_up_fraction'] * 100 for k in k_vals], width, label='POD-DEIM', color='darkorange')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in k_vals])
    ax2.set_ylabel('% rollouts that blew up')
    ax2.set_title('Stability: blow-up rate')
    ax2.legend()
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_trajectory_errors(results: dict, save_path: str | None = None):
    """Mean trajectory L2 error vs time for each method/k combination."""
    fig, axes = plt.subplots(1, len(results['exp_config'].pod_k_values), figsize=(5 * len(results['exp_config'].pod_k_values), 4), sharey=True)
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    k_vals = sorted(results['pod_galerkin'].keys())
    times = results['times_vec'].numpy()
    cmap_g = plt.cm.Blues
    cmap_d = plt.cm.Oranges

    rho_vals = sorted(results['krom'].keys())

    for ax_idx, k in enumerate(k_vals):
        ax = axes[ax_idx]
        gal_traj = results['pod_galerkin'][k]['mean_traj_err'].numpy()
        deim_traj = results['pod_deim'][k]['mean_traj_err'].numpy()
        ax.semilogy(times, gal_traj, '-', color='steelblue', label=f'Galerkin k={k}', linewidth=1.5)
        ax.semilogy(times, deim_traj, '--', color='darkorange', label=f'DEIM k={k}', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_title(f'k={k}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Add KROM on all panels for reference
    for ax_idx, k in enumerate(k_vals):
        ax = axes[ax_idx]
        for rho_idx, rho in enumerate(rho_vals):
            color = plt.cm.Reds(0.4 + 0.4 * rho_idx / max(1, len(rho_vals) - 1))
            # Trajectory error not tracked per-step for KROM; show final error as horizontal line
            ax.axhline(results['krom'][rho]['mean_final_err'], color=color, linestyle=':', linewidth=1.5, label=f'KROM ρ={rho}')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Mean relative L2 error')
    fig.suptitle('Trajectory error vs time', y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_singular_values(results: dict, save_path: str | None = None):
    """POD singular value decay and cumulative variance."""
    sigma = results['sigma'].numpy()
    cum_var = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    k_plot = min(80, len(sigma))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(np.arange(1, k_plot + 1), sigma[:k_plot], 'o-', markersize=4)
    axes[0].set_xlabel('Mode index k')
    axes[0].set_ylabel('Singular value σ_k')
    axes[0].set_title(f'POD singular value decay\n(Re≈{1/(results["config"].viscosity*results["config"].forcing_wavenumber**2):.0f}, nx={results["config"].nx})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, k_plot + 1), cum_var[:k_plot] * 100, 'o-', markersize=4)
    axes[1].axhline(99, color='red', linestyle='--', label='99% threshold')
    axes[1].axhline(95, color='orange', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Mode index k')
    axes[1].set_ylabel('Cumulative variance (%)')
    axes[1].set_title('Cumulative energy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_vorticity_comparison(results: dict, test_idx: int = 0, save_path: str | None = None):
    """Side-by-side final vorticity fields for all methods at best k."""
    kol_config = results['config']
    nx = kol_config.nx
    truth_final = results['test_solutions'][test_idx, :, -1].reshape(nx, nx).numpy()

    k_vals = sorted(results['pod_galerkin'].keys())
    # Pick k with lowest POD-Galerkin error (or last one)
    best_k = min(k_vals, key=lambda k: results['pod_galerkin'][k]['mean_final_err'])
    best_rho = min(results['krom'].keys(), key=lambda r: results['krom'][r]['mean_final_err'])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    vmax = float(np.abs(truth_final).max())
    kwargs = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    axes[0].imshow(truth_final, **kwargs)
    axes[0].set_title('FOM')

    # Rerun best configs for a single test trajectory to get fields
    omega0 = results['test_ics'][test_idx]
    Psi_k = results.get('_Psi_full')
    if Psi_k is not None:
        Psi_k = Psi_k[:, :best_k]
        pod_ops = build_pod_galerkin_operators(Psi_k, kol_config)
        gal_res = rollout_pod_galerkin(omega0, pod_ops, kol_config, cs=results['exp_config'].smagorinsky_cs)
        axes[1].imshow(gal_res['solutions'][:, -1].reshape(nx, nx).numpy(), **kwargs)
    axes[1].set_title(f'POD-Galerkin+Smag k={best_k}')

    axes[2].set_title(f'POD-DEIM k={best_k}')
    axes[3].set_title(f'KROM ρ={best_rho}')

    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# %%
if __name__ == '__main__':
    # --- Quick smoke test (nx=16, short horizon) ---
    # Change nx=32 or nx=64 for the real experiment
    print("Running Kolmogorov ROM comparison experiment...")
    print("(Change kol_config.nx to 64 and increase n_train/tmax for the full benchmark)\n")

    kol_config = KolmogorovConfig(
        nx=24,                   # set to 64 for CG regime (n_s=12288)
        dt=2e-2,
        tmax=2.0,
        viscosity=0.02,          # Re ≈ 12  (set to 0.005 for chaotic Re≈100 regime)
        forcing_amplitude=1.0,
        forcing_wavenumber=2,
        modes=4,
    )

    exp_config = ExperimentConfig(
        n_train=10,
        n_test=4,
        krom_rho_values=[2.0, 3.0],
        pod_k_values=[5, 10, 20],
        krom_gn_steps=3,
    )

    results = run_comparison(kol_config, exp_config, verbose=True)

    # Save figures
    out_dir = PROJECT_ROOT / 'experiments' / 'kolmogorov_comparison_results'
    out_dir.mkdir(exist_ok=True)

    fig1 = plot_singular_values(results, save_path=str(out_dir / 'singular_values.png'))
    fig2 = plot_pareto(results, save_path=str(out_dir / 'pareto_frontier.png'))
    fig3 = plot_trajectory_errors(results, save_path=str(out_dir / 'trajectory_errors.png'))

    plt.show()
    print(f"\nFigures saved to {out_dir}")
