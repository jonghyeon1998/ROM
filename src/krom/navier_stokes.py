from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch

from .empirical import temporal_feature_matrix
from .factors import dense_precision_factor
from .gauss_newton import solve_gauss_newton
from .ordering import build_measurement_ordering
from .pde_baselines import NSVorticityConfig, _ns_spectral_operators, compute_velocity_from_vorticity
from .sparse_cholesky import sparse_precision_factor
from .workflows import (
    build_navier_stokes_empirical_theta,
    build_navier_stokes_matern_theta,
    navier_stokes_vorticity_residual_and_jacobian,
    navier_stokes_vorticity_residual_operator,
)


PoissonSolveMethod = Literal['fft', 'kernel']
KernelKind = Literal['empirical', 'matern']
FactorMode = Literal['sparse', 'dense']
AUTO_DIRECT_THRESHOLD = 2048


@dataclass
class PrecomputedStreamfunctionOperator:
    solution_matrix: torch.Tensor
    n_points: int

    def solve(self, omega: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.solution_matrix.matmul(omega.reshape(-1))
        psi = state[: self.n_points].reshape_as(omega)
        psi_x = state[self.n_points : 2 * self.n_points].reshape_as(omega)
        psi_y = state[2 * self.n_points : 3 * self.n_points].reshape_as(omega)
        return {
            'state': state,
            'psi': psi,
            'psi_x': psi_x,
            'psi_y': psi_y,
            'u': psi_y,
            'v': -psi_x,
        }


@dataclass
class NavierStokesFactorBundle:
    points: torch.Tensor
    ordering: object
    vorticity_factors: dict[str, dict[str, object]]
    streamfunction_operators: dict[str, dict[str, PrecomputedStreamfunctionOperator]]
    build_times: dict[str, float]

    def vorticity_factor(self, kernel_kind: KernelKind, factor_mode: FactorMode = 'sparse') -> object:
        return self.vorticity_factors[kernel_kind][factor_mode]

    def streamfunction_operator(self, kernel_kind: KernelKind, factor_mode: FactorMode = 'sparse') -> PrecomputedStreamfunctionOperator:
        return self.streamfunction_operators[kernel_kind][factor_mode]


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def streamfunction_poisson_residual_and_jacobian(
    state: torch.Tensor,
    rhs_laplace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = rhs_laplace.numel()
    residual = torch.cat(
        [
            state[:n_points],
            state[n_points : 2 * n_points],
            state[2 * n_points : 3 * n_points],
            rhs_laplace,
        ],
        dim=0,
    )
    jacobian = torch.zeros((4 * n_points, 3 * n_points), dtype=state.dtype, device=state.device)
    identity = torch.eye(n_points, dtype=state.dtype, device=state.device)
    jacobian[:n_points, :n_points] = identity
    jacobian[n_points : 2 * n_points, n_points : 2 * n_points] = identity
    jacobian[2 * n_points : 3 * n_points, 2 * n_points : 3 * n_points] = identity
    return residual, jacobian


def _streamfunction_poisson_matrices(
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    jacobian = torch.zeros((4 * n_points, 3 * n_points), dtype=dtype, device=device)
    identity = torch.eye(n_points, dtype=dtype, device=device)
    jacobian[:n_points, :n_points] = identity
    jacobian[n_points : 2 * n_points, n_points : 2 * n_points] = identity
    jacobian[2 * n_points : 3 * n_points, 2 * n_points : 3 * n_points] = identity
    rhs_selector = torch.zeros((4 * n_points, n_points), dtype=dtype, device=device)
    rhs_selector[3 * n_points :, :] = -identity
    return jacobian, rhs_selector


def precompute_streamfunction_operator(
    factor: object,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
    damping: float = 1e-8,
) -> PrecomputedStreamfunctionOperator:
    jacobian, rhs_selector = _streamfunction_poisson_matrices(n_points, dtype=dtype, device=device)
    weighted_jacobian = factor.apply_jacobian(jacobian)
    weighted_rhs = factor.apply_jacobian(rhs_selector)
    normal_matrix = 2.0 * weighted_jacobian.transpose(0, 1).matmul(weighted_jacobian)
    eye = torch.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype, device=normal_matrix.device)
    rhs_matrix = -2.0 * weighted_jacobian.transpose(0, 1).matmul(weighted_rhs)
    chol = torch.linalg.cholesky(normal_matrix + damping * eye)
    solution_matrix = torch.cholesky_solve(rhs_matrix, chol, upper=False)
    return PrecomputedStreamfunctionOperator(solution_matrix=solution_matrix, n_points=n_points)


def solve_streamfunction_with_fft(
    omega: torch.Tensor,
    config: NSVorticityConfig,
) -> dict[str, torch.Tensor]:
    u, v, _, _, psi_hat = compute_velocity_from_vorticity(omega, config)
    psi = torch.fft.ifft2(psi_hat).real
    psi_x = -v
    psi_y = u
    state = torch.cat([psi.reshape(-1), psi_x.reshape(-1), psi_y.reshape(-1)], dim=0)
    return {
        'state': state,
        'psi': psi,
        'psi_x': psi_x,
        'psi_y': psi_y,
        'u': u,
        'v': v,
    }


def solve_streamfunction_with_kernel(
    omega: torch.Tensor,
    solver: object,
    damping: float = 1e-8,
    initial_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(solver, PrecomputedStreamfunctionOperator):
        return solver.solve(omega)

    n_points = omega.numel()
    if initial_state is None:
        initial_state = torch.zeros(3 * n_points, dtype=omega.dtype, device=omega.device)
    rhs_laplace = -omega.reshape(-1)
    result = solve_gauss_newton(
        initial_state=initial_state,
        residual_and_jacobian=lambda z: streamfunction_poisson_residual_and_jacobian(z, rhs_laplace),
        factor=solver,
        max_iter=1,
        damping=damping,
        record_history=False,
        linear_solver='cg',
    )
    state = result.state
    psi = state[:n_points].reshape_as(omega)
    psi_x = state[n_points : 2 * n_points].reshape_as(omega)
    psi_y = state[2 * n_points : 3 * n_points].reshape_as(omega)
    return {
        'state': state,
        'psi': psi,
        'psi_x': psi_x,
        'psi_y': psi_y,
        'u': psi_y,
        'v': -psi_x,
    }


def compute_streamfunction_feature_bank(
    solution_bank: torch.Tensor,
    config: NSVorticityConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if solution_bank.ndim != 3:
        raise ValueError('Expected solution_bank with shape (num_samples, num_points, num_times).')

    num_samples, _, num_times = solution_bank.shape
    omega_fields = solution_bank.permute(0, 2, 1).reshape(-1, config.nx, config.nx)
    kx, ky, k_squared = _ns_spectral_operators(config, dtype=omega_fields.dtype, device=omega_fields.device)
    omega_hat = torch.fft.fft2(omega_fields)
    psi_hat = -omega_hat / k_squared
    psi_hat[:, 0, 0] = 0.0
    psi = torch.fft.ifft2(psi_hat).real
    psi_x = torch.fft.ifft2(1j * kx.unsqueeze(0) * psi_hat).real
    psi_y = torch.fft.ifft2(1j * ky.unsqueeze(0) * psi_hat).real
    reshape_back = lambda tensor: tensor.reshape(num_samples, num_times, -1).permute(0, 2, 1).contiguous()
    return reshape_back(psi), reshape_back(psi_x), reshape_back(psi_y)


def build_navier_stokes_factors(
    points: torch.Tensor,
    train_solutions: torch.Tensor,
    train_dx: torch.Tensor,
    train_dy: torch.Tensor,
    train_laplacians: torch.Tensor,
    config: NSVorticityConfig,
    rho: float,
    lengthscale: float,
    nugget: float = 1e-9,
    burnin: int = 0,
    stride: int = 1,
    streamfunction_damping: float = 1e-8,
    sparse_backend: str = 'auto',
    k_neighbors: int = 3,
    ordering_variant: str = 'dirac_first_then_unif_scale',
) -> NavierStokesFactorBundle:
    # k_neighbors=3 aligns with the official repo's stated default for PDE
    # problems with derivative measurements (same alignment already applied
    # to Burgers/Allen-Cahn/Darcy/Elliptic/Moving-Domain-Heat). Note: every
    # point here carries all three derivative groups, so this case is
    # structurally eligible for ordering_variant="follow_diracs" too -- it's
    # exposed as a parameter but left at the same default variant used
    # elsewhere unless you opt in. The Gauss-Newton vorticity time-stepping
    # and the fft/kernel Poisson-solve choice are untouched by this change.
    build_times: dict[str, float] = {}
    derivative_groups = (
        torch.arange(points.shape[0], dtype=torch.long, device=points.device),
        torch.arange(points.shape[0], dtype=torch.long, device=points.device),
        torch.arange(points.shape[0], dtype=torch.long, device=points.device),
    )
    start = time.perf_counter()
    ordering = build_measurement_ordering(points, derivative_groups, k_neighbors=k_neighbors, variant=ordering_variant)
    build_times['ordering_seconds'] = time.perf_counter() - start

    matern_assembly = build_navier_stokes_matern_theta(points, lengthscale=lengthscale, nugget=nugget)
    solution_features = temporal_feature_matrix(train_solutions, burnin=burnin, stride=stride)
    dx_features = temporal_feature_matrix(train_dx, burnin=burnin, stride=stride)
    dy_features = temporal_feature_matrix(train_dy, burnin=burnin, stride=stride)
    lap_features = temporal_feature_matrix(train_laplacians, burnin=burnin, stride=stride)
    empirical_theta = build_navier_stokes_empirical_theta(
        solution_features,
        dx_features,
        dy_features,
        lap_features,
        nugget=nugget,
    ) / solution_features.shape[1]

    psi_bank, psi_x_bank, psi_y_bank = compute_streamfunction_feature_bank(train_solutions, config)
    psi_features = temporal_feature_matrix(psi_bank, burnin=burnin, stride=stride)
    psi_x_features = temporal_feature_matrix(psi_x_bank, burnin=burnin, stride=stride)
    psi_y_features = temporal_feature_matrix(psi_y_bank, burnin=burnin, stride=stride)
    psi_laplace_features = temporal_feature_matrix(-train_solutions, burnin=burnin, stride=stride)
    stream_empirical_theta = build_navier_stokes_empirical_theta(
        psi_features,
        psi_x_features,
        psi_y_features,
        psi_laplace_features,
        nugget=nugget,
    ) / psi_features.shape[1]

    vorticity_factors = {'empirical': {}, 'matern': {}}
    streamfunction_operators = {'empirical': {}, 'matern': {}}

    start = time.perf_counter()
    vorticity_factors['matern']['dense'] = dense_precision_factor(matern_assembly.theta, nugget=nugget)
    build_times['matern_dense_factor_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    vorticity_factors['matern']['sparse'], _ = sparse_precision_factor(
        theta=matern_assembly.theta,
        dirac_points=matern_assembly.dirac_points,
        derivative_point_groups=matern_assembly.derivative_point_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    build_times['matern_sparse_factor_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    vorticity_factors['empirical']['dense'] = dense_precision_factor(empirical_theta, nugget=nugget)
    build_times['empirical_dense_factor_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    vorticity_factors['empirical']['sparse'], _ = sparse_precision_factor(
        theta=empirical_theta,
        dirac_points=points,
        derivative_point_groups=derivative_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    build_times['empirical_sparse_factor_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    stream_empirical_dense_factor = dense_precision_factor(stream_empirical_theta, nugget=nugget)
    build_times['stream_empirical_dense_factor_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    stream_empirical_sparse_factor, _ = sparse_precision_factor(
        theta=stream_empirical_theta,
        dirac_points=points,
        derivative_point_groups=derivative_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    build_times['stream_empirical_sparse_factor_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    stream_matern_dense_factor = dense_precision_factor(matern_assembly.theta, nugget=nugget)
    build_times['stream_matern_dense_factor_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    stream_matern_sparse_factor, _ = sparse_precision_factor(
        theta=matern_assembly.theta,
        dirac_points=matern_assembly.dirac_points,
        derivative_point_groups=matern_assembly.derivative_point_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    build_times['stream_matern_sparse_factor_seconds'] = time.perf_counter() - start

    for kernel_kind, dense_factor, sparse_factor in (
        ('empirical', stream_empirical_dense_factor, stream_empirical_sparse_factor),
        ('matern', stream_matern_dense_factor, stream_matern_sparse_factor),
    ):
        start = time.perf_counter()
        streamfunction_operators[kernel_kind]['dense'] = precompute_streamfunction_operator(
            dense_factor,
            n_points=points.shape[0],
            dtype=points.dtype,
            device=points.device,
            damping=streamfunction_damping,
        )
        build_times[f'stream_{kernel_kind}_dense_operator_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        streamfunction_operators[kernel_kind]['sparse'] = precompute_streamfunction_operator(
            sparse_factor,
            n_points=points.shape[0],
            dtype=points.dtype,
            device=points.device,
            damping=streamfunction_damping,
        )
        build_times[f'stream_{kernel_kind}_sparse_operator_seconds'] = time.perf_counter() - start

    return NavierStokesFactorBundle(
        points=points,
        ordering=ordering,
        vorticity_factors=vorticity_factors,
        streamfunction_operators=streamfunction_operators,
        build_times=build_times,
    )


def rollout_navier_stokes_krom(
    initial_omega: torch.Tensor,
    initial_dx: torch.Tensor,
    initial_dy: torch.Tensor,
    initial_laplace: torch.Tensor,
    vorticity_factor: object,
    config: NSVorticityConfig,
    gn_steps: int = 2,
    gn_damping: float = 1e-8,
    poisson_solver: PoissonSolveMethod = 'fft',
    streamfunction_operator: object | None = None,
    cg_max_iter: int | None = None,
    cg_tol: float = 1e-8,
) -> dict[str, torch.Tensor]:
    n_points = config.nx * config.nx
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps // config.snapshot_stride + 1

    solutions = torch.zeros((n_points, num_snapshots), dtype=initial_omega.dtype, device=initial_omega.device)
    gradients_x = torch.zeros_like(solutions)
    gradients_y = torch.zeros_like(solutions)
    laplacians = torch.zeros_like(solutions)
    streamfunctions = torch.zeros_like(solutions)
    velocities_u = torch.zeros_like(solutions)
    velocities_v = torch.zeros_like(solutions)

    current_w = initial_omega.reshape(-1).clone()
    current_wx = initial_dx.reshape(-1).clone()
    current_wy = initial_dy.reshape(-1).clone()
    current_laplace = initial_laplace.reshape(-1).clone()
    state = torch.cat([current_w, current_wx, current_wy], dim=0)
    snapshot_index = 0

    for step in range(total_steps + 1):
        omega_grid = current_w.reshape(config.nx, config.nx)
        if poisson_solver == 'fft':
            stream = solve_streamfunction_with_fft(omega_grid, config)
        else:
            if streamfunction_operator is None:
                raise ValueError('streamfunction_operator is required when poisson_solver="kernel".')
            stream = solve_streamfunction_with_kernel(omega_grid, streamfunction_operator)

        if step % config.snapshot_stride == 0:
            solutions[:, snapshot_index] = current_w
            gradients_x[:, snapshot_index] = current_wx
            gradients_y[:, snapshot_index] = current_wy
            laplacians[:, snapshot_index] = current_laplace
            streamfunctions[:, snapshot_index] = stream['psi'].reshape(-1)
            velocities_u[:, snapshot_index] = stream['u'].reshape(-1)
            velocities_v[:, snapshot_index] = stream['v'].reshape(-1)
            snapshot_index += 1

        if step == total_steps:
            break

        previous_w = current_w.clone()
        previous_wx = current_wx.clone()
        previous_wy = current_wy.clone()
        previous_laplace = current_laplace.clone()
        velocity_u = stream['u'].reshape(-1)
        velocity_v = stream['v'].reshape(-1)

        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: navier_stokes_vorticity_residual_operator(
                z,
                previous_w,
                previous_wx,
                previous_wy,
                previous_laplace,
                velocity_u,
                velocity_v,
                config.viscosity,
                config.dt,
            ),
            factor=vorticity_factor,
            max_iter=gn_steps,
            damping=gn_damping,
            record_history=False,
            linear_solver='auto',
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: navier_stokes_vorticity_residual_and_jacobian(
                z,
                previous_w,
                previous_wx,
                previous_wy,
                previous_laplace,
                velocity_u,
                velocity_v,
                config.viscosity,
                config.dt,
            ),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state = result.state
        current_w = state[:n_points].clone()
        current_wx = state[n_points : 2 * n_points].clone()
        current_wy = state[2 * n_points : 3 * n_points].clone()
        current_laplace = 2.0 / config.viscosity * (
            (current_w - previous_w) / config.dt
            + 0.5 * (velocity_u * (current_wx + previous_wx) + velocity_v * (current_wy + previous_wy))
        ) - previous_laplace

    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=initial_omega.dtype, device=initial_omega.device)
    return {
        'times': times,
        'solutions': solutions,
        'gradients_x': gradients_x,
        'gradients_y': gradients_y,
        'laplacians': laplacians,
        'streamfunctions': streamfunctions,
        'velocity_u': velocities_u,
        'velocity_v': velocities_v,
    }


def evaluate_streamfunction_kernel(
    omega_snapshots: torch.Tensor,
    config: NSVorticityConfig,
    operator: PrecomputedStreamfunctionOperator,
) -> dict[str, float]:
    if omega_snapshots.ndim == 3:
        fields = omega_snapshots[:, :, -1]
    elif omega_snapshots.ndim == 2:
        fields = omega_snapshots
    else:
        raise ValueError('Expected omega_snapshots with shape (num_samples, num_points, num_times) or (num_samples, num_points).')

    psi_errors = []
    velocity_errors = []
    for index in range(fields.shape[0]):
        omega = fields[index].reshape(config.nx, config.nx)
        fft_solution = solve_streamfunction_with_fft(omega, config)
        kernel_solution = operator.solve(omega)
        psi_errors.append(relative_l2(kernel_solution['psi'].reshape(-1), fft_solution['psi'].reshape(-1)))
        velocity_errors.append(
            0.5
            * (
                relative_l2(kernel_solution['u'].reshape(-1), fft_solution['u'].reshape(-1))
                + relative_l2(kernel_solution['v'].reshape(-1), fft_solution['v'].reshape(-1))
            )
        )
    return {
        'mean_streamfunction_rel_l2': sum(psi_errors) / len(psi_errors),
        'mean_velocity_rel_l2': sum(velocity_errors) / len(velocity_errors),
    }
