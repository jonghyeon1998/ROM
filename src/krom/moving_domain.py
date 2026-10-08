from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Union

import torch

from .empirical import mixed_empirical_kernel, temporal_feature_matrix
from .factors import dense_precision_factor
from .gauss_newton import ResidualOperator, solve_gauss_newton
from .kernels import (
    matern52_2d,
    matern52_2d_dx1,
    matern52_2d_dx1dy1,
    matern52_2d_dx1dy2,
    matern52_2d_dx2,
    matern52_2d_dx2dy1,
    matern52_2d_dx2dy2,
    matern52_2d_dy1,
    matern52_2d_dy2,
    matern52_2d_laplace_x,
    matern52_2d_laplace_x_dy1,
    matern52_2d_laplace_x_dy2,
    matern52_2d_laplace_xy,
    matern52_2d_laplace_y,
    matern52_2d_laplace_y_dx1,
    matern52_2d_laplace_y_dx2,
)
from .ordering import MeasurementOrdering, build_measurement_ordering
from .sparse_cholesky import make_snapshot_kernel_source, sparse_precision_factor
from .workflows import KernelAssembly


AUTO_DIRECT_THRESHOLD = 2048


@dataclass
class MovingDomainHeatConfig:
    domain_length: float = 1.0
    nx: int = 31
    ny: int = 31
    dt: float = 1e-2
    tmax: float = 1.0
    diffusivity: float = 5e-3
    breathing_amplitude: float = 0.18
    breathing_cycles: float = 1.0
    newton_tol: float = 1e-10
    newton_max_iter: int = 2


@dataclass
class MovingDomainHeatProblem:
    coords: torch.Tensor
    interior_points: torch.Tensor
    boundary_points: torch.Tensor
    boundary_values: torch.Tensor
    interior_x_centered: torch.Tensor
    interior_y_centered: torch.Tensor
    dx_matrix: torch.Tensor
    dy_matrix: torch.Tensor
    laplacian_matrix: torch.Tensor
    n_int: int
    n_all: int


def _block_nugget(theta: torch.Tensor, block_sizes: tuple[int, ...], nugget: float) -> torch.Tensor:
    traces = []
    start = 0
    for block_size in block_sizes:
        block = theta[start : start + block_size, start : start + block_size]
        traces.append(torch.trace(block))
        start += block_size
    baseline = torch.clamp(traces[0], min=torch.finfo(theta.dtype).eps)
    diagonal_parts = []
    for trace, block_size in zip(traces, block_sizes):
        diagonal_parts.append((trace / baseline) * torch.ones(block_size, dtype=theta.dtype, device=theta.device))
    diagonal = torch.cat(diagonal_parts, dim=0)
    return theta + nugget * torch.diag(diagonal)


def _assemble_blocks(block_rows: tuple[tuple[torch.Tensor, ...], ...]) -> torch.Tensor:
    return torch.cat([torch.cat(row, dim=1) for row in block_rows], dim=0)


def build_moving_domain_flattened_coordinates(config: MovingDomainHeatConfig) -> torch.Tensor:
    x = torch.linspace(0.0, float(config.domain_length), config.nx, dtype=torch.float64)
    y = torch.linspace(0.0, float(config.domain_length), config.ny, dtype=torch.float64)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    interior = torch.stack([x_grid[1:-1, 1:-1].reshape(-1), y_grid[1:-1, 1:-1].reshape(-1)], dim=1)
    top = torch.stack([x_grid[0, :], y_grid[0, :]], dim=1)
    bottom = torch.stack([x_grid[-1, :], y_grid[-1, :]], dim=1)
    left = torch.stack([x_grid[1:-1, 0], y_grid[1:-1, 0]], dim=1)
    right = torch.stack([x_grid[1:-1, -1], y_grid[1:-1, -1]], dim=1)
    return torch.cat([interior, top, bottom, left, right], dim=0)


def build_moving_domain_problem(
    config: MovingDomainHeatConfig,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu',
) -> MovingDomainHeatProblem:
    coord_config = MovingDomainHeatConfig(
        domain_length=config.domain_length,
        nx=config.nx,
        ny=config.ny,
        dt=config.dt,
        tmax=config.tmax,
        diffusivity=config.diffusivity,
        breathing_amplitude=config.breathing_amplitude,
        breathing_cycles=config.breathing_cycles,
    )
    coords = build_moving_domain_flattened_coordinates(coord_config).to(device=device, dtype=dtype)
    n_int = (config.nx - 2) * (config.ny - 2)
    interior_points = coords[:n_int]
    boundary_points = coords[n_int:]
    boundary_values = torch.zeros(coords.shape[0] - n_int, dtype=dtype, device=device)
    center = 0.5 * float(config.domain_length)
    interior_x_centered = interior_points[:, 0] - center
    interior_y_centered = interior_points[:, 1] - center

    dx = float(config.domain_length) / (config.nx - 1)
    dy = float(config.domain_length) / (config.ny - 1)
    nx_int = config.nx - 2
    ny_int = config.ny - 2
    n_unknowns = nx_int * ny_int
    dx_matrix = torch.zeros((n_unknowns, n_unknowns), dtype=dtype, device=device)
    dy_matrix = torch.zeros((n_unknowns, n_unknowns), dtype=dtype, device=device)
    laplacian = torch.zeros((n_unknowns, n_unknowns), dtype=dtype, device=device)

    def idx(i: int, j: int) -> int:
        return i * ny_int + j

    for i in range(nx_int):
        for j in range(ny_int):
            row = idx(i, j)
            laplacian[row, row] = -2.0 / dx**2 - 2.0 / dy**2

            if i > 0:
                dx_matrix[row, idx(i - 1, j)] = -0.5 / dx
                laplacian[row, idx(i - 1, j)] = 1.0 / dx**2
            if i < nx_int - 1:
                dx_matrix[row, idx(i + 1, j)] = 0.5 / dx
                laplacian[row, idx(i + 1, j)] = 1.0 / dx**2
            if j > 0:
                dy_matrix[row, idx(i, j - 1)] = -0.5 / dy
                laplacian[row, idx(i, j - 1)] = 1.0 / dy**2
            if j < ny_int - 1:
                dy_matrix[row, idx(i, j + 1)] = 0.5 / dy
                laplacian[row, idx(i, j + 1)] = 1.0 / dy**2

    return MovingDomainHeatProblem(
        coords=coords,
        interior_points=interior_points,
        boundary_points=boundary_points,
        boundary_values=boundary_values,
        interior_x_centered=interior_x_centered,
        interior_y_centered=interior_y_centered,
        dx_matrix=dx_matrix,
        dy_matrix=dy_matrix,
        laplacian_matrix=laplacian,
        n_int=n_int,
        n_all=coords.shape[0],
    )


def breathing_scale(time_value: torch.Tensor | float, config: MovingDomainHeatConfig) -> torch.Tensor:
    time_tensor = torch.as_tensor(time_value, dtype=torch.float64)
    angle = 2.0 * math.pi * config.breathing_cycles * time_tensor / config.tmax
    return 1.0 + config.breathing_amplitude * torch.sin(angle)


def breathing_rate_ratio(time_value: torch.Tensor | float, config: MovingDomainHeatConfig) -> torch.Tensor:
    time_tensor = torch.as_tensor(time_value, dtype=torch.float64)
    angle = 2.0 * math.pi * config.breathing_cycles * time_tensor / config.tmax
    numerator = config.breathing_amplitude * (2.0 * math.pi * config.breathing_cycles / config.tmax) * torch.cos(angle)
    return numerator / breathing_scale(time_tensor, config)


def generate_moving_heat_initial_condition(
    problem: MovingDomainHeatProblem,
    num_modes: int = 4,
    amplitude: float = 0.35,
) -> torch.Tensor:
    x = problem.interior_points[:, 0]
    y = problem.interior_points[:, 1]
    field = torch.zeros(problem.n_int, dtype=x.dtype, device=x.device)
    coeffs = torch.randn((num_modes, num_modes), dtype=x.dtype, device=x.device)
    length = float(x.max().item() - x.min().item() + (x[1] - x[0]).item())
    for i in range(1, num_modes + 1):
        for j in range(1, num_modes + 1):
            field = field + coeffs[i - 1, j - 1] * torch.sin(i * math.pi * x / length) * torch.sin(j * math.pi * y / length)
    scale = torch.clamp(field.abs().max(), min=torch.finfo(field.dtype).eps)
    interior = amplitude * field / scale
    full = torch.zeros(problem.n_all, dtype=x.dtype, device=x.device)
    full[: problem.n_int] = interior
    return full


def moving_heat_linear_operator(
    problem: MovingDomainHeatProblem,
    config: MovingDomainHeatConfig,
    time_value: float,
) -> torch.Tensor:
    scale = breathing_scale(time_value, config).to(dtype=problem.coords.dtype, device=problem.coords.device)
    beta = breathing_rate_ratio(time_value, config).to(dtype=problem.coords.dtype, device=problem.coords.device)
    advection = beta * (
        torch.diag(problem.interior_x_centered).matmul(problem.dx_matrix)
        + torch.diag(problem.interior_y_centered).matmul(problem.dy_matrix)
    )
    diffusion = (config.diffusivity / scale.square()) * problem.laplacian_matrix
    return advection + diffusion


def moving_heat_rollout(
    initial_condition: torch.Tensor,
    problem: MovingDomainHeatProblem,
    config: MovingDomainHeatConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = initial_condition.dtype
    device = initial_condition.device
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps + 1

    solutions = torch.zeros((problem.n_all, num_snapshots), dtype=dtype, device=device)
    gradients_x = torch.zeros((problem.n_int, num_snapshots), dtype=dtype, device=device)
    gradients_y = torch.zeros((problem.n_int, num_snapshots), dtype=dtype, device=device)
    laplacians = torch.zeros((problem.n_int, num_snapshots), dtype=dtype, device=device)

    current = initial_condition[: problem.n_int].clone()
    eye = torch.eye(problem.n_int, dtype=dtype, device=device)
    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=dtype, device=device)

    solutions[: problem.n_int, 0] = current
    gradients_x[:, 0] = problem.dx_matrix.matmul(current)
    gradients_y[:, 0] = problem.dy_matrix.matmul(current)
    laplacians[:, 0] = problem.laplacian_matrix.matmul(current)

    for step in range(total_steps):
        time_prev = float(times[step].item())
        time_next = float(times[step + 1].item())
        operator_prev = moving_heat_linear_operator(problem, config, time_prev).to(dtype=dtype, device=device)
        operator_next = moving_heat_linear_operator(problem, config, time_next).to(dtype=dtype, device=device)

        lhs = eye - 0.5 * config.dt * operator_next
        rhs = (eye + 0.5 * config.dt * operator_prev).matmul(current)
        current = torch.linalg.solve(lhs, rhs)

        solutions[: problem.n_int, step + 1] = current
        gradients_x[:, step + 1] = problem.dx_matrix.matmul(current)
        gradients_y[:, step + 1] = problem.dy_matrix.matmul(current)
        laplacians[:, step + 1] = problem.laplacian_matrix.matmul(current)

    return times, solutions, gradients_x, gradients_y, laplacians


def moving_heat_dataset(
    num_samples: int,
    problem: MovingDomainHeatProblem,
    config: MovingDomainHeatConfig,
    num_modes: int = 4,
    amplitude: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps + 1
    solutions = torch.zeros((num_samples, problem.n_all, num_snapshots), dtype=problem.coords.dtype, device=problem.coords.device)
    gradients_x = torch.zeros((num_samples, problem.n_int, num_snapshots), dtype=problem.coords.dtype, device=problem.coords.device)
    gradients_y = torch.zeros_like(gradients_x)
    laplacians = torch.zeros_like(gradients_x)
    initial_conditions = torch.zeros((num_samples, problem.n_all), dtype=problem.coords.dtype, device=problem.coords.device)

    for sample in range(num_samples):
        initial = generate_moving_heat_initial_condition(problem, num_modes=num_modes, amplitude=amplitude)
        times, sol, grad_x, grad_y, lap = moving_heat_rollout(initial, problem, config)
        initial_conditions[sample] = initial
        solutions[sample] = sol
        gradients_x[sample] = grad_x
        gradients_y[sample] = grad_y
        laplacians[sample] = lap

    return times, solutions, gradients_x, gradients_y, laplacians


def build_moving_heat_matern_theta(
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    lengthscale: float,
    nugget: float = 1e-10,
) -> KernelAssembly:
    dirac_points = torch.cat([interior_points, boundary_points], dim=0)
    interior_indices = torch.arange(interior_points.shape[0], dtype=torch.long, device=dirac_points.device)
    uu = matern52_2d(dirac_points, dirac_points, lengthscale)
    u_dx = matern52_2d_dy1(dirac_points, interior_points, lengthscale)
    u_dy = matern52_2d_dy2(dirac_points, interior_points, lengthscale)
    u_lap = matern52_2d_laplace_y(dirac_points, interior_points, lengthscale)
    dx_u = matern52_2d_dx1(interior_points, dirac_points, lengthscale)
    dx_dx = matern52_2d_dx1dy1(interior_points, interior_points, lengthscale)
    dx_dy = matern52_2d_dx1dy2(interior_points, interior_points, lengthscale)
    dx_lap = matern52_2d_laplace_y_dx1(interior_points, interior_points, lengthscale)
    dy_u = matern52_2d_dx2(interior_points, dirac_points, lengthscale)
    dy_dx = matern52_2d_dx2dy1(interior_points, interior_points, lengthscale)
    dy_dy = matern52_2d_dx2dy2(interior_points, interior_points, lengthscale)
    dy_lap = matern52_2d_laplace_y_dx2(interior_points, interior_points, lengthscale)
    lap_u = matern52_2d_laplace_x(interior_points, dirac_points, lengthscale)
    lap_dx = matern52_2d_laplace_x_dy1(interior_points, interior_points, lengthscale)
    lap_dy = matern52_2d_laplace_x_dy2(interior_points, interior_points, lengthscale)
    lap_lap = matern52_2d_laplace_xy(interior_points, interior_points, lengthscale)
    theta = _assemble_blocks(
        ((uu, u_dx, u_dy, u_lap), (dx_u, dx_dx, dx_dy, dx_lap), (dy_u, dy_dx, dy_dy, dy_lap), (lap_u, lap_dx, lap_dy, lap_lap))
    )
    block_sizes = (dirac_points.shape[0], interior_points.shape[0], interior_points.shape[0], interior_points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(
        theta=theta,
        dirac_points=dirac_points,
        derivative_point_groups=(interior_indices, interior_indices, interior_indices),
        block_sizes=block_sizes,
    )


def build_moving_heat_empirical_theta(
    solution_features: torch.Tensor,
    dx_features: torch.Tensor,
    dy_features: torch.Tensor,
    laplacian_features: torch.Tensor,
    nugget: float = 1e-10,
) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, dx_features, dy_features, laplacian_features))
    block_sizes = (solution_features.shape[0], dx_features.shape[0], dy_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def moving_heat_residual_and_jacobian(
    state: torch.Tensor,
    previous_u: torch.Tensor,
    previous_ux: torch.Tensor,
    previous_uy: torch.Tensor,
    previous_lap: torch.Tensor,
    boundary_values: torch.Tensor,
    centered_x: torch.Tensor,
    centered_y: torch.Tensor,
    diffusivity_next: float,
    beta_prev: float,
    beta_next: float,
    diffusivity_prev: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_int = previous_u.numel()
    u = state[:n_int]
    ux = state[n_int : 2 * n_int]
    uy = state[2 * n_int :]
    prev_contrib = 0.5 * (
        beta_prev * (centered_x * previous_ux + centered_y * previous_uy) + diffusivity_prev * previous_lap
    )
    lap = (2.0 / diffusivity_next) * ((u - previous_u) / dt - prev_contrib) - (beta_next / diffusivity_next) * (
        centered_x * ux + centered_y * uy
    )
    residual = torch.cat([u, boundary_values, ux, uy, lap], dim=0)
    jacobian = torch.zeros((n_int + boundary_values.numel() + 3 * n_int, 3 * n_int), dtype=state.dtype, device=state.device)
    identity = torch.eye(n_int, dtype=state.dtype, device=state.device)
    jacobian[:n_int, :n_int] = identity
    jacobian[n_int + boundary_values.numel() : n_int + boundary_values.numel() + n_int, n_int : 2 * n_int] = identity
    jacobian[n_int + boundary_values.numel() + n_int : n_int + boundary_values.numel() + 2 * n_int, 2 * n_int :] = identity
    lap_start = n_int + boundary_values.numel() + 2 * n_int
    jacobian[lap_start:, :n_int] = (2.0 / diffusivity_next / dt) * identity
    jacobian[lap_start:, n_int : 2 * n_int] = -(beta_next / diffusivity_next) * torch.diag(centered_x)
    jacobian[lap_start:, 2 * n_int :] = -(beta_next / diffusivity_next) * torch.diag(centered_y)
    return residual, jacobian


def moving_heat_residual_operator(
    state: torch.Tensor,
    previous_u: torch.Tensor,
    previous_ux: torch.Tensor,
    previous_uy: torch.Tensor,
    previous_lap: torch.Tensor,
    boundary_values: torch.Tensor,
    centered_x: torch.Tensor,
    centered_y: torch.Tensor,
    diffusivity_next: float,
    beta_prev: float,
    beta_next: float,
    diffusivity_prev: float,
    dt: float,
) -> ResidualOperator:
    n_int = previous_u.numel()
    u = state[:n_int]
    ux = state[n_int : 2 * n_int]
    uy = state[2 * n_int :]
    zeros_boundary = torch.zeros(boundary_values.numel(), dtype=state.dtype, device=state.device)
    coeff_u = torch.full_like(u, 2.0 / diffusivity_next / dt)
    coeff_ux = -(beta_next / diffusivity_next) * centered_x
    coeff_uy = -(beta_next / diffusivity_next) * centered_y
    prev_contrib = 0.5 * (
        beta_prev * (centered_x * previous_ux + centered_y * previous_uy) + diffusivity_prev * previous_lap
    )
    lap = (2.0 / diffusivity_next) * ((u - previous_u) / dt - prev_contrib) + coeff_ux * ux + coeff_uy * uy
    residual = torch.cat([u, boundary_values, ux, uy, lap], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        du = vector[:n_int]
        dux = vector[n_int : 2 * n_int]
        duy = vector[2 * n_int :]
        return torch.cat([du, zeros_boundary, dux, duy, coeff_u * du + coeff_ux * dux + coeff_uy * duy], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        weight_u = vector[:n_int]
        weight_ux = vector[n_int + boundary_values.numel() : n_int + boundary_values.numel() + n_int]
        weight_uy = vector[n_int + boundary_values.numel() + n_int : n_int + boundary_values.numel() + 2 * n_int]
        weight_lap = vector[n_int + boundary_values.numel() + 2 * n_int :]
        return torch.cat(
            [
                weight_u + coeff_u * weight_lap,
                weight_ux + coeff_ux * weight_lap,
                weight_uy + coeff_uy * weight_lap,
            ],
            dim=0,
        )

    diagonal = torch.cat([1.0 + coeff_u.square(), 1.0 + coeff_ux.square(), 1.0 + coeff_uy.square()], dim=0)
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def build_moving_heat_factors(
    problem: MovingDomainHeatProblem,
    train_solutions: torch.Tensor,
    train_dx: torch.Tensor,
    train_dy: torch.Tensor,
    train_laplacians: torch.Tensor,
    rho: float,
    lengthscale: float,
    nugget: float = 1e-9,
    burnin: int = 0,
    stride: int = 1,
    sparse_backend: str = 'auto',
    k_neighbors: int = 3,
) -> dict[str, object]:
    derivative_groups = (
        torch.arange(problem.n_int, dtype=torch.long, device=problem.coords.device),
        torch.arange(problem.n_int, dtype=torch.long, device=problem.coords.device),
        torch.arange(problem.n_int, dtype=torch.long, device=problem.coords.device),
    )
    start = time.perf_counter()
    # k_neighbors=3 aligns with the official repo's stated default for PDE
    # problems with derivative measurements. Boundary points here carry no
    # derivative measurements (groups only cover the n_int interior points),
    # so "follow_diracs" isn't structurally applicable; stays on the default
    # "dirac_first_then_unif_scale" variant.
    ordering = build_measurement_ordering(problem.coords, derivative_groups, k_neighbors=k_neighbors)
    ordering_seconds = time.perf_counter() - start

    solution_features = temporal_feature_matrix(train_solutions, burnin=burnin, stride=stride)
    dx_features = temporal_feature_matrix(train_dx, burnin=burnin, stride=stride)
    dy_features = temporal_feature_matrix(train_dy, burnin=burnin, stride=stride)
    lap_features = temporal_feature_matrix(train_laplacians, burnin=burnin, stride=stride)
    empirical_theta = build_moving_heat_empirical_theta(
        solution_features,
        dx_features,
        dy_features,
        lap_features,
        nugget=nugget,
    ) / solution_features.shape[1]
    matern_assembly = build_moving_heat_matern_theta(
        problem.interior_points,
        problem.boundary_points,
        lengthscale=lengthscale,
        nugget=nugget,
    )

    start = time.perf_counter()
    empirical_dense = dense_precision_factor(empirical_theta, nugget=nugget)
    empirical_dense_seconds = time.perf_counter() - start
    M = solution_features.shape[1]
    empirical_ks = make_snapshot_kernel_source(
        (solution_features, dx_features, dy_features, lap_features),
        nugget=nugget,
        scale=1.0 / M,
    )
    start = time.perf_counter()
    empirical_sparse, _ = sparse_precision_factor(
        theta=empirical_ks,
        dirac_points=problem.coords,
        derivative_point_groups=derivative_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    empirical_sparse_seconds = time.perf_counter() - start

    start = time.perf_counter()
    matern_dense = dense_precision_factor(matern_assembly.theta, nugget=nugget)
    matern_dense_seconds = time.perf_counter() - start
    start = time.perf_counter()
    matern_sparse, _ = sparse_precision_factor(
        theta=matern_assembly.theta,
        dirac_points=matern_assembly.dirac_points,
        derivative_point_groups=matern_assembly.derivative_point_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    matern_sparse_seconds = time.perf_counter() - start

    return {
        'ordering': ordering,
        'empirical_sparse': empirical_sparse,
        'empirical_dense': empirical_dense,
        'matern_sparse': matern_sparse,
        'matern_dense': matern_dense,
        'build_times': {
            'ordering_seconds': ordering_seconds,
            'empirical_dense_factor_seconds': empirical_dense_seconds,
            'empirical_sparse_factor_seconds': empirical_sparse_seconds,
            'matern_dense_factor_seconds': matern_dense_seconds,
            'matern_sparse_factor_seconds': matern_sparse_seconds,
        },
    }


def rollout_moving_heat_krom(
    initial_solution: torch.Tensor,
    initial_dx: torch.Tensor,
    initial_dy: torch.Tensor,
    initial_laplace: torch.Tensor,
    factor: object,
    problem: MovingDomainHeatProblem,
    config: MovingDomainHeatConfig,
    gn_steps: int = 1,
    gn_damping: float = 1e-8,
    cg_max_iter: int | None = None,
    cg_tol: float = 1e-8,
) -> dict[str, torch.Tensor]:
    total_steps = int(round(config.tmax / config.dt))
    num_snapshots = total_steps + 1
    times = torch.linspace(0.0, config.tmax, num_snapshots, dtype=initial_solution.dtype, device=initial_solution.device)

    solutions = torch.zeros((problem.n_all, num_snapshots), dtype=initial_solution.dtype, device=initial_solution.device)
    gradients_x = torch.zeros((problem.n_int, num_snapshots), dtype=initial_solution.dtype, device=initial_solution.device)
    gradients_y = torch.zeros_like(gradients_x)
    laplacians = torch.zeros_like(gradients_x)

    previous_u = initial_solution[: problem.n_int].clone()
    previous_ux = initial_dx.clone()
    previous_uy = initial_dy.clone()
    previous_lap = initial_laplace.clone()
    state = torch.cat([previous_u, previous_ux, previous_uy], dim=0)

    solutions[:, 0] = initial_solution
    gradients_x[:, 0] = previous_ux
    gradients_y[:, 0] = previous_uy
    laplacians[:, 0] = previous_lap

    for step in range(total_steps):
        time_prev = float(times[step].item())
        time_next = float(times[step + 1].item())
        scale_prev = breathing_scale(time_prev, config).to(dtype=previous_u.dtype, device=previous_u.device)
        scale_next = breathing_scale(time_next, config).to(dtype=previous_u.dtype, device=previous_u.device)
        beta_prev = breathing_rate_ratio(time_prev, config).to(dtype=previous_u.dtype, device=previous_u.device)
        beta_next = breathing_rate_ratio(time_next, config).to(dtype=previous_u.dtype, device=previous_u.device)
        diffusivity_prev = config.diffusivity / float(scale_prev.square().item())
        diffusivity_next = config.diffusivity / float(scale_next.square().item())

        saved_u = previous_u.clone()
        saved_ux = previous_ux.clone()
        saved_uy = previous_uy.clone()
        saved_lap = previous_lap.clone()
        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: moving_heat_residual_operator(
                z,
                saved_u,
                saved_ux,
                saved_uy,
                saved_lap,
                problem.boundary_values,
                problem.interior_x_centered,
                problem.interior_y_centered,
                diffusivity_next,
                float(beta_prev.item()),
                float(beta_next.item()),
                diffusivity_prev,
                config.dt,
            ),
            factor=factor,
            max_iter=gn_steps,
            damping=gn_damping,
            record_history=False,
            linear_solver='auto',
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: moving_heat_residual_and_jacobian(
                z,
                saved_u,
                saved_ux,
                saved_uy,
                saved_lap,
                problem.boundary_values,
                problem.interior_x_centered,
                problem.interior_y_centered,
                diffusivity_next,
                float(beta_prev.item()),
                float(beta_next.item()),
                diffusivity_prev,
                config.dt,
            ),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state = result.state
        previous_u = state[: problem.n_int].clone()
        previous_ux = state[problem.n_int : 2 * problem.n_int].clone()
        previous_uy = state[2 * problem.n_int :].clone()
        prev_contrib = 0.5 * (
            beta_prev * (problem.interior_x_centered * saved_ux + problem.interior_y_centered * saved_uy)
            + diffusivity_prev * saved_lap
        )
        previous_lap = (2.0 / diffusivity_next) * ((previous_u - saved_u) / config.dt - prev_contrib) - (
            beta_next / diffusivity_next
        ) * (problem.interior_x_centered * previous_ux + problem.interior_y_centered * previous_uy)

        solutions[: problem.n_int, step + 1] = previous_u
        gradients_x[:, step + 1] = previous_ux
        gradients_y[:, step + 1] = previous_uy
        laplacians[:, step + 1] = previous_lap

    return {
        'times': times,
        'solutions': solutions,
        'gradients_x': gradients_x,
        'gradients_y': gradients_y,
        'laplacians': laplacians,
    }
