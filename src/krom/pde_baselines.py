from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BurgersCNConfig:
    x_start: float = -1.0
    x_end: float = 1.0
    nx: int = 201
    dt: float = 1e-2
    tmax: float = 1.0
    viscosity: float = 1e-3
    newton_tol: float = 1e-10
    newton_max_iter: int = 20


@dataclass
class AllenCahnCNConfig:
    epsilon: float = 1e-2
    domain_length: float = 2.0 * torch.pi
    nx: int = 61
    ny: int = 61
    dt: float = 1e-2
    tmax: float = 5.0
    newton_tol: float = 1e-10
    newton_max_iter: int = 25


def generate_burgers_initial_condition(
    x: torch.Tensor,
    num_terms: int = 4,
    amplitude: float = 0.5,
) -> torch.Tensor:
    shifted = (x - x[0]) / (x[-1] - x[0])
    coeffs = torch.randn(num_terms, dtype=x.dtype, device=x.device)
    u0 = torch.zeros_like(x)
    for mode in range(1, num_terms + 1):
        u0 = u0 + coeffs[mode - 1] * torch.sin(mode * torch.pi * shifted)
    scale = torch.clamp(u0.abs().max(), min=torch.finfo(u0.dtype).eps)
    return amplitude * u0 / scale


def _burgers_operators(config: BurgersCNConfig, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(config.x_start, config.x_end, config.nx, dtype=dtype, device=device)
    dx = x[1] - x[0]
    n_int = config.nx - 2
    d1 = torch.zeros((n_int, n_int), dtype=dtype, device=device)
    d2 = torch.zeros((n_int, n_int), dtype=dtype, device=device)
    for i in range(n_int):
        d2[i, i] = -2.0 / dx.square()
        if i > 0:
            d1[i, i - 1] = -0.5 / dx
            d2[i, i - 1] = 1.0 / dx.square()
        if i < n_int - 1:
            d1[i, i + 1] = 0.5 / dx
            d2[i, i + 1] = 1.0 / dx.square()
    return x, d1, d2


def _burgers_residual_and_jacobian(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    config: BurgersCNConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv_prev = u_prev * (d1.matmul(u_prev))
    adv_next = u_next * (d1.matmul(u_next))
    diff_prev = config.viscosity * d2.matmul(u_prev)
    diff_next = config.viscosity * d2.matmul(u_next)
    residual = u_next - u_prev + 0.5 * config.dt * (adv_prev + adv_next - diff_prev - diff_next)
    jacobian = (
        torch.eye(u_next.numel(), dtype=u_next.dtype, device=u_next.device)
        + 0.5 * config.dt * (torch.diag(d1.matmul(u_next)) + torch.diag(u_next).matmul(d1) - config.viscosity * d2)
    )
    return residual, jacobian


def burgers_crank_nicolson_rollout(
    initial_condition: torch.Tensor,
    config: BurgersCNConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = initial_condition.dtype
    device = initial_condition.device
    x, d1, d2 = _burgers_operators(config, dtype=dtype, device=device)
    n_steps = int(round(config.tmax / config.dt)) + 1
    interior = initial_condition[1:-1].clone()

    snapshots = torch.zeros((config.nx, n_steps), dtype=dtype, device=device)
    grad_snapshots = torch.zeros_like(snapshots)
    lap_snapshots = torch.zeros_like(snapshots)
    snapshots[:, 0] = initial_condition

    def reconstruct(interior_state: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(config.nx, dtype=dtype, device=device)
        full[1:-1] = interior_state
        return full

    def derivative_fields(interior_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = reconstruct(interior_state)
        grad = torch.zeros_like(full)
        grad[1:-1] = d1.matmul(interior_state)
        lap = torch.zeros_like(full)
        lap[1:-1] = d2.matmul(interior_state)
        return grad, lap

    grad_snapshots[:, 0], lap_snapshots[:, 0] = derivative_fields(interior)

    for step in range(1, n_steps):
        current = interior.clone()
        for _ in range(config.newton_max_iter):
            residual, jacobian = _burgers_residual_and_jacobian(current, interior, d1, d2, config)
            delta = torch.linalg.solve(jacobian, residual)
            current = current - delta
            if torch.linalg.norm(delta) <= config.newton_tol * max(1.0, float(torch.linalg.norm(current))):
                break
        interior = current
        snapshots[:, step] = reconstruct(interior)
        grad_snapshots[:, step], lap_snapshots[:, step] = derivative_fields(interior)

    times = torch.linspace(0.0, config.tmax, n_steps, dtype=dtype, device=device)
    return x, times, snapshots, grad_snapshots, lap_snapshots


def burgers_crank_nicolson_dataset(
    num_samples: int,
    config: BurgersCNConfig,
    num_terms: int = 4,
    amplitude: float = 0.5,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(config.x_start, config.x_end, config.nx, dtype=dtype, device=device)
    n_steps = int(round(config.tmax / config.dt)) + 1
    solutions = torch.zeros((num_samples, config.nx, n_steps), dtype=dtype, device=device)
    gradients = torch.zeros_like(solutions)
    laplacians = torch.zeros_like(solutions)

    for sample in range(num_samples):
        u0 = generate_burgers_initial_condition(x, num_terms=num_terms, amplitude=amplitude)
        _, times, sol, grad, lap = burgers_crank_nicolson_rollout(u0, config)
        solutions[sample] = sol
        gradients[sample] = grad
        laplacians[sample] = lap

    return x, times, solutions, gradients, laplacians


def generate_allen_cahn_initial_condition(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    num_modes: int = 5,
    amplitude: float = 0.25,
) -> torch.Tensor:
    coeffs = torch.randn((num_modes, num_modes), dtype=x_grid.dtype, device=x_grid.device)
    u0 = torch.zeros_like(x_grid)
    for mode_x in range(1, num_modes + 1):
        for mode_y in range(1, num_modes + 1):
            u0 = u0 + coeffs[mode_x - 1, mode_y - 1] * torch.sin(mode_x * x_grid) * torch.sin(mode_y * y_grid)
    scale = torch.clamp(u0.abs().max(), min=torch.finfo(u0.dtype).eps)
    return amplitude * u0 / scale


def _allen_cahn_laplacian_matrix(config: AllenCahnCNConfig, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    nx_int = config.nx - 2
    ny_int = config.ny - 2
    dx = config.domain_length / (config.nx - 1)
    dy = config.domain_length / (config.ny - 1)

    tx = torch.diag(torch.full((nx_int,), -2.0 / dx**2, dtype=dtype, device=device))
    ty = torch.diag(torch.full((ny_int,), -2.0 / dy**2, dtype=dtype, device=device))
    tx = tx + torch.diag(torch.full((nx_int - 1,), 1.0 / dx**2, dtype=dtype, device=device), diagonal=1)
    tx = tx + torch.diag(torch.full((nx_int - 1,), 1.0 / dx**2, dtype=dtype, device=device), diagonal=-1)
    ty = ty + torch.diag(torch.full((ny_int - 1,), 1.0 / dy**2, dtype=dtype, device=device), diagonal=1)
    ty = ty + torch.diag(torch.full((ny_int - 1,), 1.0 / dy**2, dtype=dtype, device=device), diagonal=-1)

    eye_x = torch.eye(nx_int, dtype=dtype, device=device)
    eye_y = torch.eye(ny_int, dtype=dtype, device=device)
    return torch.kron(eye_y, tx) + torch.kron(ty, eye_x)


def solve_allen_cahn_crank_nicolson(
    initial_condition: torch.Tensor,
    config: AllenCahnCNConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = initial_condition.dtype
    device = initial_condition.device
    nx_int = config.nx - 2
    ny_int = config.ny - 2
    n_steps = int(round(config.tmax / config.dt)) + 1
    laplacian = _allen_cahn_laplacian_matrix(config, dtype=dtype, device=device)

    snapshots = torch.zeros((n_steps, config.nx, config.ny), dtype=dtype, device=device)
    laplacian_snapshots = torch.zeros_like(snapshots)
    snapshots[0] = initial_condition

    interior = initial_condition[1:-1, 1:-1].reshape(-1).clone()

    def reaction(u_vec: torch.Tensor) -> torch.Tensor:
        return u_vec - u_vec.pow(3)

    def full_from_interior(u_vec: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((config.nx, config.ny), dtype=dtype, device=device)
        full[1:-1, 1:-1] = u_vec.reshape(nx_int, ny_int)
        return full

    def lap_from_interior(u_vec: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((config.nx, config.ny), dtype=dtype, device=device)
        full[1:-1, 1:-1] = laplacian.matmul(u_vec).reshape(nx_int, ny_int)
        return full

    laplacian_snapshots[0] = lap_from_interior(interior)

    eye = torch.eye(interior.numel(), dtype=dtype, device=device)
    for step in range(1, n_steps):
        previous = interior.clone()
        current = previous.clone()
        lap_prev = laplacian.matmul(previous)
        reaction_prev = reaction(previous)

        for _ in range(config.newton_max_iter):
            lap_current = laplacian.matmul(current)
            reaction_current = reaction(current)
            residual = current - previous - 0.5 * config.dt * (
                config.epsilon**2 * (lap_current + lap_prev) + reaction_current + reaction_prev
            )
            jacobian = eye - 0.5 * config.dt * (
                config.epsilon**2 * laplacian + torch.diag(1.0 - 3.0 * current.square())
            )
            delta = torch.linalg.solve(jacobian, residual)
            current = current - delta
            if torch.linalg.norm(delta) <= config.newton_tol * max(1.0, float(torch.linalg.norm(current))):
                break

        interior = current
        snapshots[step] = full_from_interior(interior)
        laplacian_snapshots[step] = lap_from_interior(interior)

    return snapshots, laplacian_snapshots


def build_allen_cahn_flattened_coordinates(config: AllenCahnCNConfig) -> torch.Tensor:
    x = torch.linspace(0.0, float(config.domain_length), config.nx, dtype=torch.float64)
    y = torch.linspace(0.0, float(config.domain_length), config.ny, dtype=torch.float64)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")

    interior = torch.stack([x_grid[1:-1, 1:-1].reshape(-1), y_grid[1:-1, 1:-1].reshape(-1)], dim=1)
    top = torch.stack([x_grid[0, :], y_grid[0, :]], dim=1)
    bottom = torch.stack([x_grid[-1, :], y_grid[-1, :]], dim=1)
    left = torch.stack([x_grid[1:-1, 0], y_grid[1:-1, 0]], dim=1)
    right = torch.stack([x_grid[1:-1, -1], y_grid[1:-1, -1]], dim=1)
    return torch.cat([interior, top, bottom, left, right], dim=0)
