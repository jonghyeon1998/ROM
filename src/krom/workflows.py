from __future__ import annotations

from dataclasses import dataclass

import torch

from .empirical import mixed_empirical_kernel
from .gauss_newton import ResidualOperator
from .kernels import (
    matern52_1d,
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


@dataclass
class KernelAssembly:
    theta: torch.Tensor
    dirac_points: torch.Tensor
    derivative_point_groups: tuple[torch.Tensor, ...]
    block_sizes: tuple[int, ...]


def _block_nugget(theta: torch.Tensor, block_sizes: tuple[int, ...], nugget: float) -> torch.Tensor:
    traces = []
    start = 0
    for block_size in block_sizes:
        block = theta[start : start + block_size, start : start + block_size]
        traces.append(torch.trace(block))
        start += block_size
    baseline = torch.clamp(traces[0], min=torch.finfo(theta.dtype).eps)
    weights = []
    for trace, block_size in zip(traces, block_sizes):
        weights.append((trace / baseline) * torch.ones(block_size, dtype=theta.dtype, device=theta.device))
    diagonal = torch.cat(weights, dim=0)
    return theta + nugget * torch.diag(diagonal)


def _assemble_blocks(block_rows: tuple[tuple[torch.Tensor, ...], ...]) -> torch.Tensor:
    return torch.cat([torch.cat(row, dim=1) for row in block_rows], dim=0)


def build_burgers_matern_theta(interior_points: torch.Tensor, boundary_points: torch.Tensor, lengthscale: float, nugget: float = 1e-10) -> KernelAssembly:
    dirac_points = torch.cat([interior_points, boundary_points], dim=0)
    interior_indices = torch.arange(interior_points.shape[0], dtype=torch.long, device=dirac_points.device)
    uu = matern52_1d(dirac_points, dirac_points, lengthscale, 0, 0)
    u_ux = matern52_1d(dirac_points, interior_points, lengthscale, 0, 1)
    u_uxx = matern52_1d(dirac_points, interior_points, lengthscale, 0, 2)
    ux_u = matern52_1d(interior_points, dirac_points, lengthscale, 1, 0)
    ux_ux = matern52_1d(interior_points, interior_points, lengthscale, 1, 1)
    ux_uxx = matern52_1d(interior_points, interior_points, lengthscale, 1, 2)
    uxx_u = matern52_1d(interior_points, dirac_points, lengthscale, 2, 0)
    uxx_ux = matern52_1d(interior_points, interior_points, lengthscale, 2, 1)
    uxx_uxx = matern52_1d(interior_points, interior_points, lengthscale, 2, 2)
    theta = _assemble_blocks(((uu, u_ux, u_uxx), (ux_u, ux_ux, ux_uxx), (uxx_u, uxx_ux, uxx_uxx)))
    block_sizes = (dirac_points.shape[0], interior_points.shape[0], interior_points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(theta=theta, dirac_points=dirac_points, derivative_point_groups=(interior_indices, interior_indices), block_sizes=block_sizes)


def build_burgers_empirical_theta(solution_features: torch.Tensor, gradient_features: torch.Tensor, laplacian_features: torch.Tensor, nugget: float = 1e-10) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, gradient_features, laplacian_features))
    block_sizes = (solution_features.shape[0], gradient_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def burgers_residual_and_jacobian(state: torch.Tensor, previous_u: torch.Tensor, previous_ux: torch.Tensor, previous_uxx: torch.Tensor, boundary_values: torch.Tensor, viscosity: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_int = previous_u.numel()
    u = state[:n_int]
    ux = state[n_int:]
    uxx = 2.0 / viscosity * ((u - previous_u) / dt + 0.5 * (u * ux + previous_u * previous_ux)) - previous_uxx
    residual = torch.cat([u, boundary_values, ux, uxx], dim=0)
    jacobian = torch.zeros((n_int + boundary_values.numel() + 2 * n_int, 2 * n_int), dtype=state.dtype, device=state.device)
    identity = torch.eye(n_int, dtype=state.dtype, device=state.device)
    jacobian[:n_int, :n_int] = identity
    jacobian[n_int + boundary_values.numel() : n_int + boundary_values.numel() + n_int, n_int:] = identity
    jacobian[n_int + boundary_values.numel() + n_int :, :n_int] = (2.0 / viscosity) * (identity / dt + 0.5 * torch.diag(ux))
    jacobian[n_int + boundary_values.numel() + n_int :, n_int:] = (1.0 / viscosity) * torch.diag(u)
    return residual, jacobian


def burgers_residual_operator(state: torch.Tensor, previous_u: torch.Tensor, previous_ux: torch.Tensor, previous_uxx: torch.Tensor, boundary_values: torch.Tensor, viscosity: float, dt: float) -> ResidualOperator:
    n_int = previous_u.numel()
    u = state[:n_int]
    ux = state[n_int:]
    zeros_boundary = torch.zeros(boundary_values.numel(), dtype=state.dtype, device=state.device)
    coeff_u = (2.0 / viscosity) * (1.0 / dt + 0.5 * ux)
    coeff_ux = (1.0 / viscosity) * u
    residual_uxx = 2.0 / viscosity * ((u - previous_u) / dt + 0.5 * (u * ux + previous_u * previous_ux)) - previous_uxx
    residual = torch.cat([u, boundary_values, ux, residual_uxx], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        du = vector[:n_int]
        dux = vector[n_int:]
        return torch.cat([du, zeros_boundary, dux, coeff_u * du + coeff_ux * dux], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        weight_u = vector[:n_int]
        weight_ux = vector[n_int + boundary_values.numel() : n_int + boundary_values.numel() + n_int]
        weight_uxx = vector[n_int + boundary_values.numel() + n_int :]
        return torch.cat([weight_u + coeff_u * weight_uxx, weight_ux + coeff_ux * weight_uxx], dim=0)

    diagonal = torch.cat([1.0 + coeff_u.square(), 1.0 + coeff_ux.square()], dim=0)
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def build_allen_cahn_empirical_theta(solution_features: torch.Tensor, laplacian_features: torch.Tensor, nugget: float = 1e-10) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, laplacian_features))
    block_sizes = (solution_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def allen_cahn_residual_and_jacobian(state: torch.Tensor, previous_u: torch.Tensor, previous_laplace: torch.Tensor, boundary_values: torch.Tensor, epsilon: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    laplace = 2.0 / epsilon**2 * ((state - previous_u) / dt - 0.5 * ((previous_u - previous_u.pow(3)) + (state - state.pow(3)))) - previous_laplace
    residual = torch.cat([state, boundary_values, laplace], dim=0)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = (2.0 / epsilon**2) * (identity / dt - 0.5 * (identity - 3.0 * torch.diag(state.square())))
    return residual, jacobian


def allen_cahn_residual_operator(state: torch.Tensor, previous_u: torch.Tensor, previous_laplace: torch.Tensor, boundary_values: torch.Tensor, epsilon: float, dt: float) -> ResidualOperator:
    zeros_boundary = torch.zeros(boundary_values.numel(), dtype=state.dtype, device=state.device)
    coeff = (2.0 / epsilon**2) * (1.0 / dt - 0.5 * (1.0 - 3.0 * state.square()))
    laplace = 2.0 / epsilon**2 * ((state - previous_u) / dt - 0.5 * ((previous_u - previous_u.pow(3)) + (state - state.pow(3)))) - previous_laplace
    residual = torch.cat([state, boundary_values, laplace], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        return torch.cat([vector, zeros_boundary, coeff * vector], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        return vector[: state.numel()] + coeff * vector[state.numel() + boundary_values.numel() :]

    diagonal = 1.0 + coeff.square()
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def build_elliptic_matern_theta(interior_points: torch.Tensor, boundary_points: torch.Tensor, lengthscale: float, nugget: float = 1e-10) -> KernelAssembly:
    dirac_points = torch.cat([interior_points, boundary_points], dim=0)
    interior_indices = torch.arange(interior_points.shape[0], dtype=torch.long, device=dirac_points.device)
    uu = matern52_2d(dirac_points, dirac_points, lengthscale)
    lap_u = -matern52_2d_laplace_x(interior_points, dirac_points, lengthscale)
    lap_lap = matern52_2d_laplace_xy(interior_points, interior_points, lengthscale)
    theta = _assemble_blocks(((uu, lap_u.transpose(0, 1)), (lap_u, lap_lap)))
    block_sizes = (dirac_points.shape[0], interior_points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(theta=theta, dirac_points=dirac_points, derivative_point_groups=(interior_indices,), block_sizes=block_sizes)


def build_elliptic_empirical_theta(solution_features: torch.Tensor, nonlinear_rhs_features: torch.Tensor, nugget: float = 1e-10) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, nonlinear_rhs_features))
    block_sizes = (solution_features.shape[0], nonlinear_rhs_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def elliptic_residual_and_jacobian(state: torch.Tensor, rhs_values: torch.Tensor, boundary_values: torch.Tensor, alpha: float, power: int) -> tuple[torch.Tensor, torch.Tensor]:
    nonlinear = rhs_values - alpha * state.pow(power)
    residual = torch.cat([state, boundary_values, nonlinear], dim=0)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = -alpha * power * torch.diag(state.pow(power - 1))
    return residual, jacobian


def elliptic_residual_operator(state: torch.Tensor, rhs_values: torch.Tensor, boundary_values: torch.Tensor, alpha: float, power: int) -> ResidualOperator:
    zeros_boundary = torch.zeros(boundary_values.numel(), dtype=state.dtype, device=state.device)
    coeff = -alpha * power * state.pow(power - 1)
    residual = torch.cat([state, boundary_values, rhs_values - alpha * state.pow(power)], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        return torch.cat([vector, zeros_boundary, coeff * vector], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        return vector[: state.numel()] + coeff * vector[state.numel() + boundary_values.numel() :]

    diagonal = 1.0 + coeff.square()
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def darcy_residual_and_jacobian(state: torch.Tensor, rhs_values: torch.Tensor, boundary_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nonlinear = rhs_values - state.pow(3)
    residual = torch.cat([state, boundary_values, nonlinear], dim=0)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = -3.0 * torch.diag(state.square())
    return residual, jacobian


def darcy_residual_operator(state: torch.Tensor, rhs_values: torch.Tensor, boundary_values: torch.Tensor) -> ResidualOperator:
    zeros_boundary = torch.zeros(boundary_values.numel(), dtype=state.dtype, device=state.device)
    coeff = -3.0 * state.square()
    residual = torch.cat([state, boundary_values, rhs_values - state.pow(3)], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        return torch.cat([vector, zeros_boundary, coeff * vector], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        return vector[: state.numel()] + coeff * vector[state.numel() + boundary_values.numel() :]

    diagonal = 1.0 + coeff.square()
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)


def build_navier_stokes_matern_theta(points: torch.Tensor, lengthscale: float, nugget: float = 1e-10) -> KernelAssembly:
    point_indices = torch.arange(points.shape[0], dtype=torch.long, device=points.device)
    uu = matern52_2d(points, points, lengthscale)
    u_dx = matern52_2d_dy1(points, points, lengthscale)
    u_dy = matern52_2d_dy2(points, points, lengthscale)
    u_lap = matern52_2d_laplace_y(points, points, lengthscale)
    dx_u = matern52_2d_dx1(points, points, lengthscale)
    dx_dx = matern52_2d_dx1dy1(points, points, lengthscale)
    dx_dy = matern52_2d_dx1dy2(points, points, lengthscale)
    dx_lap = matern52_2d_laplace_y_dx1(points, points, lengthscale)
    dy_u = matern52_2d_dx2(points, points, lengthscale)
    dy_dx = matern52_2d_dx2dy1(points, points, lengthscale)
    dy_dy = matern52_2d_dx2dy2(points, points, lengthscale)
    dy_lap = matern52_2d_laplace_y_dx2(points, points, lengthscale)
    lap_u = matern52_2d_laplace_x(points, points, lengthscale)
    lap_dx = matern52_2d_laplace_x_dy1(points, points, lengthscale)
    lap_dy = matern52_2d_laplace_x_dy2(points, points, lengthscale)
    lap_lap = matern52_2d_laplace_xy(points, points, lengthscale)
    theta = _assemble_blocks(((uu, u_dx, u_dy, u_lap), (dx_u, dx_dx, dx_dy, dx_lap), (dy_u, dy_dx, dy_dy, dy_lap), (lap_u, lap_dx, lap_dy, lap_lap)))
    block_sizes = (points.shape[0], points.shape[0], points.shape[0], points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(theta=theta, dirac_points=points, derivative_point_groups=(point_indices, point_indices, point_indices), block_sizes=block_sizes)


def build_navier_stokes_empirical_theta(solution_features: torch.Tensor, dx_features: torch.Tensor, dy_features: torch.Tensor, laplacian_features: torch.Tensor, nugget: float = 1e-10) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, dx_features, dy_features, laplacian_features))
    block_sizes = (solution_features.shape[0], dx_features.shape[0], dy_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def navier_stokes_vorticity_residual_and_jacobian(state: torch.Tensor, previous_w: torch.Tensor, previous_wx: torch.Tensor, previous_wy: torch.Tensor, previous_lap: torch.Tensor, velocity_u: torch.Tensor, velocity_v: torch.Tensor, viscosity: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = previous_w.numel()
    w = state[:n_points]
    wx = state[n_points : 2 * n_points]
    wy = state[2 * n_points : 3 * n_points]
    lap = 2.0 / viscosity * ((w - previous_w) / dt + 0.5 * (velocity_u * (wx + previous_wx) + velocity_v * (wy + previous_wy))) - previous_lap
    residual = torch.cat([w, wx, wy, lap], dim=0)
    jacobian = torch.zeros((4 * n_points, 3 * n_points), dtype=state.dtype, device=state.device)
    identity = torch.eye(n_points, dtype=state.dtype, device=state.device)
    jacobian[:n_points, :n_points] = identity
    jacobian[n_points : 2 * n_points, n_points : 2 * n_points] = identity
    jacobian[2 * n_points : 3 * n_points, 2 * n_points : 3 * n_points] = identity
    jacobian[3 * n_points :, :n_points] = (2.0 / viscosity) * (identity / dt)
    jacobian[3 * n_points :, n_points : 2 * n_points] = (1.0 / viscosity) * torch.diag(velocity_u)
    jacobian[3 * n_points :, 2 * n_points : 3 * n_points] = (1.0 / viscosity) * torch.diag(velocity_v)
    return residual, jacobian


def navier_stokes_vorticity_residual_operator(state: torch.Tensor, previous_w: torch.Tensor, previous_wx: torch.Tensor, previous_wy: torch.Tensor, previous_lap: torch.Tensor, velocity_u: torch.Tensor, velocity_v: torch.Tensor, viscosity: float, dt: float) -> ResidualOperator:
    n_points = previous_w.numel()
    w = state[:n_points]
    wx = state[n_points : 2 * n_points]
    wy = state[2 * n_points : 3 * n_points]
    coeff_w = torch.full_like(w, 2.0 / viscosity / dt)
    coeff_wx = (1.0 / viscosity) * velocity_u
    coeff_wy = (1.0 / viscosity) * velocity_v
    lap = 2.0 / viscosity * ((w - previous_w) / dt + 0.5 * (velocity_u * (wx + previous_wx) + velocity_v * (wy + previous_wy))) - previous_lap
    residual = torch.cat([w, wx, wy, lap], dim=0)

    def jvp(vector: torch.Tensor) -> torch.Tensor:
        dw = vector[:n_points]
        dwx = vector[n_points : 2 * n_points]
        dwy = vector[2 * n_points :]
        return torch.cat([dw, dwx, dwy, coeff_w * dw + coeff_wx * dwx + coeff_wy * dwy], dim=0)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        weight_w = vector[:n_points]
        weight_wx = vector[n_points : 2 * n_points]
        weight_wy = vector[2 * n_points : 3 * n_points]
        weight_lap = vector[3 * n_points :]
        return torch.cat([
            weight_w + coeff_w * weight_lap,
            weight_wx + coeff_wx * weight_lap,
            weight_wy + coeff_wy * weight_lap,
        ], dim=0)

    diagonal = torch.cat([1.0 + coeff_w.square(), 1.0 + coeff_wx.square(), 1.0 + coeff_wy.square()], dim=0)
    return ResidualOperator(residual=residual, jvp=jvp, vjp=vjp, preconditioner_diag=2.0 * diagonal)
