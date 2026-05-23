from __future__ import annotations

from dataclasses import dataclass

import torch

from .empirical import mixed_empirical_kernel
from .kernels import (
    matern52_1d,
    matern52_2d,
    matern52_2d_laplace_x,
    matern52_2d_laplace_xy,
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


def build_burgers_matern_theta(
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    lengthscale: float,
    nugget: float = 1e-10,
) -> KernelAssembly:
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

    theta = torch.cat(
        [
            torch.cat([uu, u_ux, u_uxx], dim=1),
            torch.cat([ux_u, ux_ux, ux_uxx], dim=1),
            torch.cat([uxx_u, uxx_ux, uxx_uxx], dim=1),
        ],
        dim=0,
    )
    block_sizes = (dirac_points.shape[0], interior_points.shape[0], interior_points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(
        theta=theta,
        dirac_points=dirac_points,
        derivative_point_groups=(interior_indices, interior_indices),
        block_sizes=block_sizes,
    )


def build_burgers_empirical_theta(
    solution_features: torch.Tensor,
    gradient_features: torch.Tensor,
    laplacian_features: torch.Tensor,
    nugget: float = 1e-10,
) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, gradient_features, laplacian_features))
    block_sizes = (solution_features.shape[0], gradient_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def burgers_residual_and_jacobian(
    state: torch.Tensor,
    previous_u: torch.Tensor,
    previous_ux: torch.Tensor,
    previous_uxx: torch.Tensor,
    boundary_values: torch.Tensor,
    viscosity: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def build_allen_cahn_empirical_theta(
    solution_features: torch.Tensor,
    laplacian_features: torch.Tensor,
    nugget: float = 1e-10,
) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, laplacian_features))
    block_sizes = (solution_features.shape[0], laplacian_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def allen_cahn_residual_and_jacobian(
    state: torch.Tensor,
    previous_u: torch.Tensor,
    previous_laplace: torch.Tensor,
    boundary_values: torch.Tensor,
    epsilon: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    laplace = 2.0 / epsilon**2 * (
        (state - previous_u) / dt - 0.5 * ((previous_u - previous_u.pow(3)) + (state - state.pow(3)))
    ) - previous_laplace

    residual = torch.cat([state, boundary_values, laplace], dim=0)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = (2.0 / epsilon**2) * (
        identity / dt - 0.5 * (identity - 3.0 * torch.diag(state.square()))
    )
    return residual, jacobian


def build_elliptic_matern_theta(
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    lengthscale: float,
    nugget: float = 1e-10,
) -> KernelAssembly:
    dirac_points = torch.cat([interior_points, boundary_points], dim=0)
    interior_indices = torch.arange(interior_points.shape[0], dtype=torch.long, device=dirac_points.device)
    uu = matern52_2d(dirac_points, dirac_points, lengthscale)
    lap_u = -matern52_2d_laplace_x(interior_points, dirac_points, lengthscale)
    lap_lap = matern52_2d_laplace_xy(interior_points, interior_points, lengthscale)
    theta = torch.cat(
        [
            torch.cat([uu, lap_u.transpose(0, 1)], dim=1),
            torch.cat([lap_u, lap_lap], dim=1),
        ],
        dim=0,
    )
    block_sizes = (dirac_points.shape[0], interior_points.shape[0])
    theta = _block_nugget(theta, block_sizes, nugget)
    return KernelAssembly(
        theta=theta,
        dirac_points=dirac_points,
        derivative_point_groups=(interior_indices,),
        block_sizes=block_sizes,
    )


def build_elliptic_empirical_theta(
    solution_features: torch.Tensor,
    nonlinear_rhs_features: torch.Tensor,
    nugget: float = 1e-10,
) -> torch.Tensor:
    theta = mixed_empirical_kernel((solution_features, nonlinear_rhs_features))
    block_sizes = (solution_features.shape[0], nonlinear_rhs_features.shape[0])
    return _block_nugget(theta, block_sizes, nugget)


def elliptic_residual_and_jacobian(
    state: torch.Tensor,
    rhs_values: torch.Tensor,
    boundary_values: torch.Tensor,
    alpha: float,
    power: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonlinear = rhs_values - alpha * state.pow(power)
    residual = torch.cat([state, boundary_values, nonlinear], dim=0)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = -alpha * power * torch.diag(state.pow(power - 1))
    return residual, jacobian


def darcy_residual_and_jacobian(
    state: torch.Tensor,
    rhs_values: torch.Tensor,
    boundary_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonlinear = rhs_values - state.pow(3)
    residual = torch.cat([state, boundary_values, nonlinear], dim=0)
    jacobian = torch.zeros((state.numel() + boundary_values.numel() + state.numel(), state.numel()), dtype=state.dtype, device=state.device)
    identity = torch.eye(state.numel(), dtype=state.dtype, device=state.device)
    jacobian[: state.numel(), :] = identity
    jacobian[state.numel() + boundary_values.numel() :, :] = -3.0 * torch.diag(state.square())
    return residual, jacobian
