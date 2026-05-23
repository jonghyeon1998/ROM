from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


ResidualJacobianFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class GaussNewtonResult:
    state: torch.Tensor
    losses: list[float]
    gradient_norms: list[float]


def weighted_norm_squared(residual: torch.Tensor, factor: object) -> torch.Tensor:
    weighted = factor.apply(residual)
    return torch.dot(weighted, weighted)


def gauss_newton_matrices(
    state: torch.Tensor,
    residual_and_jacobian: ResidualJacobianFn,
    factor: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, jacobian = residual_and_jacobian(state)
    weighted_residual = factor.apply(residual)
    weighted_jacobian = factor.apply_jacobian(jacobian)
    loss = torch.dot(weighted_residual, weighted_residual)
    gradient = 2.0 * weighted_jacobian.transpose(0, 1).matmul(weighted_residual)
    hessian = 2.0 * weighted_jacobian.transpose(0, 1).matmul(weighted_jacobian)
    return loss, residual, gradient, hessian


def solve_gauss_newton(
    initial_state: torch.Tensor,
    residual_and_jacobian: ResidualJacobianFn,
    factor: object,
    max_iter: int,
    step_size: float = 1.0,
    damping: float = 1e-10,
    record_history: bool = True,
) -> GaussNewtonResult:
    state = initial_state.clone()
    losses: list[float] = []
    gradient_norms: list[float] = []

    for _ in range(max_iter):
        loss, _, gradient, hessian = gauss_newton_matrices(state, residual_and_jacobian, factor)
        if record_history:
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(torch.linalg.norm(gradient).detach().cpu()))
        eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
        delta = torch.linalg.solve(hessian + damping * eye, gradient)
        state = state - step_size * delta

    return GaussNewtonResult(state=state, losses=losses, gradient_norms=gradient_norms)
