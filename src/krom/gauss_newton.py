from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


ResidualJacobianFn = Callable[[torch.Tensor], object]
LinearOperatorFn = Callable[[torch.Tensor], torch.Tensor]
PreconditionerFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ResidualOperator:
    residual: torch.Tensor
    jvp: LinearOperatorFn
    vjp: LinearOperatorFn
    preconditioner_diag: torch.Tensor | None = None


@dataclass
class ConjugateGradientResult:
    solution: torch.Tensor
    iterations: int
    converged: bool
    residual_norms: list[float]


@dataclass
class GaussNewtonLinearization:
    loss: torch.Tensor
    residual: torch.Tensor
    gradient: torch.Tensor
    hessian_matvec: LinearOperatorFn
    diagonal: torch.Tensor | None
    preconditioner: PreconditionerFn | None
    weighted_jacobian: torch.Tensor | None = None


@dataclass
class GaussNewtonResult:
    state: torch.Tensor
    losses: list[float]
    gradient_norms: list[float]
    linear_iterations: list[int]


def weighted_norm_squared(residual: torch.Tensor, factor: object) -> torch.Tensor:
    weighted = factor.apply(residual)
    return torch.dot(weighted, weighted)


def conjugate_gradient(
    matvec: LinearOperatorFn,
    rhs: torch.Tensor,
    initial: torch.Tensor | None = None,
    max_iter: int | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    preconditioner: PreconditionerFn | None = None,
) -> ConjugateGradientResult:
    max_iter = rhs.numel() if max_iter is None else max_iter
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    residual = rhs - matvec(x)
    rhs_norm = float(torch.linalg.norm(rhs).detach().cpu())
    threshold = max(float(atol), float(tol) * rhs_norm)
    residual_norm = float(torch.linalg.norm(residual).detach().cpu())
    residual_norms = [residual_norm]
    if residual_norm <= threshold:
        return ConjugateGradientResult(solution=x, iterations=0, converged=True, residual_norms=residual_norms)

    z = preconditioner(residual) if preconditioner is not None else residual.clone()
    direction = z.clone()
    rz_old = torch.dot(residual, z)
    eps = torch.finfo(rhs.dtype).eps

    for iteration in range(1, max_iter + 1):
        matvec_direction = matvec(direction)
        denominator = torch.dot(direction, matvec_direction)
        if torch.abs(denominator) <= eps:
            return ConjugateGradientResult(solution=x, iterations=iteration - 1, converged=False, residual_norms=residual_norms)
        alpha = rz_old / denominator
        x = x + alpha * direction
        residual = residual - alpha * matvec_direction
        residual_norm = float(torch.linalg.norm(residual).detach().cpu())
        residual_norms.append(residual_norm)
        if residual_norm <= threshold:
            return ConjugateGradientResult(solution=x, iterations=iteration, converged=True, residual_norms=residual_norms)
        z = preconditioner(residual) if preconditioner is not None else residual.clone()
        rz_new = torch.dot(residual, z)
        beta = rz_new / torch.clamp(rz_old, min=eps)
        direction = z + beta * direction
        rz_old = rz_new

    return ConjugateGradientResult(solution=x, iterations=max_iter, converged=False, residual_norms=residual_norms)


def _as_linear_model(state: torch.Tensor, residual_and_jacobian: ResidualJacobianFn) -> object:
    model = residual_and_jacobian(state)
    if isinstance(model, ResidualOperator):
        return model
    if isinstance(model, tuple) and len(model) == 2:
        return model
    raise TypeError('Residual model must return either ResidualOperator or (residual, jacobian).')


def gauss_newton_linearization(
    state: torch.Tensor,
    residual_and_jacobian: ResidualJacobianFn,
    factor: object,
    damping: float = 0.0,
    preconditioner: str | None = 'jacobi',
) -> GaussNewtonLinearization:
    model = _as_linear_model(state, residual_and_jacobian)

    if isinstance(model, ResidualOperator):
        residual = model.residual
        weighted_residual = factor.apply(residual)
        backweighted_residual = factor.apply_transpose(weighted_residual)
        gradient = 2.0 * model.vjp(backweighted_residual)

        def hessian_matvec(vector: torch.Tensor) -> torch.Tensor:
            jv = model.jvp(vector)
            weighted_jv = factor.apply(jv)
            backweighted_jv = factor.apply_transpose(weighted_jv)
            result = 2.0 * model.vjp(backweighted_jv)
            if damping != 0.0:
                result = result + damping * vector
            return result

        diagonal = None if model.preconditioner_diag is None else model.preconditioner_diag.clone()
        if diagonal is not None and damping != 0.0:
            diagonal = diagonal + damping
    else:
        residual, jacobian = model
        weighted_residual = factor.apply(residual)
        backweighted_residual = factor.apply_transpose(weighted_residual)
        gradient = 2.0 * jacobian.transpose(0, 1).matmul(backweighted_residual)

        def hessian_matvec(vector: torch.Tensor) -> torch.Tensor:
            jv = jacobian.matmul(vector)
            weighted_jv = factor.apply(jv)
            backweighted_jv = factor.apply_transpose(weighted_jv)
            result = 2.0 * jacobian.transpose(0, 1).matmul(backweighted_jv)
            if damping != 0.0:
                result = result + damping * vector
            return result

        weighted_jacobian = factor.apply_jacobian(jacobian)
        if preconditioner == 'weighted_jacobi':
            diagonal = 2.0 * weighted_jacobian.square().sum(dim=0)
        else:
            diagonal = 2.0 * jacobian.square().sum(dim=0)
        if damping != 0.0:
            diagonal = diagonal + damping
        loss = torch.dot(weighted_residual, weighted_residual)
        preconditioner_fn = None
        if preconditioner in {'jacobi', 'weighted_jacobi'}:
            inverse_diagonal = torch.clamp(diagonal, min=torch.finfo(diagonal.dtype).eps).reciprocal()
            preconditioner_fn = lambda vector: inverse_diagonal * vector
        return GaussNewtonLinearization(
            loss=loss,
            residual=residual,
            gradient=gradient,
            hessian_matvec=hessian_matvec,
            diagonal=diagonal,
            preconditioner=preconditioner_fn,
            weighted_jacobian=weighted_jacobian,
        )

    loss = torch.dot(weighted_residual, weighted_residual)
    preconditioner_fn = None
    if preconditioner in {'jacobi', 'weighted_jacobi'} and diagonal is not None:
        inverse_diagonal = torch.clamp(diagonal, min=torch.finfo(diagonal.dtype).eps).reciprocal()
        preconditioner_fn = lambda vector: inverse_diagonal * vector
    return GaussNewtonLinearization(
        loss=loss,
        residual=residual,
        gradient=gradient,
        hessian_matvec=hessian_matvec,
        diagonal=diagonal,
        preconditioner=preconditioner_fn,
        weighted_jacobian=None,
    )


def gauss_newton_matrices(
    state: torch.Tensor,
    residual_and_jacobian: ResidualJacobianFn,
    factor: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    linearization = gauss_newton_linearization(state, residual_and_jacobian, factor, damping=0.0, preconditioner=None)
    if linearization.weighted_jacobian is None:
        raise ValueError('gauss_newton_matrices requires an explicit Jacobian model.')
    hessian = 2.0 * linearization.weighted_jacobian.transpose(0, 1).matmul(linearization.weighted_jacobian)
    return linearization.loss, linearization.residual, linearization.gradient, hessian


def solve_gauss_newton(
    initial_state: torch.Tensor,
    residual_and_jacobian: ResidualJacobianFn,
    factor: object,
    max_iter: int,
    step_size: float = 1.0,
    damping: float = 1e-10,
    record_history: bool = True,
    linear_solver: str = 'cg',
    cg_max_iter: int | None = None,
    cg_tol: float = 1e-8,
    cg_atol: float = 0.0,
    preconditioner: str | None = 'jacobi',
    direct_residual_and_jacobian: ResidualJacobianFn | None = None,
    direct_threshold: int | None = 2048,
) -> GaussNewtonResult:
    state = initial_state.clone()
    losses: list[float] = []
    gradient_norms: list[float] = []
    linear_iterations: list[int] = []

    for _ in range(max_iter):
        if linear_solver == 'auto':
            use_direct = (
                direct_residual_and_jacobian is not None
                and direct_threshold is not None
                and state.numel() <= direct_threshold
            )
            active_solver = 'direct' if use_direct else 'cg'
        else:
            active_solver = linear_solver

        active_model = residual_and_jacobian
        if active_solver == 'direct' and direct_residual_and_jacobian is not None:
            active_model = direct_residual_and_jacobian

        linearization = gauss_newton_linearization(
            state,
            active_model,
            factor,
            damping=damping if active_solver == 'cg' else 0.0,
            preconditioner=preconditioner if active_solver == 'cg' else None,
        )
        if record_history:
            losses.append(float(linearization.loss.detach().cpu()))
            gradient_norms.append(float(torch.linalg.norm(linearization.gradient).detach().cpu()))

        if active_solver == 'cg':
            cg_result = conjugate_gradient(
                linearization.hessian_matvec,
                linearization.gradient,
                max_iter=cg_max_iter,
                tol=cg_tol,
                atol=cg_atol,
                preconditioner=linearization.preconditioner,
            )
            delta = cg_result.solution
            linear_iterations.append(cg_result.iterations)
        elif active_solver == 'direct':
            if linearization.weighted_jacobian is None:
                raise ValueError('Direct Gauss-Newton requires an explicit Jacobian model.')
            hessian = 2.0 * linearization.weighted_jacobian.transpose(0, 1).matmul(linearization.weighted_jacobian)
            eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
            delta = torch.linalg.solve(hessian + damping * eye, linearization.gradient)
            linear_iterations.append(hessian.shape[0])
        else:
            raise ValueError(f'Unknown linear solver: {active_solver}')

        state = state - step_size * delta

    return GaussNewtonResult(state=state, losses=losses, gradient_norms=gradient_norms, linear_iterations=linear_iterations)
