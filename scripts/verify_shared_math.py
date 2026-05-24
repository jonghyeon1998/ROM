from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.gauss_newton import gauss_newton_matrices
from src.krom.kernels import (
    matern52_1d,
    matern52_2d,
    matern52_2d_laplace_x,
    matern52_2d_laplace_x_dy1,
    matern52_2d_laplace_x_dy2,
    matern52_2d_laplace_xy,
    matern52_2d_laplace_y_dx1,
    matern52_2d_laplace_y_dx2,
)
from src.krom.navier_stokes import streamfunction_poisson_residual_and_jacobian
from src.krom.pde_baselines import AllenCahnCNConfig, BurgersCNConfig
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.workflows import (
    allen_cahn_residual_and_jacobian,
    burgers_residual_and_jacobian,
    elliptic_residual_and_jacobian,
    navier_stokes_vorticity_residual_and_jacobian,
)


torch.set_default_dtype(torch.float64)


def check_close(name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float = 1e-9) -> None:
    error = float(torch.max(torch.abs(lhs.detach() - rhs.detach())))
    print(f'{name}: max error = {error:.3e}')
    if error > atol:
        raise AssertionError(f'{name} failed with error {error:.3e}')


def verify_kernels() -> None:
    ell = 0.37

    x = torch.tensor(0.21, requires_grad=True)
    y = torch.tensor(0.74, requires_grad=True)
    r = torch.abs(x - y)
    scalar_1d = (1 + math.sqrt(5) * r / ell + 5 * r * r / (3 * ell * ell)) * torch.exp(-math.sqrt(5) * r / ell)
    dx = torch.autograd.grad(scalar_1d, x, create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, create_graph=True)[0]
    dxy = torch.autograd.grad(dx, y)[0]
    x_mat = x.detach().view(1, 1)
    y_mat = y.detach().view(1, 1)
    check_close('1D value', scalar_1d.detach(), matern52_1d(x_mat, y_mat, ell, 0, 0)[0, 0])
    check_close('1D dxx', dxx.detach(), matern52_1d(x_mat, y_mat, ell, 2, 0)[0, 0])
    check_close('1D dxy', dxy.detach(), matern52_1d(x_mat, y_mat, ell, 1, 1)[0, 0])

    x1 = torch.tensor(0.23, requires_grad=True)
    x2 = torch.tensor(0.61, requires_grad=True)
    y1 = torch.tensor(0.77, requires_grad=True)
    y2 = torch.tensor(0.12, requires_grad=True)
    r2 = torch.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
    scalar_2d = (1 + math.sqrt(5) * r2 / ell + 5 * r2 * r2 / (3 * ell * ell)) * torch.exp(-math.sqrt(5) * r2 / ell)
    dx1_val = torch.autograd.grad(scalar_2d, x1, create_graph=True)[0]
    dx2_val = torch.autograd.grad(scalar_2d, x2, create_graph=True)[0]
    laplace_x = torch.autograd.grad(dx1_val, x1, create_graph=True)[0] + torch.autograd.grad(dx2_val, x2, create_graph=True)[0]
    laplace_x_dy1 = torch.autograd.grad(laplace_x, y1, create_graph=True)[0]
    laplace_x_dy2 = torch.autograd.grad(laplace_x, y2, create_graph=True)[0]
    laplace_y_dx1 = torch.autograd.grad(
        torch.autograd.grad(torch.autograd.grad(scalar_2d, y1, create_graph=True)[0], y1, create_graph=True)[0]
        + torch.autograd.grad(torch.autograd.grad(scalar_2d, y2, create_graph=True)[0], y2, create_graph=True)[0],
        x1,
        create_graph=True,
    )[0]
    laplace_y_dx2 = torch.autograd.grad(
        torch.autograd.grad(torch.autograd.grad(scalar_2d, y1, create_graph=True)[0], y1, create_graph=True)[0]
        + torch.autograd.grad(torch.autograd.grad(scalar_2d, y2, create_graph=True)[0], y2, create_graph=True)[0],
        x2,
        create_graph=True,
    )[0]
    laplace_xy = torch.autograd.grad(laplace_x_dy1, y1, create_graph=True)[0]
    laplace_xy = laplace_xy + torch.autograd.grad(laplace_x_dy2, y2)[0]
    x_mat = torch.tensor([[x1.detach(), x2.detach()]])
    y_mat = torch.tensor([[y1.detach(), y2.detach()]])
    check_close('2D value', scalar_2d.detach(), matern52_2d(x_mat, y_mat, ell)[0, 0])
    check_close('2D Laplace-x', laplace_x.detach(), matern52_2d_laplace_x(x_mat, y_mat, ell)[0, 0], atol=1e-8)
    check_close('2D Laplace-x-dy1', laplace_x_dy1.detach(), matern52_2d_laplace_x_dy1(x_mat, y_mat, ell)[0, 0], atol=1e-8)
    check_close('2D Laplace-x-dy2', laplace_x_dy2.detach(), matern52_2d_laplace_x_dy2(x_mat, y_mat, ell)[0, 0], atol=1e-8)
    check_close('2D Laplace-y-dx1', laplace_y_dx1.detach(), matern52_2d_laplace_y_dx1(x_mat, y_mat, ell)[0, 0], atol=1e-8)
    check_close('2D Laplace-y-dx2', laplace_y_dx2.detach(), matern52_2d_laplace_y_dx2(x_mat, y_mat, ell)[0, 0], atol=1e-8)
    check_close('2D Laplace-xy', laplace_xy.detach(), matern52_2d_laplace_xy(x_mat, y_mat, ell)[0, 0], atol=1e-8)


def verify_sparse_factor() -> None:
    theta = torch.eye(8, dtype=torch.float64)
    dirac_points = torch.rand(5, 2)
    derivative_groups = (torch.arange(3, dtype=torch.long),)
    factor, ordering = sparse_precision_factor(theta, dirac_points, derivative_groups, rho=1.0)
    probe = torch.randn(theta.shape[0], dtype=torch.float64)
    transformed = factor.apply(probe)
    permuted = probe.index_select(0, ordering.permutation)
    check_close('Sparse factor identity action', permuted, transformed)


def verify_residual_jacobians() -> None:
    bconfig = BurgersCNConfig()
    prev_u = torch.randn(4)
    prev_ux = torch.randn(4)
    prev_uxx = torch.randn(4)
    bdy = torch.zeros(2)
    state = torch.randn(8, requires_grad=True)

    def burgers_residual_only(v: torch.Tensor) -> torch.Tensor:
        return burgers_residual_and_jacobian(v, prev_u, prev_ux, prev_uxx, bdy, bconfig.viscosity, bconfig.dt)[0]

    _, analytic_jacobian = burgers_residual_and_jacobian(state, prev_u, prev_ux, prev_uxx, bdy, bconfig.viscosity, bconfig.dt)
    autodiff_jacobian = torch.autograd.functional.jacobian(burgers_residual_only, state)
    check_close('Burgers residual Jacobian', analytic_jacobian, autodiff_jacobian)

    aconfig = AllenCahnCNConfig()
    prev_u = torch.randn(5)
    prev_lap = torch.randn(5)
    bdy = torch.zeros(4)
    state = torch.randn(5, requires_grad=True)

    def allen_residual_only(v: torch.Tensor) -> torch.Tensor:
        return allen_cahn_residual_and_jacobian(v, prev_u, prev_lap, bdy, aconfig.epsilon, aconfig.dt)[0]

    _, analytic_jacobian = allen_cahn_residual_and_jacobian(state, prev_u, prev_lap, bdy, aconfig.epsilon, aconfig.dt)
    autodiff_jacobian = torch.autograd.functional.jacobian(allen_residual_only, state)
    check_close('Allen-Cahn residual Jacobian', analytic_jacobian, autodiff_jacobian, atol=1e-8)

    rhs = torch.randn(6)
    boundary = torch.zeros(4)
    elliptic_state = torch.randn(6, requires_grad=True)
    elliptic_residual, elliptic_jacobian = elliptic_residual_and_jacobian(elliptic_state, rhs, boundary, alpha=1.0, power=3)

    def elliptic_residual_only(v: torch.Tensor) -> torch.Tensor:
        return elliptic_residual_and_jacobian(v, rhs, boundary, alpha=1.0, power=3)[0]

    autodiff_jacobian = torch.autograd.functional.jacobian(elliptic_residual_only, elliptic_state)
    check_close('Elliptic residual Jacobian', elliptic_jacobian, autodiff_jacobian)

    previous_w = torch.randn(3)
    previous_wx = torch.randn(3)
    previous_wy = torch.randn(3)
    previous_lap = torch.randn(3)
    vel_u = torch.randn(3)
    vel_v = torch.randn(3)
    state = torch.randn(9, requires_grad=True)

    def ns_residual_only(v: torch.Tensor) -> torch.Tensor:
        return navier_stokes_vorticity_residual_and_jacobian(
            v,
            previous_w,
            previous_wx,
            previous_wy,
            previous_lap,
            vel_u,
            vel_v,
            1e-3,
            1e-2,
        )[0]

    _, analytic_jacobian = navier_stokes_vorticity_residual_and_jacobian(
        state,
        previous_w,
        previous_wx,
        previous_wy,
        previous_lap,
        vel_u,
        vel_v,
        1e-3,
        1e-2,
    )
    autodiff_jacobian = torch.autograd.functional.jacobian(ns_residual_only, state)
    check_close('Navier-Stokes residual Jacobian', analytic_jacobian, autodiff_jacobian)

    rhs_laplace = torch.randn(4)
    state = torch.randn(12, requires_grad=True)

    def poisson_residual_only(v: torch.Tensor) -> torch.Tensor:
        return streamfunction_poisson_residual_and_jacobian(v, rhs_laplace)[0]

    _, poisson_jacobian = streamfunction_poisson_residual_and_jacobian(state, rhs_laplace)
    autodiff_jacobian = torch.autograd.functional.jacobian(poisson_residual_only, state)
    check_close('Streamfunction Poisson Jacobian', poisson_jacobian, autodiff_jacobian)

    class IdentityFactor:
        @staticmethod
        def apply(v: torch.Tensor) -> torch.Tensor:
            return v

        @staticmethod
        def apply_transpose(v: torch.Tensor) -> torch.Tensor:
            return v

        @staticmethod
        def apply_jacobian(j: torch.Tensor) -> torch.Tensor:
            return j

    loss, residual, gradient, hessian = gauss_newton_matrices(
        elliptic_state.detach(),
        lambda z: elliptic_residual_and_jacobian(z, rhs, boundary, alpha=1.0, power=3),
        IdentityFactor(),
    )
    expected_gradient = 2.0 * elliptic_jacobian.transpose(0, 1).matmul(elliptic_residual)
    expected_hessian = 2.0 * elliptic_jacobian.transpose(0, 1).matmul(elliptic_jacobian)
    check_close('Gauss-Newton gradient', gradient, expected_gradient)
    check_close('Gauss-Newton Hessian', hessian, expected_hessian)
    print(f'Elliptic identity-factor loss = {float(loss):.3e}, residual norm = {float(torch.linalg.norm(residual)):.3e}')


if __name__ == '__main__':
    verify_kernels()
    verify_sparse_factor()
    verify_residual_jacobians()
    print('All shared math checks passed.')
