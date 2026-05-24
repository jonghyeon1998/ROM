from __future__ import annotations

import math

import torch


SQRT5 = math.sqrt(5.0)


def _as_column(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 1:
        return points.unsqueeze(-1)
    return points


def _matern52_1d_terms(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _as_column(x)
    y = _as_column(y)
    diff = x[:, None, 0] - y[None, :, 0]
    radius = diff.abs()
    ell = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device)
    exp_term = torch.exp(-SQRT5 * radius / ell)
    return diff, radius, exp_term


def matern52_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float,
    derivative_x: int = 0,
    derivative_y: int = 0,
) -> torch.Tensor:
    diff, radius, exp_term = _matern52_1d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=diff.dtype, device=diff.device)

    if derivative_x == 0 and derivative_y == 0:
        return (1.0 + SQRT5 * radius / ell + 5.0 * radius.square() / (3.0 * ell.square())) * exp_term
    if derivative_x == 1 and derivative_y == 0:
        return -5.0 * diff * exp_term * (ell + SQRT5 * radius) / (3.0 * ell.pow(3))
    if derivative_x == 0 and derivative_y == 1:
        return 5.0 * diff * exp_term * (ell + SQRT5 * radius) / (3.0 * ell.pow(3))
    if derivative_x == 1 and derivative_y == 1:
        return 5.0 * exp_term * (SQRT5 * ell * radius + ell.square() - 5.0 * diff.square()) / (3.0 * ell.pow(4))
    if derivative_x == 2 and derivative_y == 0:
        return -5.0 * exp_term * (SQRT5 * ell * radius + ell.square() - 5.0 * diff.square()) / (3.0 * ell.pow(4))
    if derivative_x == 0 and derivative_y == 2:
        return -5.0 * exp_term * (SQRT5 * ell * radius + ell.square() - 5.0 * diff.square()) / (3.0 * ell.pow(4))
    if derivative_x == 2 and derivative_y == 1:
        return 25.0 * diff * exp_term * (SQRT5 * radius - 3.0 * ell) / (3.0 * ell.pow(5))
    if derivative_x == 1 and derivative_y == 2:
        return -25.0 * diff * exp_term * (SQRT5 * radius - 3.0 * ell) / (3.0 * ell.pow(5))
    if derivative_x == 2 and derivative_y == 2:
        return -25.0 * exp_term * (5.0 * SQRT5 * ell * radius - (3.0 * ell.square() + 5.0 * diff.square())) / (3.0 * ell.pow(6))
    raise NotImplementedError(f"Unsupported 1D Matérn derivative order ({derivative_x}, {derivative_y}).")


def _matern52_2d_terms(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _as_column(x)
    y = _as_column(y)
    dx1 = x[:, None, 0] - y[None, :, 0]
    dx2 = x[:, None, 1] - y[None, :, 1]
    radius = torch.sqrt(dx1.square() + dx2.square())
    ell = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device)
    exp_term = torch.exp(-SQRT5 * radius / ell)
    return dx1, dx2, radius, exp_term


def matern52_2d(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return (1.0 + SQRT5 * radius / ell + 5.0 * (dx1.square() + dx2.square()) / (3.0 * ell.square())) * exp_term


def matern52_2d_dx1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, _, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -5.0 * dx1 * exp_term * (ell + SQRT5 * radius) / (3.0 * ell.pow(3))


def matern52_2d_dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern52_2d_dx1(x, y, lengthscale)


def matern52_2d_dx2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, dx2, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return -5.0 * dx2 * exp_term * (ell + SQRT5 * radius) / (3.0 * ell.pow(3))


def matern52_2d_dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern52_2d_dx2(x, y, lengthscale)


def matern52_2d_dx1dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, _, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -5.0 * exp_term * (5.0 * dx1.square() / ell - ell - SQRT5 * radius) / (3.0 * ell.pow(3))


def matern52_2d_dx1dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, _, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -25.0 * dx1 * dx2 * exp_term / (3.0 * ell.pow(4))


def matern52_2d_dx2dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return matern52_2d_dx1dy2(x, y, lengthscale)


def matern52_2d_dx2dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, dx2, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return -5.0 * exp_term * (5.0 * dx2.square() / ell - ell - SQRT5 * radius) / (3.0 * ell.pow(3))


def matern52_2d_laplace_x(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -5.0 * exp_term * (2.0 * ell + 2.0 * SQRT5 * radius - 5.0 * (dx1.square() + dx2.square()) / ell) / (3.0 * ell.pow(3))


def matern52_2d_laplace_y(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return matern52_2d_laplace_x(x, y, lengthscale)


def matern52_2d_laplace_x_dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, _, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -25.0 * dx1 * exp_term * (4.0 - SQRT5 * radius / ell) / (3.0 * ell.pow(4))


def matern52_2d_laplace_x_dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, dx2, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return -25.0 * dx2 * exp_term * (4.0 - SQRT5 * radius / ell) / (3.0 * ell.pow(4))


def matern52_2d_laplace_y_dx1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern52_2d_laplace_x_dy1(x, y, lengthscale)


def matern52_2d_laplace_y_dx2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern52_2d_laplace_x_dy2(x, y, lengthscale)


def matern52_2d_laplace_xy(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, _, radius, exp_term = _matern52_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=radius.dtype, device=radius.device)
    return -5.0 * exp_term * (35.0 * SQRT5 * radius / ell.square() - 40.0 / ell - 25.0 * radius.square() / ell.pow(3)) / (3.0 * ell.pow(3))
