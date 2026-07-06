from __future__ import annotations

import math

import torch


SQRT5 = math.sqrt(5.0)
SQRT7 = math.sqrt(7.0)


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


# ---------------------------------------------------------------------------
# Matern 7/2. Closed forms below were derived symbolically (sympy) and each
# was cross-checked against centered finite differences of the base kernel
# before transcription; see the derivation notes in the PR/commit introducing
# this block. Mirrors the structure of the Matern 5/2 section above exactly
# (same _terms helper pattern, same derivative_x/derivative_y dispatch for the
# 1D case, same dx1/dy1/dx2/dy2/laplace naming for the 2D case) so the two
# families are interchangeable at call sites.
# ---------------------------------------------------------------------------


def _matern72_1d_terms(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _as_column(x)
    y = _as_column(y)
    diff = x[:, None, 0] - y[None, :, 0]
    radius = diff.abs()
    ell = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device)
    exp_term = torch.exp(-SQRT7 * radius / ell)
    return diff, radius, exp_term


def matern72_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float,
    derivative_x: int = 0,
    derivative_y: int = 0,
) -> torch.Tensor:
    diff, radius, exp_term = _matern72_1d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=diff.dtype, device=diff.device)

    if derivative_x == 0 and derivative_y == 0:
        return (1.0 + SQRT7 * radius / ell + 14.0 * radius.square() / (5.0 * ell.square()) + 7.0 * SQRT7 * radius.pow(3) / (15.0 * ell.pow(3))) * exp_term
    if derivative_x == 1 and derivative_y == 0:
        return -7.0 * diff * exp_term * (3.0 * ell.square() + 3.0 * SQRT7 * ell * radius + 7.0 * radius.square()) / (15.0 * ell.pow(4))
    if derivative_x == 0 and derivative_y == 1:
        return 7.0 * diff * exp_term * (3.0 * ell.square() + 3.0 * SQRT7 * ell * radius + 7.0 * radius.square()) / (15.0 * ell.pow(4))
    if derivative_x == 1 and derivative_y == 1:
        return 7.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius - 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(5))
    if derivative_x == 2 and derivative_y == 0:
        return -7.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius - 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(5))
    if derivative_x == 0 and derivative_y == 2:
        return -7.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius - 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(5))
    if derivative_x == 2 and derivative_y == 1:
        return 7.0 * diff * exp_term * (49.0 * radius.square() - 21.0 * ell.square() - 21.0 * SQRT7 * ell * radius) / (15.0 * ell.pow(6))
    if derivative_x == 1 and derivative_y == 2:
        return -7.0 * diff * exp_term * (49.0 * radius.square() - 21.0 * ell.square() - 21.0 * SQRT7 * ell * radius) / (15.0 * ell.pow(6))
    if derivative_x == 2 and derivative_y == 2:
        return 49.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius - 42.0 * ell * radius.square() + 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(7))
    raise NotImplementedError(f"Unsupported 1D Matérn-7/2 derivative order ({derivative_x}, {derivative_y}).")


def _matern72_2d_terms(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _as_column(x)
    y = _as_column(y)
    dx1 = x[:, None, 0] - y[None, :, 0]
    dx2 = x[:, None, 1] - y[None, :, 1]
    radius = torch.sqrt(dx1.square() + dx2.square())
    ell = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device)
    exp_term = torch.exp(-SQRT7 * radius / ell)
    return dx1, dx2, radius, exp_term


def matern72_2d(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return (1.0 + SQRT7 * radius / ell + 14.0 * (dx1.square() + dx2.square()) / (5.0 * ell.square()) + 7.0 * SQRT7 * radius.pow(3) / (15.0 * ell.pow(3))) * exp_term


def matern72_2d_dx1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, _, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -7.0 * dx1 * exp_term * (3.0 * ell.square() + 3.0 * SQRT7 * ell * radius + 7.0 * radius.square()) / (15.0 * ell.pow(4))


def matern72_2d_dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern72_2d_dx1(x, y, lengthscale)


def matern72_2d_dx2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return -7.0 * dx2 * exp_term * (3.0 * ell.square() + 3.0 * SQRT7 * ell * radius + 7.0 * radius.square()) / (15.0 * ell.pow(4))


def matern72_2d_dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern72_2d_dx2(x, y, lengthscale)


def matern72_2d_dx1dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return 7.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius + 7.0 * dx2.square() * ell - 7.0 * SQRT7 * dx1.square() * radius) / (15.0 * ell.pow(5))


def matern72_2d_dx1dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -49.0 * dx1 * dx2 * exp_term * (SQRT7 * radius + ell) / (15.0 * ell.pow(5))


def matern72_2d_dx2dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return matern72_2d_dx1dy2(x, y, lengthscale)


def matern72_2d_dx2dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return 7.0 * exp_term * (3.0 * ell.pow(3) + 3.0 * SQRT7 * ell.square() * radius + 7.0 * dx1.square() * ell - 7.0 * SQRT7 * dx2.square() * radius) / (15.0 * ell.pow(5))


def matern72_2d_laplace_x(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, _, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=radius.dtype, device=radius.device)
    return 7.0 * exp_term * (-6.0 * ell.pow(3) - 6.0 * SQRT7 * ell.square() * radius - 7.0 * ell * radius.square() + 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(5))


def matern72_2d_laplace_y(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return matern72_2d_laplace_x(x, y, lengthscale)


def matern72_2d_laplace_x_dy1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    dx1, _, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx1.dtype, device=dx1.device)
    return -49.0 * dx1 * exp_term * (4.0 * ell.square() + 4.0 * SQRT7 * ell * radius - 7.0 * radius.square()) / (15.0 * ell.pow(6))


def matern72_2d_laplace_x_dy2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, dx2, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=dx2.dtype, device=dx2.device)
    return -49.0 * dx2 * exp_term * (4.0 * ell.square() + 4.0 * SQRT7 * ell * radius - 7.0 * radius.square()) / (15.0 * ell.pow(6))


def matern72_2d_laplace_y_dx1(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern72_2d_laplace_x_dy1(x, y, lengthscale)


def matern72_2d_laplace_y_dx2(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return -matern72_2d_laplace_x_dy2(x, y, lengthscale)


def matern72_2d_laplace_xy(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    _, _, radius, exp_term = _matern72_2d_terms(x, y, lengthscale)
    ell = torch.as_tensor(lengthscale, dtype=radius.dtype, device=radius.device)
    return 49.0 * exp_term * (8.0 * ell.pow(3) + 8.0 * SQRT7 * ell.square() * radius - 56.0 * ell * radius.square() + 7.0 * SQRT7 * radius.pow(3)) / (15.0 * ell.pow(7))


# Lightweight registry so call sites can select the kernel "order" (smoothness)
# as a single string/parameter instead of importing a different function name
# per order. Both the 5/2 and 7/2 families are exposed; new orders just need a
# new entry here once their closed forms exist.
MATERN_2D_KERNELS: dict[str, dict[str, object]] = {
    "5/2": {
        "value": matern52_2d,
        "dx1": matern52_2d_dx1,
        "dy1": matern52_2d_dy1,
        "dx2": matern52_2d_dx2,
        "dy2": matern52_2d_dy2,
        "dx1dy1": matern52_2d_dx1dy1,
        "dx1dy2": matern52_2d_dx1dy2,
        "dx2dy1": matern52_2d_dx2dy1,
        "dx2dy2": matern52_2d_dx2dy2,
        "laplace_x": matern52_2d_laplace_x,
        "laplace_y": matern52_2d_laplace_y,
        "laplace_x_dy1": matern52_2d_laplace_x_dy1,
        "laplace_x_dy2": matern52_2d_laplace_x_dy2,
        "laplace_y_dx1": matern52_2d_laplace_y_dx1,
        "laplace_y_dx2": matern52_2d_laplace_y_dx2,
        "laplace_xy": matern52_2d_laplace_xy,
    },
    "7/2": {
        "value": matern72_2d,
        "dx1": matern72_2d_dx1,
        "dy1": matern72_2d_dy1,
        "dx2": matern72_2d_dx2,
        "dy2": matern72_2d_dy2,
        "dx1dy1": matern72_2d_dx1dy1,
        "dx1dy2": matern72_2d_dx1dy2,
        "dx2dy1": matern72_2d_dx2dy1,
        "dx2dy2": matern72_2d_dx2dy2,
        "laplace_x": matern72_2d_laplace_x,
        "laplace_y": matern72_2d_laplace_y,
        "laplace_x_dy1": matern72_2d_laplace_x_dy1,
        "laplace_x_dy2": matern72_2d_laplace_x_dy2,
        "laplace_y_dx1": matern72_2d_laplace_y_dx1,
        "laplace_y_dx2": matern72_2d_laplace_y_dx2,
        "laplace_xy": matern72_2d_laplace_xy,
    },
}
