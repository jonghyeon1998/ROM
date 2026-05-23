# %%
import math
import sys
import time

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = "/Users/jonghyeonlee/KROM"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.krom.benchmark import benchmark_callable
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.pde_baselines import AllenCahnCNConfig, build_allen_cahn_flattened_coordinates
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.workflows import (
    build_elliptic_empirical_theta,
    build_elliptic_matern_theta,
    darcy_residual_and_jacobian,
)

torch.set_default_dtype(torch.float64)

n = 25
rho = 4.0
lengthscale = 0.30
gn_steps = 3
num_train = 12
num_test = 10
domain = AllenCahnCNConfig(nx=n, ny=n, dt=1.0, tmax=1.0)

coords = build_allen_cahn_flattened_coordinates(domain)
n_int = (n - 2) * (n - 2)
interior_points = coords[:n_int]
boundary_points = coords[n_int:]
boundary_values = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)


# %%
def a_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    term1 = (1.1 + torch.sin(10.0 * torch.pi * x)) / (1.1 + torch.sin(10.0 * torch.pi * y))
    term2 = (1.1 + torch.sin(26.0 * torch.pi * y)) / (1.1 + torch.cos(26.0 * torch.pi * x))
    term3 = (1.1 + torch.cos(34.0 * torch.pi * x)) / (1.1 + torch.sin(34.0 * torch.pi * y))
    term4 = (1.1 + torch.sin(62.0 * torch.pi * y)) / (1.1 + torch.cos(62.0 * torch.pi * x))
    term5 = (1.1 + torch.cos(130.0 * torch.pi * x)) / (1.1 + torch.sin(130.0 * torch.pi * y))
    term6 = torch.sin(4.0 * x.square() * y.square())
    return (term1 + term2 + term3 + term4 + term5 + term6 + 1.0) / 6.0


def sample_smooth_forcing(num_modes: int = 4) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n)
    y = torch.linspace(0.0, 1.0, n)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    field = torch.zeros_like(X)
    coeffs = torch.randn((num_modes, num_modes), dtype=torch.float64)
    for i in range(1, num_modes + 1):
        for j in range(1, num_modes + 1):
            field = field + coeffs[i - 1, j - 1] * torch.sin(i * torch.pi * X) * torch.sin(j * torch.pi * Y)
    scale = torch.clamp(field.abs().max(), min=torch.finfo(field.dtype).eps)
    field = field / scale
    return field[1:-1, 1:-1].reshape(-1)


def build_darcy_matrix() -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n)
    y = torch.linspace(0.0, 1.0, n)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    a_grid = a_coefficient(X, Y)
    h = x[1] - x[0]
    nx_int = n - 2
    matrix = torch.zeros((n_int, n_int), dtype=torch.float64)

    def idx(i: int, j: int) -> int:
        return i * nx_int + j

    for i in range(nx_int):
        for j in range(nx_int):
            ii = i + 1
            jj = j + 1
            a_e = 0.5 * (a_grid[ii, jj] + a_grid[ii + 1, jj])
            a_w = 0.5 * (a_grid[ii, jj] + a_grid[ii - 1, jj])
            a_n = 0.5 * (a_grid[ii, jj] + a_grid[ii, jj + 1])
            a_s = 0.5 * (a_grid[ii, jj] + a_grid[ii, jj - 1])
            center = idx(i, j)
            matrix[center, center] = (a_e + a_w + a_n + a_s) / h**2
            if i < nx_int - 1:
                matrix[center, idx(i + 1, j)] = -a_e / h**2
            if i > 0:
                matrix[center, idx(i - 1, j)] = -a_w / h**2
            if j < nx_int - 1:
                matrix[center, idx(i, j + 1)] = -a_n / h**2
            if j > 0:
                matrix[center, idx(i, j - 1)] = -a_s / h**2
    return matrix


A = build_darcy_matrix()


def solve_full_order(rhs_values: torch.Tensor, max_iter: int = 30) -> torch.Tensor:
    state = torch.zeros(n_int, dtype=torch.float64)
    eye = torch.eye(n_int, dtype=torch.float64)
    for _ in range(max_iter):
        residual = A.matmul(state) + state.pow(3) - rhs_values
        jacobian = A + 3.0 * torch.diag(state.square())
        delta = torch.linalg.solve(jacobian + 1e-10 * eye, residual)
        state = state - delta
        if torch.linalg.norm(delta) <= 1e-10 * max(1.0, float(torch.linalg.norm(state))):
            break
    return state


# %%
train_rhs = torch.stack([sample_smooth_forcing() for _ in range(num_train)], dim=1)
test_rhs = torch.stack([sample_smooth_forcing() for _ in range(num_test)], dim=1)
train_solutions = torch.stack([solve_full_order(train_rhs[:, k]) for k in range(num_train)], dim=1)
test_solutions = torch.stack([solve_full_order(test_rhs[:, k]) for k in range(num_test)], dim=1)

train_solution_bank = torch.cat([train_solutions, torch.zeros(boundary_points.shape[0], num_train)], dim=0)
test_solution_bank = torch.cat([test_solutions, torch.zeros(boundary_points.shape[0], num_test)], dim=0)
train_nonlinear_bank = train_rhs - train_solutions.pow(3)


# %%
# Empirical sparse factor
theta_empirical = build_elliptic_empirical_theta(train_solution_bank, train_nonlinear_bank, nugget=1e-10) / num_train
empirical_factor, _ = sparse_precision_factor(
    theta=theta_empirical,
    dirac_points=coords,
    derivative_point_groups=(torch.arange(n_int, dtype=torch.long),),
    rho=rho,
    nugget=1e-10,
)

print("Empirical factor nnz:", empirical_factor.nnz)


# %%
# Matérn sparse factor
matern_assembly = build_elliptic_matern_theta(interior_points, boundary_points, lengthscale=lengthscale, nugget=1e-10)
matern_factor, _ = sparse_precision_factor(
    theta=matern_assembly.theta,
    dirac_points=matern_assembly.dirac_points,
    derivative_point_groups=matern_assembly.derivative_point_groups,
    rho=rho,
    nugget=1e-10,
)

print("Matern factor nnz:", matern_factor.nnz)


# %%
def solve_krom(rhs_values: torch.Tensor, factor) -> torch.Tensor:
    result = solve_gauss_newton(
        initial_state=torch.zeros(n_int, dtype=torch.float64),
        residual_and_jacobian=lambda z: darcy_residual_and_jacobian(z, rhs_values, boundary_values),
        factor=factor,
        max_iter=gn_steps,
        damping=1e-8,
    )
    return result.state


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))


def evaluate_factor(name: str, factor) -> dict[str, float]:
    errors = []
    krom_times = []
    full_order_times = []
    for sample in range(num_test):
        full_result = benchmark_callable(solve_full_order, test_rhs[:, sample], warmup=0, repeats=1)
        full_order_times.append(full_result.seconds)
        start = time.perf_counter()
        pred = solve_krom(test_rhs[:, sample], factor)
        krom_times.append(time.perf_counter() - start)
        errors.append(relative_l2(pred, test_solutions[:, sample]))
    summary = {
        "mean_rel_l2": sum(errors) / len(errors),
        "mean_krom_seconds": sum(krom_times) / len(krom_times),
        "mean_full_order_seconds": sum(full_order_times) / len(full_order_times),
    }
    print(name, summary)
    return summary


# %%
# Empirical KROM
empirical_summary = evaluate_factor("Empirical", empirical_factor)


# %%
# Matérn KROM
matern_summary = evaluate_factor("Matern", matern_factor)


# %%
sample_index = 0
truth = test_solutions[:, sample_index].reshape(n - 2, n - 2)
empirical_pred = solve_krom(test_rhs[:, sample_index], empirical_factor).reshape(n - 2, n - 2)
matern_pred = solve_krom(test_rhs[:, sample_index], matern_factor).reshape(n - 2, n - 2)

x = interior_points[:, 0].reshape(n - 2, n - 2)
y = interior_points[:, 1].reshape(n - 2, n - 2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.contourf(x, y, truth, levels=40)
plt.title("Full-order Darcy")

plt.subplot(1, 3, 2)
plt.contourf(x, y, empirical_pred, levels=40)
plt.title("Empirical KROM")

plt.subplot(1, 3, 3)
plt.contourf(x, y, matern_pred, levels=40)
plt.title("Matern KROM")
plt.tight_layout()
