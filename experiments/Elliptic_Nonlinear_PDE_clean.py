# %%
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = "/Users/jonghyeonlee/KROM"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.krom.gauss_newton import solve_gauss_newton
from src.krom.pde_baselines import build_allen_cahn_flattened_coordinates, AllenCahnCNConfig
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.workflows import (
    build_elliptic_empirical_theta,
    build_elliptic_matern_theta,
    elliptic_residual_and_jacobian,
)

torch.set_default_dtype(torch.float64)

grid_config = AllenCahnCNConfig(nx=33, ny=33, dt=1.0, tmax=1.0)
rho = 4.0
lengthscale = 0.30
gn_steps = 4
alpha = 1.0
power = 3

coords = build_allen_cahn_flattened_coordinates(grid_config)
n_int = (grid_config.nx - 2) * (grid_config.ny - 2)
interior_points = coords[:n_int]
boundary_points = coords[n_int:]
boundary_values = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)

print("Elliptic grid config:", asdict(grid_config))


# %%
def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)


def rhs_f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    term1 = -torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    term2 = -8.0 * torch.pi**2 * torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    return -(term1 + term2) + alpha * u_exact(x, y).pow(power)


truth = u_exact(interior_points[:, 0], interior_points[:, 1])
rhs_values = rhs_f(interior_points[:, 0], interior_points[:, 1])


# %%
# Simple synthetic snapshot bank for the empirical kernel.
num_snapshots = 256
phases = torch.linspace(0.25, 2.0, num_snapshots)
solution_bank = torch.zeros((coords.shape[0], num_snapshots), dtype=torch.float64)
nonlinear_bank = torch.zeros((n_int, num_snapshots), dtype=torch.float64)

for index, phase in enumerate(phases):
    values = phase * u_exact(coords[:, 0], coords[:, 1])
    solution_bank[:, index] = values
    nonlinear_bank[:, index] = rhs_f(interior_points[:, 0], interior_points[:, 1]) - alpha * values[:n_int].pow(power)

theta_empirical = build_elliptic_empirical_theta(solution_bank, nonlinear_bank, nugget=1e-10) / num_snapshots
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
def solve_with_factor(factor) -> torch.Tensor:
    initial_state = torch.zeros(n_int, dtype=torch.float64)
    result = solve_gauss_newton(
        initial_state=initial_state,
        residual_and_jacobian=lambda z: elliptic_residual_and_jacobian(
            z,
            rhs_values,
            boundary_values,
            alpha=alpha,
            power=power,
        ),
        factor=factor,
        max_iter=gn_steps,
        damping=1e-8,
    )
    return result.state


# %%
# Empirical KROM
empirical_solution = solve_with_factor(empirical_factor)
empirical_error = torch.linalg.norm(empirical_solution - truth) / torch.linalg.norm(truth)
print("Empirical relative L2:", float(empirical_error))


# %%
# Matérn KROM
matern_solution = solve_with_factor(matern_factor)
matern_error = torch.linalg.norm(matern_solution - truth) / torch.linalg.norm(truth)
print("Matern relative L2:", float(matern_error))


# %%
x = interior_points[:, 0].reshape(grid_config.nx - 2, grid_config.ny - 2)
y = interior_points[:, 1].reshape(grid_config.nx - 2, grid_config.ny - 2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.contourf(x, y, truth.reshape(grid_config.nx - 2, grid_config.ny - 2), levels=40)
plt.title("Exact solution")

plt.subplot(1, 3, 2)
plt.contourf(x, y, empirical_solution.reshape(grid_config.nx - 2, grid_config.ny - 2), levels=40)
plt.title("Empirical KROM")

plt.subplot(1, 3, 3)
plt.contourf(x, y, matern_solution.reshape(grid_config.nx - 2, grid_config.ny - 2), levels=40)
plt.title("Matern KROM")
plt.tight_layout()
