# %%
import sys
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = "/Users/jonghyeonlee/KROM"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.krom.benchmark import benchmark_callable
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.pde_baselines import (
    AllenCahnCNConfig,
    build_allen_cahn_flattened_coordinates,
    generate_allen_cahn_initial_condition,
    solve_allen_cahn_crank_nicolson,
)
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.workflows import (
    allen_cahn_residual_and_jacobian,
    build_allen_cahn_empirical_theta,
    build_elliptic_matern_theta,
)
from src.krom.empirical import temporal_feature_matrix

torch.set_default_dtype(torch.float64)

config = AllenCahnCNConfig(nx=41, ny=41, dt=1e-2, tmax=1.0, epsilon=1e-2)
rho = 4.0
lengthscale = 0.30
gn_steps = 2
gn_damping = 1e-8
num_train = 8
num_test = 8

print("Allen-Cahn config:", asdict(config))


# %%
coords = build_allen_cahn_flattened_coordinates(config)
n_int = (config.nx - 2) * (config.ny - 2)
n_bdy = coords.shape[0] - n_int
interior_points = coords[:n_int]
boundary_points = coords[n_int:]
boundary_values = torch.zeros(n_bdy, dtype=torch.float64)
derivative_groups = (torch.arange(n_int, dtype=torch.long),)

x = torch.linspace(0.0, float(config.domain_length), config.nx)
y = torch.linspace(0.0, float(config.domain_length), config.ny)
X, Y = torch.meshgrid(x, y, indexing="xy")


# %%
def flatten_with_boundary_last(u_grid: torch.Tensor) -> torch.Tensor:
    interior = u_grid[1:-1, 1:-1].reshape(-1)
    top = u_grid[0, :]
    bottom = u_grid[-1, :]
    left = u_grid[1:-1, 0]
    right = u_grid[1:-1, -1]
    return torch.cat([interior, top, bottom, left, right], dim=0)


def generate_dataset(num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_steps = int(round(config.tmax / config.dt)) + 1
    flattened_solutions = torch.zeros((num_samples, coords.shape[0], n_steps), dtype=torch.float64)
    flattened_laplacians = torch.zeros_like(flattened_solutions)
    initial_conditions = []

    for sample in range(num_samples):
        u0 = generate_allen_cahn_initial_condition(X, Y, num_modes=4)
        sol, lap = solve_allen_cahn_crank_nicolson(u0, config)
        initial_conditions.append(u0)
        for step in range(n_steps):
            flattened_solutions[sample, :, step] = flatten_with_boundary_last(sol[step])
            flattened_laplacians[sample, :, step] = flatten_with_boundary_last(lap[step])

    return flattened_solutions, flattened_laplacians


train_sol, train_lap = generate_dataset(num_train)
test_sol, test_lap = generate_dataset(num_test)


# %%
# Empirical sparse factor
dirac_features = temporal_feature_matrix(train_sol)
lap_features = temporal_feature_matrix(train_lap[:, :n_int, :])
theta_empirical = build_allen_cahn_empirical_theta(dirac_features, lap_features, nugget=1e-9) / dirac_features.shape[1]

empirical_factor, _ = sparse_precision_factor(
    theta=theta_empirical,
    dirac_points=coords,
    derivative_point_groups=derivative_groups,
    rho=rho,
    nugget=1e-9,
)

print("Empirical factor nnz:", empirical_factor.nnz)


# %%
# Matérn sparse factor
matern_assembly = build_elliptic_matern_theta(interior_points, boundary_points, lengthscale=lengthscale, nugget=1e-9)
matern_factor, _ = sparse_precision_factor(
    theta=matern_assembly.theta,
    dirac_points=matern_assembly.dirac_points,
    derivative_point_groups=matern_assembly.derivative_point_groups,
    rho=rho,
    nugget=1e-9,
)

print("Matern factor nnz:", matern_factor.nnz)


# %%
def rollout_allen_cahn_krom(initial_u: torch.Tensor, initial_lap: torch.Tensor, factor) -> torch.Tensor:
    n_steps = int(round(config.tmax / config.dt)) + 1
    rollout = torch.zeros((n_int, n_steps), dtype=torch.float64)
    previous_u = initial_u[:n_int].clone()
    previous_lap = initial_lap[:n_int].clone()
    rollout[:, 0] = previous_u

    state = previous_u.clone()
    for step in range(n_steps - 1):
        saved_u = previous_u.clone()
        saved_lap = previous_lap.clone()
        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: allen_cahn_residual_and_jacobian(
                z,
                saved_u,
                saved_lap,
                boundary_values,
                config.epsilon,
                config.dt,
            ),
            factor=factor,
            max_iter=gn_steps,
            damping=gn_damping,
        )
        state = result.state
        previous_u = state.clone()
        previous_lap = 2.0 / config.epsilon**2 * (
            (previous_u - saved_u) / config.dt
            - 0.5 * ((saved_u - saved_u.pow(3)) + (previous_u - previous_u.pow(3)))
        ) - saved_lap
        rollout[:, step + 1] = previous_u

    return rollout


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))


def evaluate_factor(name: str, factor) -> dict[str, float]:
    final_errors = []
    krom_times = []
    full_order_times = []

    for sample in range(num_test):
        u0 = test_sol[sample, :, 0]
        u0_grid = torch.zeros((config.nx, config.ny), dtype=torch.float64)
        u0_grid[1:-1, 1:-1] = u0[:n_int].reshape(config.nx - 2, config.ny - 2)

        full_order_result = benchmark_callable(solve_allen_cahn_crank_nicolson, u0_grid, config, warmup=0, repeats=1)
        full_order_times.append(full_order_result.seconds)

        start = time.perf_counter()
        rollout = rollout_allen_cahn_krom(test_sol[sample, :, 0], test_lap[sample, :, 0], factor)
        krom_times.append(time.perf_counter() - start)
        final_errors.append(relative_l2(rollout[:, -1], test_sol[sample, :n_int, -1]))

    summary = {
        "mean_final_rel_l2": sum(final_errors) / len(final_errors),
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
empirical_rollout = rollout_allen_cahn_krom(test_sol[sample_index, :, 0], test_lap[sample_index, :, 0], empirical_factor)
matern_rollout = rollout_allen_cahn_krom(test_sol[sample_index, :, 0], test_lap[sample_index, :, 0], matern_factor)
truth = test_sol[sample_index, :n_int, -1].reshape(config.nx - 2, config.ny - 2)

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(truth, origin="lower")
plt.title("Full-order CN")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(empirical_rollout[:, -1].reshape(config.nx - 2, config.ny - 2), origin="lower")
plt.title("Empirical KROM")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(matern_rollout[:, -1].reshape(config.nx - 2, config.ny - 2), origin="lower")
plt.title("Matern KROM")
plt.colorbar()
plt.tight_layout()
