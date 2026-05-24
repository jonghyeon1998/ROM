# %%
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.benchmark import benchmark_callable
from src.krom.empirical import temporal_feature_matrix
from src.krom.factors import dense_precision_factor
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import AllenCahnCNConfig, build_allen_cahn_flattened_coordinates, generate_allen_cahn_initial_condition, solve_allen_cahn_crank_nicolson
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.sweeps import plot_sweep_result, run_comparison_sweep
from src.krom.workflows import allen_cahn_residual_and_jacobian, allen_cahn_residual_operator, build_allen_cahn_empirical_theta, build_elliptic_matern_theta


torch.set_default_dtype(torch.float64)

DEFAULT_CONFIG = AllenCahnCNConfig(nx=41, ny=41, dt=1e-2, tmax=1.0, epsilon=1e-2)
AUTO_DIRECT_THRESHOLD = 2048


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def flatten_with_boundary_last(u_grid: torch.Tensor) -> torch.Tensor:
    interior = u_grid[1:-1, 1:-1].reshape(-1)
    top = u_grid[0, :]
    bottom = u_grid[-1, :]
    left = u_grid[1:-1, 0]
    right = u_grid[1:-1, -1]
    return torch.cat([interior, top, bottom, left, right], dim=0)


def generate_dataset(config: AllenCahnCNConfig, num_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    coords = build_allen_cahn_flattened_coordinates(config)
    x = torch.linspace(0.0, float(config.domain_length), config.nx)
    y = torch.linspace(0.0, float(config.domain_length), config.ny)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    n_steps = int(round(config.tmax / config.dt)) + 1
    flattened_solutions = torch.zeros((num_samples, coords.shape[0], n_steps), dtype=torch.float64)
    flattened_laplacians = torch.zeros_like(flattened_solutions)
    for sample in range(num_samples):
        initial = generate_allen_cahn_initial_condition(x_grid, y_grid, num_modes=4)
        solution, laplacian = solve_allen_cahn_crank_nicolson(initial, config)
        for step in range(n_steps):
            flattened_solutions[sample, :, step] = flatten_with_boundary_last(solution[step])
            flattened_laplacians[sample, :, step] = flatten_with_boundary_last(laplacian[step])
    return flattened_solutions, flattened_laplacians


def build_factor_pair(theta, dirac_points, derivative_groups, ordering, rho: float, nugget: float, sparse_backend: str = 'auto'):
    start = time.perf_counter()
    dense_factor = dense_precision_factor(theta, nugget=nugget)
    dense_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sparse_factor, _ = sparse_precision_factor(theta=theta, dirac_points=dirac_points, derivative_point_groups=derivative_groups, rho=rho, nugget=nugget, ordering=ordering, backend=sparse_backend)
    sparse_seconds = time.perf_counter() - start
    return {'dense': dense_factor, 'sparse': sparse_factor}, {'dense_seconds': dense_seconds, 'sparse_seconds': sparse_seconds}


def build_factors(config: AllenCahnCNConfig, train_sol: torch.Tensor, train_lap: torch.Tensor, rho: float, lengthscale: float, nugget: float = 1e-9, sparse_backend: str = 'auto'):
    coords = build_allen_cahn_flattened_coordinates(config)
    n_int = (config.nx - 2) * (config.ny - 2)
    interior_points = coords[:n_int]
    boundary_points = coords[n_int:]
    derivative_groups = (torch.arange(n_int, dtype=torch.long),)
    start = time.perf_counter()
    ordering = build_measurement_ordering(coords, derivative_groups)
    ordering_seconds = time.perf_counter() - start

    dirac_features = temporal_feature_matrix(train_sol)
    lap_features = temporal_feature_matrix(train_lap[:, :n_int, :])
    empirical_theta = build_allen_cahn_empirical_theta(dirac_features, lap_features, nugget=nugget) / dirac_features.shape[1]
    matern_assembly = build_elliptic_matern_theta(interior_points, boundary_points, lengthscale=lengthscale, nugget=nugget)
    empirical_factors, empirical_times = build_factor_pair(empirical_theta, coords, derivative_groups, ordering, rho=rho, nugget=nugget, sparse_backend=sparse_backend)
    matern_factors, matern_times = build_factor_pair(matern_assembly.theta, matern_assembly.dirac_points, matern_assembly.derivative_point_groups, ordering, rho=rho, nugget=nugget, sparse_backend=sparse_backend)
    return {
        'coords': coords,
        'n_int': n_int,
        'boundary_values': torch.zeros(coords.shape[0] - n_int, dtype=torch.float64),
        'empirical_sparse': empirical_factors['sparse'],
        'empirical_dense': empirical_factors['dense'],
        'matern_sparse': matern_factors['sparse'],
        'matern_dense': matern_factors['dense'],
        'build_times': {
            'ordering_seconds': ordering_seconds,
            'empirical_dense_factor_seconds': empirical_times['dense_seconds'],
            'empirical_sparse_factor_seconds': empirical_times['sparse_seconds'],
            'matern_dense_factor_seconds': matern_times['dense_seconds'],
            'matern_sparse_factor_seconds': matern_times['sparse_seconds'],
        },
    }


def rollout_allen_cahn_krom(initial_u: torch.Tensor, initial_lap: torch.Tensor, factor: object, config: AllenCahnCNConfig, boundary_values: torch.Tensor, gn_steps: int, gn_damping: float) -> torch.Tensor:
    n_int = (config.nx - 2) * (config.ny - 2)
    n_steps = int(round(config.tmax / config.dt)) + 1
    rollout = torch.zeros((n_int, n_steps), dtype=torch.float64)
    previous_u = initial_u[:n_int].clone()
    previous_lap = initial_lap[:n_int].clone()
    state = previous_u.clone()
    rollout[:, 0] = previous_u
    for step in range(n_steps - 1):
        saved_u = previous_u.clone()
        saved_lap = previous_lap.clone()
        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: allen_cahn_residual_operator(z, saved_u, saved_lap, boundary_values, config.epsilon, config.dt),
            factor=factor,
            max_iter=gn_steps,
            damping=gn_damping,
            record_history=False,
            linear_solver='auto',
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: allen_cahn_residual_and_jacobian(z, saved_u, saved_lap, boundary_values, config.epsilon, config.dt),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state = result.state
        previous_u = state.clone()
        previous_lap = 2.0 / config.epsilon**2 * ((previous_u - saved_u) / config.dt - 0.5 * ((saved_u - saved_u.pow(3)) + (previous_u - previous_u.pow(3)))) - saved_lap
        rollout[:, step + 1] = previous_u
    return rollout


def benchmark_full_order(config: AllenCahnCNConfig, test_sol: torch.Tensor, n_int: int) -> list[float]:
    times = []
    for sample in range(test_sol.shape[0]):
        initial_grid = torch.zeros((config.nx, config.ny), dtype=torch.float64)
        initial_grid[1:-1, 1:-1] = test_sol[sample, :n_int, 0].reshape(config.nx - 2, config.ny - 2)
        times.append(benchmark_callable(solve_allen_cahn_crank_nicolson, initial_grid, config, warmup=0, repeats=1).seconds)
    return times


def evaluate_factor(factor: object, config: AllenCahnCNConfig, test_sol: torch.Tensor, test_lap: torch.Tensor, boundary_values: torch.Tensor, n_int: int, gn_steps: int, gn_damping: float, full_order_times: list[float]) -> dict[str, object]:
    final_errors = []
    krom_times = []
    for sample in range(test_sol.shape[0]):
        start = time.perf_counter()
        rollout = rollout_allen_cahn_krom(test_sol[sample, :, 0], test_lap[sample, :, 0], factor, config, boundary_values, gn_steps, gn_damping)
        krom_times.append(time.perf_counter() - start)
        final_errors.append(relative_l2(rollout[:, -1], test_sol[sample, :n_int, -1]))
    return {
        'mean_final_rel_l2': sum(final_errors) / len(final_errors),
        'mean_krom_seconds': sum(krom_times) / len(krom_times),
        'per_case_seconds': krom_times,
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }


def run_experiment(config: AllenCahnCNConfig = DEFAULT_CONFIG, rho: float = 4.0, lengthscale: float = 0.30, gn_steps: int = 2, gn_damping: float = 1e-8, num_train: int = 8, num_test: int = 8, train_seed: int = 0, test_seed: int = 1, sparse_backend: str = 'auto') -> dict[str, object]:
    start = time.perf_counter()
    train_sol, train_lap = generate_dataset(config, num_train, train_seed)
    train_snapshot_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_sol, test_lap = generate_dataset(config, num_test, test_seed)
    test_snapshot_seconds = time.perf_counter() - start
    factors = build_factors(config, train_sol, train_lap, rho=rho, lengthscale=lengthscale, sparse_backend=sparse_backend)
    full_order_times = benchmark_full_order(config, test_sol, factors['n_int'])

    empirical_sparse = evaluate_factor(factors['empirical_sparse'], config, test_sol, test_lap, factors['boundary_values'], factors['n_int'], gn_steps, gn_damping, full_order_times)
    empirical_dense = evaluate_factor(factors['empirical_dense'], config, test_sol, test_lap, factors['boundary_values'], factors['n_int'], gn_steps, gn_damping, full_order_times)
    matern_sparse = evaluate_factor(factors['matern_sparse'], config, test_sol, test_lap, factors['boundary_values'], factors['n_int'], gn_steps, gn_damping, full_order_times)
    matern_dense = evaluate_factor(factors['matern_dense'], config, test_sol, test_lap, factors['boundary_values'], factors['n_int'], gn_steps, gn_damping, full_order_times)

    sample_index = 0
    empirical_rollout = rollout_allen_cahn_krom(test_sol[sample_index, :, 0], test_lap[sample_index, :, 0], factors['empirical_sparse'], config, factors['boundary_values'], gn_steps, gn_damping)
    matern_rollout = rollout_allen_cahn_krom(test_sol[sample_index, :, 0], test_lap[sample_index, :, 0], factors['matern_sparse'], config, factors['boundary_values'], gn_steps, gn_damping)
    truth = test_sol[sample_index, : factors['n_int'], -1].reshape(config.nx - 2, config.ny - 2)
    return {
        'config': config,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_train': num_train,
        'num_test': num_test,
        'truth': truth,
        'empirical_field': empirical_rollout[:, -1].reshape(config.nx - 2, config.ny - 2),
        'matern_field': matern_rollout[:, -1].reshape(config.nx - 2, config.ny - 2),
        'empirical': empirical_sparse,
        'matern': matern_sparse,
        'empirical_sparse': empirical_sparse,
        'empirical_dense': empirical_dense,
        'matern_sparse': matern_sparse,
        'matern_dense': matern_dense,
        'timings': {
            'train_snapshot_seconds': train_snapshot_seconds,
            'test_snapshot_seconds': test_snapshot_seconds,
            'factor_build_seconds': factors['build_times'],
        },
    }


def plot_experiment(result: dict[str, object]):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(result['truth'], origin='lower')
    axes[0].set_title('Full-order CN')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(result['empirical_field'], origin='lower')
    axes[1].set_title('Empirical sparse')
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(result['matern_field'], origin='lower')
    axes[2].set_title('Matern sparse')
    plt.colorbar(im2, ax=axes[2])
    axes[3].bar(
        ['Emp-S', 'Emp-D', 'Mat-S', 'Mat-D'],
        [
            result['empirical_sparse']['mean_krom_seconds'],
            result['empirical_dense']['mean_krom_seconds'],
            result['matern_sparse']['mean_krom_seconds'],
            result['matern_dense']['mean_krom_seconds'],
        ],
    )
    axes[3].axhline(result['empirical_sparse']['mean_full_order_seconds'], color='black', linestyle='--')
    axes[3].set_title('Dense vs sparse runtime')
    axes[3].set_ylabel('Seconds per rollout')
    fig.tight_layout()
    return fig, axes


def run_rho_sweep(rho_values: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=rho_values, evaluator=lambda value: run_experiment(rho=float(value), **kwargs), metric_name='mean_final_rel_l2', parameter_name='rho', ylabel='Final relative L2 error', title='Allen-Cahn error vs sparse radius')


def run_train_sweep(train_sizes: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=train_sizes, evaluator=lambda value: run_experiment(num_train=int(value), **kwargs), metric_name='mean_final_rel_l2', parameter_name='num_train', ylabel='Final relative L2 error', title='Allen-Cahn error vs number of training solutions')


def plot_sweeps(rho_sweep, train_sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_sweep_result(rho_sweep, ax=axes[0])
    plot_sweep_result(train_sweep, ax=axes[1])
    fig.tight_layout()
    return fig, axes


# %%
if __name__ == '__main__':
    default_result = run_experiment()
    plot_experiment(default_result)
    rho_sweep = run_rho_sweep([2.0, 3.0, 4.0, 5.0])
    train_sweep = run_train_sweep([4, 8, 12])
    plot_sweeps(rho_sweep, train_sweep)
    plt.show()
