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
from src.krom.pde_baselines import BurgersCNConfig, burgers_crank_nicolson_dataset, burgers_crank_nicolson_rollout
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.sweeps import plot_sweep_result, run_comparison_sweep
from src.krom.workflows import build_burgers_empirical_theta, build_burgers_matern_theta, burgers_residual_and_jacobian, burgers_residual_operator


torch.set_default_dtype(torch.float64)

DEFAULT_CONFIG = BurgersCNConfig(nx=2001, dt=0.02, tmax=1.0, viscosity=1e-3, newton_max_iter=20)
AUTO_DIRECT_THRESHOLD = 2048


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def build_factor_pair(theta, dirac_points, derivative_groups, ordering, rho: float, nugget: float, sparse_backend: str = 'auto'):
    start = time.perf_counter()
    dense_factor = dense_precision_factor(theta, nugget=nugget)
    dense_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sparse_factor, _ = sparse_precision_factor(
        theta=theta,
        dirac_points=dirac_points,
        derivative_point_groups=derivative_groups,
        rho=rho,
        nugget=nugget,
        ordering=ordering,
        backend=sparse_backend,
    )
    sparse_seconds = time.perf_counter() - start
    return {'dense': dense_factor, 'sparse': sparse_factor}, {'dense_seconds': dense_seconds, 'sparse_seconds': sparse_seconds}


def build_factors(x: torch.Tensor, train_u: torch.Tensor, train_ux: torch.Tensor, train_uxx: torch.Tensor, rho: float, lengthscale: float, nugget: float = 1e-9, sparse_backend: str = 'auto'):
    n_int = x.numel() - 2
    interior_points = x[1:-1, None]
    boundary_points = torch.tensor([[x[0]], [x[-1]]], dtype=x.dtype, device=x.device)
    dirac_points = torch.cat([interior_points, boundary_points], dim=0)
    derivative_groups = (
        torch.arange(n_int, dtype=torch.long, device=x.device),
        torch.arange(n_int, dtype=torch.long, device=x.device),
    )
    start = time.perf_counter()
    ordering = build_measurement_ordering(dirac_points, derivative_groups)
    ordering_seconds = time.perf_counter() - start

    dirac_snapshots = torch.cat([train_u[:, 1:-1, :], train_u[:, [0, -1], :]], dim=1)
    dirac_features = temporal_feature_matrix(dirac_snapshots)
    ux_features = temporal_feature_matrix(train_ux[:, 1:-1, :])
    uxx_features = temporal_feature_matrix(train_uxx[:, 1:-1, :])
    empirical_theta = build_burgers_empirical_theta(dirac_features, ux_features, uxx_features, nugget=nugget) / dirac_features.shape[1]
    matern_assembly = build_burgers_matern_theta(interior_points=interior_points, boundary_points=boundary_points, lengthscale=lengthscale, nugget=nugget)

    empirical_factors, empirical_times = build_factor_pair(empirical_theta, dirac_points, derivative_groups, ordering, rho=rho, nugget=nugget, sparse_backend=sparse_backend)
    matern_factors, matern_times = build_factor_pair(matern_assembly.theta, matern_assembly.dirac_points, matern_assembly.derivative_point_groups, ordering, rho=rho, nugget=nugget, sparse_backend=sparse_backend)

    return {
        'boundary_values': torch.zeros(2, dtype=x.dtype, device=x.device),
        'ordering': ordering,
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


def rollout_burgers_krom(initial_u: torch.Tensor, initial_ux: torch.Tensor, initial_uxx: torch.Tensor, factor: object, config: BurgersCNConfig, boundary_values: torch.Tensor, gn_steps: int, gn_damping: float) -> torch.Tensor:
    n_int = config.nx - 2
    n_steps = int(round(config.tmax / config.dt)) + 1
    rollout = torch.zeros((n_int, n_steps), dtype=initial_u.dtype, device=initial_u.device)
    previous_u = initial_u[1:-1].clone()
    previous_ux = initial_ux[1:-1].clone()
    previous_uxx = initial_uxx[1:-1].clone()
    state = torch.cat([previous_u, previous_ux], dim=0)
    rollout[:, 0] = previous_u
    for step in range(n_steps - 1):
        saved_u = previous_u.clone()
        saved_ux = previous_ux.clone()
        saved_uxx = previous_uxx.clone()
        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: burgers_residual_operator(z, saved_u, saved_ux, saved_uxx, boundary_values, config.viscosity, config.dt),
            factor=factor,
            max_iter=gn_steps,
            damping=gn_damping,
            record_history=False,
            linear_solver='auto',
            preconditioner='jacobi',
            direct_residual_and_jacobian=lambda z: burgers_residual_and_jacobian(z, saved_u, saved_ux, saved_uxx, boundary_values, config.viscosity, config.dt),
            direct_threshold=AUTO_DIRECT_THRESHOLD,
        )
        state = result.state
        previous_u = state[:n_int].clone()
        previous_ux = state[n_int:].clone()
        previous_uxx = 2.0 / config.viscosity * ((previous_u - saved_u) / config.dt + 0.5 * (previous_u * previous_ux + saved_u * saved_ux)) - saved_uxx
        rollout[:, step + 1] = previous_u
    return rollout


def benchmark_full_order(test_u: torch.Tensor, config: BurgersCNConfig) -> list[float]:
    return [benchmark_callable(burgers_crank_nicolson_rollout, test_u[sample, :, 0], config, warmup=0, repeats=1).seconds for sample in range(test_u.shape[0])]


def evaluate_factor(factor: object, test_u: torch.Tensor, test_ux: torch.Tensor, test_uxx: torch.Tensor, config: BurgersCNConfig, boundary_values: torch.Tensor, gn_steps: int, gn_damping: float, full_order_times: list[float]) -> dict[str, object]:
    final_errors = []
    krom_times = []
    for sample in range(test_u.shape[0]):
        start = time.perf_counter()
        rollout = rollout_burgers_krom(test_u[sample, :, 0], test_ux[sample, :, 0], test_uxx[sample, :, 0], factor, config, boundary_values, gn_steps, gn_damping)
        krom_times.append(time.perf_counter() - start)
        final_errors.append(relative_l2(rollout[:, -1], test_u[sample, 1:-1, -1]))
    return {
        'mean_final_rel_l2': sum(final_errors) / len(final_errors),
        'mean_krom_seconds': sum(krom_times) / len(krom_times),
        'per_case_seconds': krom_times,
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }


def run_experiment(config: BurgersCNConfig = DEFAULT_CONFIG, rho: float = 4.0, lengthscale: float = 0.30, gn_steps: int = 3, gn_damping: float = 1e-8, num_train: int = 10, num_test: int = 12, train_seed: int = 0, test_seed: int = 1, sparse_backend: str = 'auto') -> dict[str, object]:
    torch.manual_seed(train_seed)
    start = time.perf_counter()
    x, times, train_u, train_ux, train_uxx = burgers_crank_nicolson_dataset(num_train, config, num_terms=4)
    train_snapshot_seconds = time.perf_counter() - start
    torch.manual_seed(test_seed)
    start = time.perf_counter()
    _, _, test_u, test_ux, test_uxx = burgers_crank_nicolson_dataset(num_test, config, num_terms=4)
    test_snapshot_seconds = time.perf_counter() - start
    factors = build_factors(x, train_u, train_ux, train_uxx, rho=rho, lengthscale=lengthscale, sparse_backend=sparse_backend)
    full_order_times = benchmark_full_order(test_u, config)

    empirical_sparse = evaluate_factor(factors['empirical_sparse'], test_u, test_ux, test_uxx, config, factors['boundary_values'], gn_steps, gn_damping, full_order_times)
    empirical_dense = evaluate_factor(factors['empirical_dense'], test_u, test_ux, test_uxx, config, factors['boundary_values'], gn_steps, gn_damping, full_order_times)
    matern_sparse = evaluate_factor(factors['matern_sparse'], test_u, test_ux, test_uxx, config, factors['boundary_values'], gn_steps, gn_damping, full_order_times)
    matern_dense = evaluate_factor(factors['matern_dense'], test_u, test_ux, test_uxx, config, factors['boundary_values'], gn_steps, gn_damping, full_order_times)

    sample_index = 0
    empirical_rollout = rollout_burgers_krom(test_u[sample_index, :, 0], test_ux[sample_index, :, 0], test_uxx[sample_index, :, 0], factors['empirical_sparse'], config, factors['boundary_values'], gn_steps, gn_damping)
    matern_rollout = rollout_burgers_krom(test_u[sample_index, :, 0], test_ux[sample_index, :, 0], test_uxx[sample_index, :, 0], factors['matern_sparse'], config, factors['boundary_values'], gn_steps, gn_damping)

    return {
        'config': config,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_train': num_train,
        'num_test': num_test,
        'x': x,
        'times': times,
        'truth': test_u[sample_index, 1:-1, :],
        'empirical_rollout': empirical_rollout,
        'matern_rollout': matern_rollout,
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
    x = result['x']
    truth = result['truth']
    empirical_rollout = result['empirical_rollout']
    matern_rollout = result['matern_rollout']
    labels = ['Emp-S', 'Emp-D', 'Mat-S', 'Mat-D']
    values = [
        result['empirical_sparse']['mean_krom_seconds'],
        result['empirical_dense']['mean_krom_seconds'],
        result['matern_sparse']['mean_krom_seconds'],
        result['matern_dense']['mean_krom_seconds'],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x[1:-1], truth[:, -1], linewidth=2)
    axes[0].plot(x[1:-1], empirical_rollout[:, -1], linestyle='--')
    axes[0].plot(x[1:-1], matern_rollout[:, -1], linestyle=':')
    axes[0].set_title('Burgers final-time comparison')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x, t_final)')
    axes[0].legend(['Full-order CN', 'Empirical sparse', 'Matern sparse'])

    axes[1].bar(labels, values)
    axes[1].axhline(result['empirical_sparse']['mean_full_order_seconds'], color='black', linestyle='--')
    axes[1].set_ylabel('Seconds per rollout')
    axes[1].set_title('Dense vs sparse kernel runtime')
    fig.tight_layout()
    return fig, axes


def run_rho_sweep(rho_values: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=rho_values, evaluator=lambda value: run_experiment(rho=float(value), **kwargs), metric_name='mean_final_rel_l2', parameter_name='rho', ylabel='Final relative L2 error', title='Burgers error vs sparse radius')


def run_train_sweep(train_sizes: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=train_sizes, evaluator=lambda value: run_experiment(num_train=int(value), **kwargs), metric_name='mean_final_rel_l2', parameter_name='num_train', ylabel='Final relative L2 error', title='Burgers error vs number of training solutions')


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
    rho_sweep = run_rho_sweep([1.0,2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0,9.0,10.0])
    train_sweep = run_train_sweep([10, 20, 30,40,50,60,70,80,90,100])
    plot_sweeps(rho_sweep, train_sweep)
    plt.show()
