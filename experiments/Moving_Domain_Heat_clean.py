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
from src.krom.moving_domain import (
    MovingDomainHeatConfig,
    breathing_scale,
    build_moving_domain_problem,
    build_moving_heat_factors,
    moving_heat_dataset,
    moving_heat_rollout,
    rollout_moving_heat_krom,
)
from src.krom.sweeps import plot_sweep_result, run_comparison_sweep


torch.set_default_dtype(torch.float64)

DEFAULT_CONFIG = MovingDomainHeatConfig(
    domain_length=1.0,
    nx=31,
    ny=31,
    dt=1e-2,
    tmax=1.0,
    diffusivity=5e-3,
    breathing_amplitude=0.18,
    breathing_cycles=1.0,
)


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def benchmark_full_order(problem, config: MovingDomainHeatConfig, test_initial_conditions: torch.Tensor) -> list[float]:
    return [
        benchmark_callable(moving_heat_rollout, test_initial_conditions[sample], problem, config, warmup=0, repeats=1).seconds
        for sample in range(test_initial_conditions.shape[0])
    ]


def evaluate_factor(
    problem,
    factor: object,
    config: MovingDomainHeatConfig,
    test_solutions: torch.Tensor,
    test_dx: torch.Tensor,
    test_dy: torch.Tensor,
    test_lap: torch.Tensor,
    gn_steps: int,
    gn_damping: float,
    full_order_times: list[float],
) -> dict[str, object]:
    final_errors = []
    krom_times = []
    for sample in range(test_solutions.shape[0]):
        start = time.perf_counter()
        rollout = rollout_moving_heat_krom(
            initial_solution=test_solutions[sample, :, 0],
            initial_dx=test_dx[sample, :, 0],
            initial_dy=test_dy[sample, :, 0],
            initial_laplace=test_lap[sample, :, 0],
            factor=factor,
            problem=problem,
            config=config,
            gn_steps=gn_steps,
            gn_damping=gn_damping,
        )
        krom_times.append(time.perf_counter() - start)
        final_errors.append(relative_l2(rollout['solutions'][: problem.n_int, -1], test_solutions[sample, : problem.n_int, -1]))
    return {
        'mean_final_rel_l2': sum(final_errors) / len(final_errors),
        'mean_krom_seconds': sum(krom_times) / len(krom_times),
        'per_case_seconds': krom_times,
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }


def run_experiment(
    config: MovingDomainHeatConfig = DEFAULT_CONFIG,
    rho: float = 3.0,
    lengthscale: float = 0.22,
    gn_steps: int = 1,
    gn_damping: float = 1e-8,
    num_train: int = 10,
    num_test: int = 8,
    train_seed: int = 0,
    test_seed: int = 1,
    sparse_backend: str = 'auto',
    k_neighbors: int = 3,
) -> dict[str, object]:
    problem = build_moving_domain_problem(config)

    torch.manual_seed(train_seed)
    start = time.perf_counter()
    times, train_solutions, train_dx, train_dy, train_lap = moving_heat_dataset(num_train, problem, config)
    train_snapshot_seconds = time.perf_counter() - start

    torch.manual_seed(test_seed)
    start = time.perf_counter()
    _, test_solutions, test_dx, test_dy, test_lap = moving_heat_dataset(num_test, problem, config)
    test_snapshot_seconds = time.perf_counter() - start

    factors = build_moving_heat_factors(
        problem,
        train_solutions,
        train_dx,
        train_dy,
        train_laplacians=train_lap,
        rho=rho,
        lengthscale=lengthscale,
        sparse_backend=sparse_backend,
        k_neighbors=k_neighbors,
    )
    full_order_times = benchmark_full_order(problem, config, test_solutions[:, :, 0])

    empirical_sparse = evaluate_factor(problem, factors['empirical_sparse'], config, test_solutions, test_dx, test_dy, test_lap, gn_steps, gn_damping, full_order_times)
    empirical_dense = evaluate_factor(problem, factors['empirical_dense'], config, test_solutions, test_dx, test_dy, test_lap, gn_steps, gn_damping, full_order_times)
    matern_sparse = evaluate_factor(problem, factors['matern_sparse'], config, test_solutions, test_dx, test_dy, test_lap, gn_steps, gn_damping, full_order_times)
    matern_dense = evaluate_factor(problem, factors['matern_dense'], config, test_solutions, test_dx, test_dy, test_lap, gn_steps, gn_damping, full_order_times)

    sample_index = 0
    empirical_rollout = rollout_moving_heat_krom(
        initial_solution=test_solutions[sample_index, :, 0],
        initial_dx=test_dx[sample_index, :, 0],
        initial_dy=test_dy[sample_index, :, 0],
        initial_laplace=test_lap[sample_index, :, 0],
        factor=factors['empirical_sparse'],
        problem=problem,
        config=config,
        gn_steps=gn_steps,
        gn_damping=gn_damping,
    )
    matern_rollout = rollout_moving_heat_krom(
        initial_solution=test_solutions[sample_index, :, 0],
        initial_dx=test_dx[sample_index, :, 0],
        initial_dy=test_dy[sample_index, :, 0],
        initial_laplace=test_lap[sample_index, :, 0],
        factor=factors['matern_sparse'],
        problem=problem,
        config=config,
        gn_steps=gn_steps,
        gn_damping=gn_damping,
    )

    scale_history = breathing_scale(times, config).to(dtype=times.dtype, device=times.device)
    return {
        'config': config,
        'problem': problem,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_train': num_train,
        'num_test': num_test,
        'times': times,
        'domain_scale': scale_history,
        'truth': test_solutions[sample_index, : problem.n_int, -1].reshape(config.nx - 2, config.ny - 2),
        'empirical_field': empirical_rollout['solutions'][: problem.n_int, -1].reshape(config.nx - 2, config.ny - 2),
        'matern_field': matern_rollout['solutions'][: problem.n_int, -1].reshape(config.nx - 2, config.ny - 2),
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
    config = result['config']
    problem = result['problem']
    x = problem.interior_points[:, 0].reshape(config.nx - 2, config.ny - 2)
    y = problem.interior_points[:, 1].reshape(config.nx - 2, config.ny - 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    im0 = axes[0, 0].contourf(x, y, result['truth'], levels=40)
    axes[0, 0].set_title('Full-order final field')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].contourf(x, y, result['empirical_field'], levels=40)
    axes[0, 1].set_title('Empirical sparse')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].contourf(x, y, result['matern_field'], levels=40)
    axes[0, 2].set_title('Matérn sparse')
    plt.colorbar(im2, ax=axes[0, 2])

    axes[1, 0].bar(
        ['Emp-S', 'Emp-D', 'Mat-S', 'Mat-D'],
        [
            result['empirical_sparse']['mean_krom_seconds'],
            result['empirical_dense']['mean_krom_seconds'],
            result['matern_sparse']['mean_krom_seconds'],
            result['matern_dense']['mean_krom_seconds'],
        ],
    )
    axes[1, 0].axhline(result['empirical_sparse']['mean_full_order_seconds'], color='black', linestyle='--')
    axes[1, 0].set_title('Dense vs sparse runtime')
    axes[1, 0].set_ylabel('Seconds per rollout')

    axes[1, 1].plot(result['times'], result['domain_scale'], linewidth=2)
    axes[1, 1].set_title('Breathing scale factor')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('s(t)')
    axes[1, 1].grid(True, linestyle='--', linewidth=0.5)

    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.02,
        0.98,
        (
            f"Reference domain: [0,1]^2\n"
            f"Boundary condition: u=0\n"
            f"Motion: x = c + s(t)(xhat-c)\n"
            f"Amplitude = {config.breathing_amplitude:.2f}, cycles = {config.breathing_cycles:.1f}\n"
            f"Emp-S err = {result['empirical_sparse']['mean_final_rel_l2']:.3e}\n"
            f"Mat-S err = {result['matern_sparse']['mean_final_rel_l2']:.3e}"
        ),
        transform=axes[1, 2].transAxes,
        va='top',
        ha='left',
    )
    fig.tight_layout()
    return fig, axes


def run_rho_sweep(rho_values: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(
        parameter_values=rho_values,
        evaluator=lambda value: run_experiment(rho=float(value), **kwargs),
        metric_name='mean_final_rel_l2',
        parameter_name='rho',
        ylabel='Final relative L2 error',
        title='Moving-domain heat error vs sparse radius',
    )


def run_train_sweep(train_sizes: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(
        parameter_values=train_sizes,
        evaluator=lambda value: run_experiment(num_train=int(value), **kwargs),
        metric_name='mean_final_rel_l2',
        parameter_name='num_train',
        ylabel='Final relative L2 error',
        title='Moving-domain heat error vs number of training solutions',
    )


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
    train_sweep = run_train_sweep([6, 10, 14])
    plot_sweeps(rho_sweep, train_sweep)
    plt.show()
