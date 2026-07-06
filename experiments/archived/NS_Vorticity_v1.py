# %%
from __future__ import annotations

import math
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
from src.krom.navier_stokes import build_navier_stokes_factors, evaluate_streamfunction_kernel, relative_l2, rollout_navier_stokes_krom
from src.krom.pde_baselines import NSVorticityConfig, compute_energy_spectrum, navier_stokes_vorticity_dataset, navier_stokes_vorticity_rollout
from src.krom.sweeps import plot_sweep_result, run_comparison_sweep


torch.set_default_dtype(torch.float64)

DEFAULT_CONFIG = NSVorticityConfig(nx=24, domain_length=math.pi, dt=1e-2, tmax=1.0, viscosity=1e-3, modes=6, snapshot_stride=1)


def benchmark_full_order(initial_conditions: torch.Tensor, config: NSVorticityConfig) -> list[float]:
    return [benchmark_callable(navier_stokes_vorticity_rollout, initial_conditions[sample], config, warmup=0, repeats=1).seconds for sample in range(initial_conditions.shape[0])]


def evaluate_kernel_kind(kernel_kind: str, factor_mode: str, poisson_solver: str, factor_bundle, config: NSVorticityConfig, test_solutions: torch.Tensor, test_dx: torch.Tensor, test_dy: torch.Tensor, test_lap: torch.Tensor, initial_conditions: torch.Tensor, gn_steps: int, gn_damping: float, full_order_times: list[float]) -> dict[str, object]:
    final_errors = []
    energy_errors = []
    krom_times = []
    vorticity_factor = factor_bundle.vorticity_factor(kernel_kind, factor_mode)
    streamfunction_operator = None if poisson_solver == 'fft' else factor_bundle.streamfunction_operator(kernel_kind, factor_mode)
    for sample in range(test_solutions.shape[0]):
        start = time.perf_counter()
        rollout = rollout_navier_stokes_krom(
            initial_omega=test_solutions[sample, :, 0].reshape(config.nx, config.nx),
            initial_dx=test_dx[sample, :, 0],
            initial_dy=test_dy[sample, :, 0],
            initial_laplace=test_lap[sample, :, 0],
            vorticity_factor=vorticity_factor,
            config=config,
            gn_steps=gn_steps,
            gn_damping=gn_damping,
            poisson_solver=poisson_solver,
            streamfunction_operator=streamfunction_operator,
        )
        krom_times.append(time.perf_counter() - start)
        truth_final = test_solutions[sample, :, -1]
        pred_final = rollout['solutions'][:, -1]
        final_errors.append(relative_l2(pred_final, truth_final))
        truth_spectrum = compute_energy_spectrum(truth_final.reshape(config.nx, config.nx), config)[1:]
        pred_spectrum = compute_energy_spectrum(pred_final.reshape(config.nx, config.nx), config)[1:]
        energy_errors.append(relative_l2(pred_spectrum, truth_spectrum))
    summary = {
        'mean_final_rel_l2': sum(final_errors) / len(final_errors),
        'mean_energy_spectrum_rel_l2': sum(energy_errors) / len(energy_errors),
        'mean_krom_seconds': sum(krom_times) / len(krom_times),
        'per_case_seconds': krom_times,
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }
    if poisson_solver == 'kernel':
        summary.update(evaluate_streamfunction_kernel(test_solutions, config, streamfunction_operator))
    else:
        summary['mean_streamfunction_rel_l2'] = 0.0
        summary['mean_velocity_rel_l2'] = 0.0
    return summary


def run_experiment(config: NSVorticityConfig = DEFAULT_CONFIG, poisson_solver: str = 'fft', rho: float = 3.0, lengthscale: float = 0.35, gn_steps: int = 2, gn_damping: float = 1e-8, num_train: int = 8, num_test: int = 6, train_seed: int = 0, test_seed: int = 1, sparse_backend: str = 'auto') -> dict[str, object]:
    torch.manual_seed(train_seed)
    start = time.perf_counter()
    points, times, train_solutions, train_dx, train_dy, train_lap, _ = navier_stokes_vorticity_dataset(num_train, config)
    train_snapshot_seconds = time.perf_counter() - start
    torch.manual_seed(test_seed)
    start = time.perf_counter()
    _, _, test_solutions, test_dx, test_dy, test_lap, test_initial = navier_stokes_vorticity_dataset(num_test, config)
    test_snapshot_seconds = time.perf_counter() - start
    factor_bundle = build_navier_stokes_factors(points=points, train_solutions=train_solutions, train_dx=train_dx, train_dy=train_dy, train_laplacians=train_lap, config=config, rho=rho, lengthscale=lengthscale, nugget=1e-9, sparse_backend=sparse_backend)
    full_order_times = benchmark_full_order(test_initial, config)

    empirical_sparse = evaluate_kernel_kind('empirical', 'sparse', poisson_solver, factor_bundle, config, test_solutions, test_dx, test_dy, test_lap, test_initial, gn_steps, gn_damping, full_order_times)
    empirical_dense = evaluate_kernel_kind('empirical', 'dense', poisson_solver, factor_bundle, config, test_solutions, test_dx, test_dy, test_lap, test_initial, gn_steps, gn_damping, full_order_times)
    matern_sparse = evaluate_kernel_kind('matern', 'sparse', poisson_solver, factor_bundle, config, test_solutions, test_dx, test_dy, test_lap, test_initial, gn_steps, gn_damping, full_order_times)
    matern_dense = evaluate_kernel_kind('matern', 'dense', poisson_solver, factor_bundle, config, test_solutions, test_dx, test_dy, test_lap, test_initial, gn_steps, gn_damping, full_order_times)

    sample_index = 0
    empirical_rollout = rollout_navier_stokes_krom(
        initial_omega=test_solutions[sample_index, :, 0].reshape(config.nx, config.nx),
        initial_dx=test_dx[sample_index, :, 0],
        initial_dy=test_dy[sample_index, :, 0],
        initial_laplace=test_lap[sample_index, :, 0],
        vorticity_factor=factor_bundle.vorticity_factor('empirical', 'sparse'),
        config=config,
        gn_steps=gn_steps,
        gn_damping=gn_damping,
        poisson_solver=poisson_solver,
        streamfunction_operator=None if poisson_solver == 'fft' else factor_bundle.streamfunction_operator('empirical', 'sparse'),
    )
    matern_rollout = rollout_navier_stokes_krom(
        initial_omega=test_solutions[sample_index, :, 0].reshape(config.nx, config.nx),
        initial_dx=test_dx[sample_index, :, 0],
        initial_dy=test_dy[sample_index, :, 0],
        initial_laplace=test_lap[sample_index, :, 0],
        vorticity_factor=factor_bundle.vorticity_factor('matern', 'sparse'),
        config=config,
        gn_steps=gn_steps,
        gn_damping=gn_damping,
        poisson_solver=poisson_solver,
        streamfunction_operator=None if poisson_solver == 'fft' else factor_bundle.streamfunction_operator('matern', 'sparse'),
    )
    truth_final = test_solutions[sample_index, :, -1].reshape(config.nx, config.nx)
    empirical_final = empirical_rollout['solutions'][:, -1].reshape(config.nx, config.nx)
    matern_final = matern_rollout['solutions'][:, -1].reshape(config.nx, config.nx)
    return {
        'config': config,
        'poisson_solver': poisson_solver,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_train': num_train,
        'num_test': num_test,
        'times': times,
        'truth_final': truth_final,
        'empirical_final': empirical_final,
        'matern_final': matern_final,
        'truth_spectrum': compute_energy_spectrum(truth_final, config)[1:],
        'empirical_spectrum': compute_energy_spectrum(empirical_final, config)[1:],
        'matern_spectrum': compute_energy_spectrum(matern_final, config)[1:],
        'empirical': empirical_sparse,
        'matern': matern_sparse,
        'empirical_sparse': empirical_sparse,
        'empirical_dense': empirical_dense,
        'matern_sparse': matern_sparse,
        'matern_dense': matern_dense,
        'timings': {
            'train_snapshot_seconds': train_snapshot_seconds,
            'test_snapshot_seconds': test_snapshot_seconds,
            'factor_build_seconds': factor_bundle.build_times,
        },
    }


def plot_experiment(result: dict[str, object]):
    empirical = result['empirical']
    matern = result['matern']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    im0 = axes[0, 0].contourf(result['truth_final'], levels=40)
    axes[0, 0].set_title('Full-order vorticity')
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].contourf(result['empirical_final'], levels=40)
    axes[0, 1].set_title(f"Empirical sparse ({result['poisson_solver']} Poisson)")
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[0, 2].contourf(result['matern_final'], levels=40)
    axes[0, 2].set_title(f"Matern sparse ({result['poisson_solver']} Poisson)")
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
    wavenumbers = torch.arange(1, result['truth_spectrum'].numel() + 1)
    axes[1, 1].loglog(wavenumbers, result['truth_spectrum'], linewidth=2)
    axes[1, 1].loglog(wavenumbers, result['empirical_spectrum'], linestyle='--')
    axes[1, 1].loglog(wavenumbers, result['matern_spectrum'], linestyle=':')
    axes[1, 1].set_title('Energy spectrum')
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('E(k)')
    axes[1, 1].legend(['Full-order', 'Empirical sparse', 'Matern sparse'])
    if result['poisson_solver'] == 'kernel':
        axes[1, 2].bar(
            ['Emp-S psi', 'Emp-D psi', 'Mat-S psi', 'Mat-D psi'],
            [
                result['empirical_sparse']['mean_streamfunction_rel_l2'],
                result['empirical_dense']['mean_streamfunction_rel_l2'],
                result['matern_sparse']['mean_streamfunction_rel_l2'],
                result['matern_dense']['mean_streamfunction_rel_l2'],
            ],
        )
        axes[1, 2].set_title('Kernel Poisson validation')
        axes[1, 2].set_ylabel('Relative L2 error')
    else:
        axes[1, 2].axis('off')
        axes[1, 2].text(0.05, 0.65, 'FFT Poisson is the practical fast-path baseline.', transform=axes[1, 2].transAxes, va='top')
    fig.tight_layout()
    return fig, axes


def run_rho_sweep(rho_values: Sequence[float], poisson_solver: str = 'fft', **kwargs) -> object:
    return run_comparison_sweep(parameter_values=rho_values, evaluator=lambda value: run_experiment(rho=float(value), poisson_solver=poisson_solver, **kwargs), metric_name='mean_final_rel_l2', parameter_name='rho', ylabel='Final relative L2 error', title=f'NS error vs sparse radius ({poisson_solver} Poisson)')


def run_train_sweep(train_sizes: Sequence[float], poisson_solver: str = 'fft', **kwargs) -> object:
    return run_comparison_sweep(parameter_values=train_sizes, evaluator=lambda value: run_experiment(num_train=int(value), poisson_solver=poisson_solver, **kwargs), metric_name='mean_final_rel_l2', parameter_name='num_train', ylabel='Final relative L2 error', title=f'NS error vs number of training solutions ({poisson_solver} Poisson)')


def plot_sweeps(rho_sweep, train_sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_sweep_result(rho_sweep, ax=axes[0])
    plot_sweep_result(train_sweep, ax=axes[1])
    fig.tight_layout()
    return fig, axes


# %%
if __name__ == '__main__':
    fft_result = run_experiment(poisson_solver='fft')
    plot_experiment(fft_result)
    kernel_result = run_experiment(poisson_solver='kernel')
    plot_experiment(kernel_result)
    fft_rho = run_rho_sweep([2.0, 3.0, 4.0], poisson_solver='fft')
    fft_train = run_train_sweep([4, 8, 12], poisson_solver='fft')
    plot_sweeps(fft_rho, fft_train)
    kernel_rho = run_rho_sweep([2.0, 3.0, 4.0], poisson_solver='kernel')
    kernel_train = run_train_sweep([4, 8, 12], poisson_solver='kernel')
    plot_sweeps(kernel_rho, kernel_train)
    plt.show()
