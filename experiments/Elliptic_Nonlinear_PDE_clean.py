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
from src.krom.factors import dense_precision_factor
from src.krom.gauss_newton import solve_gauss_newton
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import AllenCahnCNConfig, build_allen_cahn_flattened_coordinates
from src.krom.sparse_cholesky import SnapshotKernelSource, make_snapshot_kernel_source, sparse_precision_factor
from src.krom.sweeps import plot_sweep_result, run_comparison_sweep
from src.krom.workflows import build_elliptic_empirical_theta, build_elliptic_matern_theta, elliptic_residual_and_jacobian, elliptic_residual_operator


torch.set_default_dtype(torch.float64)

DEFAULT_GRID_CONFIG = AllenCahnCNConfig(nx=33, ny=33, dt=1.0, tmax=1.0, domain_length=1.0)
AUTO_DIRECT_THRESHOLD = 2048


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)


def rhs_f(x: torch.Tensor, y: torch.Tensor, alpha: float, power: int) -> torch.Tensor:
    term1 = -torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    term2 = -8.0 * torch.pi**2 * torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    return -(term1 + term2) + alpha * u_exact(x, y).pow(power)


def build_full_order_matrix(grid_config: AllenCahnCNConfig) -> torch.Tensor:
    nx_int = grid_config.nx - 2
    ny_int = grid_config.ny - 2
    dx = grid_config.domain_length / (grid_config.nx - 1)
    dy = grid_config.domain_length / (grid_config.ny - 1)

    tx = torch.diag(torch.full((nx_int,), 2.0 / dx**2, dtype=torch.float64))
    ty = torch.diag(torch.full((ny_int,), 2.0 / dy**2, dtype=torch.float64))
    tx = tx + torch.diag(torch.full((nx_int - 1,), -1.0 / dx**2, dtype=torch.float64), diagonal=1)
    tx = tx + torch.diag(torch.full((nx_int - 1,), -1.0 / dx**2, dtype=torch.float64), diagonal=-1)
    ty = ty + torch.diag(torch.full((ny_int - 1,), -1.0 / dy**2, dtype=torch.float64), diagonal=1)
    ty = ty + torch.diag(torch.full((ny_int - 1,), -1.0 / dy**2, dtype=torch.float64), diagonal=-1)

    eye_x = torch.eye(nx_int, dtype=torch.float64)
    eye_y = torch.eye(ny_int, dtype=torch.float64)
    return torch.kron(eye_y, tx) + torch.kron(ty, eye_x)


def build_problem(grid_config: AllenCahnCNConfig, alpha: float, power: int) -> dict[str, object]:
    coords = build_allen_cahn_flattened_coordinates(grid_config)
    n_int = (grid_config.nx - 2) * (grid_config.ny - 2)
    interior_points = coords[:n_int]
    boundary_points = coords[n_int:]
    boundary_values = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)
    truth = u_exact(interior_points[:, 0], interior_points[:, 1])
    rhs_values = rhs_f(interior_points[:, 0], interior_points[:, 1], alpha=alpha, power=power)
    matrix = build_full_order_matrix(grid_config)
    return {'coords': coords, 'n_int': n_int, 'interior_points': interior_points, 'boundary_points': boundary_points, 'boundary_values': boundary_values, 'truth': truth, 'rhs_values': rhs_values, 'matrix': matrix}


def build_snapshot_bank(problem: dict[str, object], alpha: float, power: int, num_snapshots: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = problem['coords']
    n_int = problem['n_int']
    phases = torch.linspace(0.25, 2.0, num_snapshots)
    solution_bank = torch.zeros((coords.shape[0], num_snapshots), dtype=torch.float64)
    nonlinear_bank = torch.zeros((n_int, num_snapshots), dtype=torch.float64)
    for index, phase in enumerate(phases):
        values = phase * u_exact(coords[:, 0], coords[:, 1])
        solution_bank[:, index] = values
        nonlinear_bank[:, index] = rhs_f(problem['interior_points'][:, 0], problem['interior_points'][:, 1], alpha=alpha, power=power) - alpha * values[:n_int].pow(power)
    return solution_bank, nonlinear_bank


def solve_full_order(problem: dict[str, object], alpha: float, power: int, max_iter: int = 30) -> torch.Tensor:
    state = torch.zeros(problem['n_int'], dtype=torch.float64)
    eye = torch.eye(problem['n_int'], dtype=torch.float64)
    matrix = problem['matrix']
    rhs_values = problem['rhs_values']
    for _ in range(max_iter):
        residual = matrix.matmul(state) + alpha * state.pow(power) - rhs_values
        jacobian = matrix + alpha * power * torch.diag(state.pow(power - 1))
        delta = torch.linalg.solve(jacobian + 1e-10 * eye, residual)
        state = state - delta
        if torch.linalg.norm(delta) <= 1e-10 * max(1.0, float(torch.linalg.norm(state))):
            break
    return state


def build_factor_pair(theta, dirac_points, derivative_groups, ordering, rho: float, nugget: float, sparse_backend: str = 'auto', sparse_kernel_source: SnapshotKernelSource | None = None):
    start = time.perf_counter()
    dense_factor = dense_precision_factor(theta, nugget=nugget)
    dense_seconds = time.perf_counter() - start
    sparse_input = sparse_kernel_source if sparse_kernel_source is not None else theta
    start = time.perf_counter()
    sparse_factor, _ = sparse_precision_factor(theta=sparse_input, dirac_points=dirac_points, derivative_point_groups=derivative_groups, rho=rho, nugget=nugget, ordering=ordering, backend=sparse_backend)
    sparse_seconds = time.perf_counter() - start
    return {'dense': dense_factor, 'sparse': sparse_factor}, {'dense_seconds': dense_seconds, 'sparse_seconds': sparse_seconds}


def build_factors(problem: dict[str, object], solution_bank: torch.Tensor, nonlinear_bank: torch.Tensor, rho: float, lengthscale: float, sparse_backend: str = 'auto', k_neighbors: int = 3):
    start = time.perf_counter()
    # k_neighbors=3 aligns with the official repo's stated default for PDE
    # problems with derivative measurements (the official NonlinElliptic2d
    # solver this most resembles uses "follow_diracs", but that variant
    # requires every Dirac point to carry the same derivative groups; the
    # boundary points here have none, so it isn't structurally applicable).
    ordering = build_measurement_ordering(problem['coords'], (torch.arange(problem['n_int'], dtype=torch.long),), k_neighbors=k_neighbors)
    ordering_seconds = time.perf_counter() - start
    empirical_theta = build_elliptic_empirical_theta(solution_bank, nonlinear_bank, nugget=1e-10) / solution_bank.shape[1]
    matern_assembly = build_elliptic_matern_theta(problem['interior_points'], problem['boundary_points'], lengthscale=lengthscale, nugget=1e-10)
    empirical_ks = make_snapshot_kernel_source(
        (solution_bank, nonlinear_bank),
        nugget=1e-10,
        scale=1.0 / solution_bank.shape[1],
    )
    empirical_factors, empirical_times = build_factor_pair(empirical_theta, problem['coords'], (torch.arange(problem['n_int'], dtype=torch.long),), ordering, rho=rho, nugget=1e-10, sparse_backend=sparse_backend, sparse_kernel_source=empirical_ks)
    matern_factors, matern_times = build_factor_pair(matern_assembly.theta, matern_assembly.dirac_points, matern_assembly.derivative_point_groups, ordering, rho=rho, nugget=1e-10, sparse_backend=sparse_backend)
    return {
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


def solve_with_factor(problem: dict[str, object], factor: object, alpha: float, power: int, gn_steps: int) -> torch.Tensor:
    result = solve_gauss_newton(
        initial_state=torch.zeros(problem['n_int'], dtype=torch.float64),
        residual_and_jacobian=lambda z: elliptic_residual_operator(z, problem['rhs_values'], problem['boundary_values'], alpha=alpha, power=power),
        factor=factor,
        max_iter=gn_steps,
        damping=1e-8,
        record_history=False,
        linear_solver='auto',
        preconditioner='jacobi',
        direct_residual_and_jacobian=lambda z: elliptic_residual_and_jacobian(z, problem['rhs_values'], problem['boundary_values'], alpha=alpha, power=power),
        direct_threshold=AUTO_DIRECT_THRESHOLD,
    )
    return result.state


def benchmark_full_order(problem: dict[str, object], alpha: float, power: int) -> list[float]:
    return [benchmark_callable(solve_full_order, problem, alpha, power, warmup=0, repeats=1).seconds]


def evaluate_factor(problem: dict[str, object], factor: object, truth: torch.Tensor, alpha: float, power: int, gn_steps: int, full_order_times: list[float]) -> dict[str, object]:
    start = time.perf_counter()
    solution = solve_with_factor(problem, factor, alpha=alpha, power=power, gn_steps=gn_steps)
    runtime = time.perf_counter() - start
    return {
        'mean_rel_l2': relative_l2(solution, truth),
        'mean_krom_seconds': runtime,
        'per_case_seconds': [runtime],
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }


def run_experiment(grid_config: AllenCahnCNConfig = DEFAULT_GRID_CONFIG, rho: float = 4.0, lengthscale: float = 0.30, gn_steps: int = 4, alpha: float = 1.0, power: int = 3, num_snapshots: int = 256, sparse_backend: str = 'auto', k_neighbors: int = 3) -> dict[str, object]:
    problem = build_problem(grid_config, alpha=alpha, power=power)
    start = time.perf_counter()
    solution_bank, nonlinear_bank = build_snapshot_bank(problem, alpha=alpha, power=power, num_snapshots=num_snapshots)
    train_snapshot_seconds = time.perf_counter() - start
    factors = build_factors(problem, solution_bank, nonlinear_bank, rho=rho, lengthscale=lengthscale, sparse_backend=sparse_backend, k_neighbors=k_neighbors)
    full_order_times = benchmark_full_order(problem, alpha=alpha, power=power)
    empirical_sparse = evaluate_factor(problem, factors['empirical_sparse'], problem['truth'], alpha=alpha, power=power, gn_steps=gn_steps, full_order_times=full_order_times)
    empirical_dense = evaluate_factor(problem, factors['empirical_dense'], problem['truth'], alpha=alpha, power=power, gn_steps=gn_steps, full_order_times=full_order_times)
    matern_sparse = evaluate_factor(problem, factors['matern_sparse'], problem['truth'], alpha=alpha, power=power, gn_steps=gn_steps, full_order_times=full_order_times)
    matern_dense = evaluate_factor(problem, factors['matern_dense'], problem['truth'], alpha=alpha, power=power, gn_steps=gn_steps, full_order_times=full_order_times)
    empirical_solution = solve_with_factor(problem, factors['empirical_sparse'], alpha=alpha, power=power, gn_steps=gn_steps)
    matern_solution = solve_with_factor(problem, factors['matern_sparse'], alpha=alpha, power=power, gn_steps=gn_steps)
    return {
        'grid_config': grid_config,
        'problem': problem,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_snapshots': num_snapshots,
        'truth': problem['truth'],
        'empirical_solution': empirical_solution,
        'matern_solution': matern_solution,
        'empirical': empirical_sparse,
        'matern': matern_sparse,
        'empirical_sparse': empirical_sparse,
        'empirical_dense': empirical_dense,
        'matern_sparse': matern_sparse,
        'matern_dense': matern_dense,
        'timings': {
            'train_snapshot_seconds': train_snapshot_seconds,
            'factor_build_seconds': factors['build_times'],
        },
    }


def plot_experiment(result: dict[str, object]):
    problem = result['problem']
    grid_config = result['grid_config']
    nx = grid_config.nx - 2
    ny = grid_config.ny - 2
    x = problem['interior_points'][:, 0].reshape(nx, ny)
    y = problem['interior_points'][:, 1].reshape(nx, ny)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].contourf(x, y, result['truth'].reshape(nx, ny), levels=40)
    axes[0].set_title('Exact solution')
    axes[1].contourf(x, y, result['empirical_solution'].reshape(nx, ny), levels=40)
    axes[1].set_title('Empirical sparse')
    axes[2].contourf(x, y, result['matern_solution'].reshape(nx, ny), levels=40)
    axes[2].set_title('Matern sparse')
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
    axes[3].set_ylabel('Seconds per solve')
    fig.tight_layout()
    return fig, axes


def run_rho_sweep(rho_values: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=rho_values, evaluator=lambda value: run_experiment(rho=float(value), **kwargs), metric_name='mean_rel_l2', parameter_name='rho', ylabel='Relative L2 error', title='Elliptic error vs sparse radius')


def run_train_sweep(snapshot_counts: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=snapshot_counts, evaluator=lambda value: run_experiment(num_snapshots=int(value), **kwargs), metric_name='mean_rel_l2', parameter_name='num_snapshots', ylabel='Relative L2 error', title='Elliptic error vs number of snapshot features')


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
    train_sweep = run_train_sweep([64, 128, 256, 384])
    plot_sweeps(rho_sweep, train_sweep)
    plt.show()
