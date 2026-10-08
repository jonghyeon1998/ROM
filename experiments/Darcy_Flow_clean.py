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
from src.krom.workflows import build_elliptic_empirical_theta, build_elliptic_matern_theta, darcy_residual_and_jacobian, darcy_residual_operator


torch.set_default_dtype(torch.float64)

DEFAULT_GRID_SIZE = 25
AUTO_DIRECT_THRESHOLD = 2048


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth).clamp_min(torch.finfo(truth.dtype).eps))


def a_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    term1 = (1.1 + torch.sin(10.0 * torch.pi * x)) / (1.1 + torch.sin(10.0 * torch.pi * y))
    term2 = (1.1 + torch.sin(26.0 * torch.pi * y)) / (1.1 + torch.cos(26.0 * torch.pi * x))
    term3 = (1.1 + torch.cos(34.0 * torch.pi * x)) / (1.1 + torch.sin(34.0 * torch.pi * y))
    term4 = (1.1 + torch.sin(62.0 * torch.pi * y)) / (1.1 + torch.cos(62.0 * torch.pi * x))
    term5 = (1.1 + torch.cos(130.0 * torch.pi * x)) / (1.1 + torch.sin(130.0 * torch.pi * y))
    term6 = torch.sin(4.0 * x.square() * y.square())
    return (term1 + term2 + term3 + term4 + term5 + term6 + 1.0) / 6.0


def sample_smooth_forcing(n: int, num_modes: int = 4) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n)
    y = torch.linspace(0.0, 1.0, n)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    field = torch.zeros_like(x_grid)
    coeffs = torch.randn((num_modes, num_modes), dtype=torch.float64)
    for i in range(1, num_modes + 1):
        for j in range(1, num_modes + 1):
            field = field + coeffs[i - 1, j - 1] * torch.sin(i * torch.pi * x_grid) * torch.sin(j * torch.pi * y_grid)
    scale = torch.clamp(field.abs().max(), min=torch.finfo(field.dtype).eps)
    return (field / scale)[1:-1, 1:-1].reshape(-1)


def build_darcy_problem(n: int) -> dict[str, object]:
    domain = AllenCahnCNConfig(nx=n, ny=n, dt=1.0, tmax=1.0, domain_length=1.0)
    coords = build_allen_cahn_flattened_coordinates(domain)
    n_int = (n - 2) * (n - 2)
    interior_points = coords[:n_int]
    boundary_points = coords[n_int:]
    boundary_values = torch.zeros(coords.shape[0] - n_int, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, n)
    y = torch.linspace(0.0, 1.0, n)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    a_grid = a_coefficient(x_grid, y_grid)
    h = x[1] - x[0]
    matrix = torch.zeros((n_int, n_int), dtype=torch.float64)

    def idx(i: int, j: int) -> int:
        return i * (n - 2) + j

    for i in range(n - 2):
        for j in range(n - 2):
            ii = i + 1
            jj = j + 1
            a_e = 0.5 * (a_grid[ii, jj] + a_grid[ii + 1, jj])
            a_w = 0.5 * (a_grid[ii, jj] + a_grid[ii - 1, jj])
            a_n = 0.5 * (a_grid[ii, jj] + a_grid[ii, jj + 1])
            a_s = 0.5 * (a_grid[ii, jj] + a_grid[ii, jj - 1])
            center = idx(i, j)
            matrix[center, center] = (a_e + a_w + a_n + a_s) / h**2
            if i < n - 3:
                matrix[center, idx(i + 1, j)] = -a_e / h**2
            if i > 0:
                matrix[center, idx(i - 1, j)] = -a_w / h**2
            if j < n - 3:
                matrix[center, idx(i, j + 1)] = -a_n / h**2
            if j > 0:
                matrix[center, idx(i, j - 1)] = -a_s / h**2
    return {'coords': coords, 'n_int': n_int, 'interior_points': interior_points, 'boundary_points': boundary_points, 'boundary_values': boundary_values, 'matrix': matrix, 'grid_size': n}


def solve_full_order(matrix: torch.Tensor, rhs_values: torch.Tensor, max_iter: int = 30) -> torch.Tensor:
    n_int = rhs_values.numel()
    state = torch.zeros(n_int, dtype=torch.float64)
    eye = torch.eye(n_int, dtype=torch.float64)
    for _ in range(max_iter):
        residual = matrix.matmul(state) + state.pow(3) - rhs_values
        jacobian = matrix + 3.0 * torch.diag(state.square())
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


def build_factors(problem: dict[str, object], train_solutions: torch.Tensor, train_rhs: torch.Tensor, rho: float, lengthscale: float, sparse_backend: str = 'auto', k_neighbors: int = 3):
    num_train = train_solutions.shape[1]
    train_solution_bank = torch.cat([train_solutions, torch.zeros(problem['boundary_points'].shape[0], num_train)], dim=0)
    train_nonlinear_bank = train_rhs - train_solutions.pow(3)
    start = time.perf_counter()
    # k_neighbors=3 aligns with the official repo's stated default for PDE
    # problems with derivative measurements. Boundary points have no
    # co-located derivative measurement here, so this stays on the default
    # "dirac_first_then_unif_scale" variant (the one the paper's theory
    # covers, and the closer analog to this elliptic problem's official
    # counterpart).
    ordering = build_measurement_ordering(problem['coords'], (torch.arange(problem['n_int'], dtype=torch.long),), k_neighbors=k_neighbors)
    ordering_seconds = time.perf_counter() - start
    empirical_theta = build_elliptic_empirical_theta(train_solution_bank, train_nonlinear_bank, nugget=1e-10) / num_train
    matern_assembly = build_elliptic_matern_theta(problem['interior_points'], problem['boundary_points'], lengthscale=lengthscale, nugget=1e-10)
    empirical_ks = make_snapshot_kernel_source(
        (train_solution_bank, train_nonlinear_bank),
        nugget=1e-10,
        scale=1.0 / num_train,
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


def solve_krom(problem: dict[str, object], rhs_values: torch.Tensor, factor: object, gn_steps: int, preconditioner_mode: str | None = None) -> torch.Tensor:
    if preconditioner_mode == 'weighted_jacobi':
        residual_model = lambda z: darcy_residual_and_jacobian(z, rhs_values, problem['boundary_values'])
    else:
        residual_model = lambda z: darcy_residual_operator(z, rhs_values, problem['boundary_values'])
    result = solve_gauss_newton(
        initial_state=torch.zeros(problem['n_int'], dtype=torch.float64),
        residual_and_jacobian=residual_model,
        factor=factor,
        max_iter=gn_steps,
        damping=1e-8,
        record_history=False,
        linear_solver='auto',
        preconditioner=preconditioner_mode,
        direct_residual_and_jacobian=lambda z: darcy_residual_and_jacobian(z, rhs_values, problem['boundary_values']),
        direct_threshold=AUTO_DIRECT_THRESHOLD,
    )
    return result.state


def benchmark_full_order(problem: dict[str, object], test_rhs: torch.Tensor) -> list[float]:
    return [benchmark_callable(solve_full_order, problem['matrix'], test_rhs[:, sample], warmup=0, repeats=1).seconds for sample in range(test_rhs.shape[1])]


def evaluate_factor(problem: dict[str, object], factor: object, test_rhs: torch.Tensor, test_solutions: torch.Tensor, gn_steps: int, full_order_times: list[float], preconditioner_mode: str | None = None) -> dict[str, object]:
    errors = []
    krom_times = []
    for sample in range(test_rhs.shape[1]):
        start = time.perf_counter()
        prediction = solve_krom(problem, test_rhs[:, sample], factor, gn_steps, preconditioner_mode=preconditioner_mode)
        krom_times.append(time.perf_counter() - start)
        errors.append(relative_l2(prediction, test_solutions[:, sample]))
    return {
        'mean_rel_l2': sum(errors) / len(errors),
        'mean_krom_seconds': sum(krom_times) / len(krom_times),
        'per_case_seconds': krom_times,
        'mean_full_order_seconds': sum(full_order_times) / len(full_order_times),
        'full_order_seconds': full_order_times,
    }


def run_experiment(grid_size: int = DEFAULT_GRID_SIZE, rho: float = 4.0, lengthscale: float = 0.30, gn_steps: int = 3, num_train: int = 12, num_test: int = 10, train_seed: int = 0, test_seed: int = 1, sparse_backend: str = 'auto', preconditioner_mode: str | None = None, k_neighbors: int = 3) -> dict[str, object]:
    problem = build_darcy_problem(grid_size)
    torch.manual_seed(train_seed)
    start = time.perf_counter()
    train_rhs = torch.stack([sample_smooth_forcing(grid_size) for _ in range(num_train)], dim=1)
    train_solutions = torch.stack([solve_full_order(problem['matrix'], train_rhs[:, index]) for index in range(num_train)], dim=1)
    train_snapshot_seconds = time.perf_counter() - start
    torch.manual_seed(test_seed)
    start = time.perf_counter()
    test_rhs = torch.stack([sample_smooth_forcing(grid_size) for _ in range(num_test)], dim=1)
    test_solutions = torch.stack([solve_full_order(problem['matrix'], test_rhs[:, index]) for index in range(num_test)], dim=1)
    test_snapshot_seconds = time.perf_counter() - start
    factors = build_factors(problem, train_solutions, train_rhs, rho=rho, lengthscale=lengthscale, sparse_backend=sparse_backend, k_neighbors=k_neighbors)
    full_order_times = benchmark_full_order(problem, test_rhs)
    empirical_sparse = evaluate_factor(problem, factors['empirical_sparse'], test_rhs, test_solutions, gn_steps, full_order_times, preconditioner_mode=preconditioner_mode)
    empirical_dense = evaluate_factor(problem, factors['empirical_dense'], test_rhs, test_solutions, gn_steps, full_order_times, preconditioner_mode=preconditioner_mode)
    matern_sparse = evaluate_factor(problem, factors['matern_sparse'], test_rhs, test_solutions, gn_steps, full_order_times, preconditioner_mode=preconditioner_mode)
    matern_dense = evaluate_factor(problem, factors['matern_dense'], test_rhs, test_solutions, gn_steps, full_order_times, preconditioner_mode=preconditioner_mode)
    sample_index = 0
    truth = test_solutions[:, sample_index].reshape(grid_size - 2, grid_size - 2)
    empirical_field = solve_krom(problem, test_rhs[:, sample_index], factors['empirical_sparse'], gn_steps, preconditioner_mode=preconditioner_mode).reshape(grid_size - 2, grid_size - 2)
    matern_field = solve_krom(problem, test_rhs[:, sample_index], factors['matern_sparse'], gn_steps, preconditioner_mode=preconditioner_mode).reshape(grid_size - 2, grid_size - 2)
    return {
        'problem': problem,
        'rho': rho,
        'lengthscale': lengthscale,
        'num_train': num_train,
        'num_test': num_test,
        'truth': truth,
        'empirical_field': empirical_field,
        'matern_field': matern_field,
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
    problem = result['problem']
    grid_size = problem['grid_size']
    x = problem['interior_points'][:, 0].reshape(grid_size - 2, grid_size - 2)
    y = problem['interior_points'][:, 1].reshape(grid_size - 2, grid_size - 2)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].contourf(x, y, result['truth'], levels=40)
    axes[0].set_title('Full-order Darcy')
    axes[1].contourf(x, y, result['empirical_field'], levels=40)
    axes[1].set_title('Empirical sparse')
    axes[2].contourf(x, y, result['matern_field'], levels=40)
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
    return run_comparison_sweep(parameter_values=rho_values, evaluator=lambda value: run_experiment(rho=float(value), **kwargs), metric_name='mean_rel_l2', parameter_name='rho', ylabel='Relative L2 error', title='Darcy error vs sparse radius')


def run_train_sweep(train_sizes: Sequence[float], **kwargs) -> object:
    return run_comparison_sweep(parameter_values=train_sizes, evaluator=lambda value: run_experiment(num_train=int(value), **kwargs), metric_name='mean_rel_l2', parameter_name='num_train', ylabel='Relative L2 error', title='Darcy error vs number of training solutions')


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
    train_sweep = run_train_sweep([8, 12, 16])
    plot_sweeps(rho_sweep, train_sweep)
    plt.show()
