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
from src.krom.ordering import build_measurement_ordering
from src.krom.pde_baselines import (
    BurgersCNConfig,
    burgers_crank_nicolson_dataset,
    burgers_crank_nicolson_rollout,
)
from src.krom.sparse_cholesky import sparse_precision_factor
from src.krom.workflows import (
    build_burgers_empirical_theta,
    build_burgers_matern_theta,
    burgers_residual_and_jacobian,
)
from src.krom.empirical import temporal_feature_matrix

torch.set_default_dtype(torch.float64)


config = BurgersCNConfig(nx=101, dt=0.02, tmax=1.0, viscosity=1e-3, newton_max_iter=20)
rho = 4.0
lengthscale = 0.30
gn_steps = 3
gn_damping = 1e-8
num_train = 10
num_test = 12

print("Burgers config:", asdict(config))


# %%
# Full-order snapshots via Crank-Nicolson, using the same time step as the KROM rollout.
x, times, train_u, train_ux, train_uxx = burgers_crank_nicolson_dataset(num_train, config, num_terms=4)
_, _, test_u, test_ux, test_uxx = burgers_crank_nicolson_dataset(num_test, config, num_terms=4)

n_int = config.nx - 2
interior_points = x[1:-1, None]
boundary_points = torch.tensor([[x[0]], [x[-1]]], dtype=x.dtype)
boundary_values = torch.zeros(boundary_points.shape[0], dtype=x.dtype)
dirac_points = torch.cat([interior_points, boundary_points], dim=0)
derivative_groups = (torch.arange(n_int, dtype=torch.long), torch.arange(n_int, dtype=torch.long))


# %%
# Empirical sparse factor
dirac_snapshots = torch.cat([train_u[:, 1:-1, :], train_u[:, [0, -1], :]], dim=1)
dirac_features = temporal_feature_matrix(dirac_snapshots)
ux_features = temporal_feature_matrix(train_ux[:, 1:-1, :])
uxx_features = temporal_feature_matrix(train_uxx[:, 1:-1, :])

theta_empirical = build_burgers_empirical_theta(
    dirac_features,
    ux_features,
    uxx_features,
    nugget=1e-9,
) / dirac_snapshots.shape[0]

empirical_factor, empirical_ordering = sparse_precision_factor(
    theta=theta_empirical,
    dirac_points=dirac_points,
    derivative_point_groups=derivative_groups,
    rho=rho,
    nugget=1e-9,
)

print("Empirical factor nnz:", empirical_factor.nnz)
print("Empirical permutation size:", empirical_ordering.permutation.shape[0])


# %%
# Matérn sparse factor
matern_assembly = build_burgers_matern_theta(
    interior_points=interior_points,
    boundary_points=boundary_points,
    lengthscale=lengthscale,
    nugget=1e-9,
)

matern_factor, matern_ordering = sparse_precision_factor(
    theta=matern_assembly.theta,
    dirac_points=matern_assembly.dirac_points,
    derivative_point_groups=matern_assembly.derivative_point_groups,
    rho=rho,
    nugget=1e-9,
)

print("Matern factor nnz:", matern_factor.nnz)
print("Matern permutation size:", matern_ordering.permutation.shape[0])


# %%
def rollout_burgers_krom(
    initial_u: torch.Tensor,
    initial_ux: torch.Tensor,
    initial_uxx: torch.Tensor,
    factor,
    gn_step_count: int = gn_steps,
) -> torch.Tensor:
    interior_rollout = torch.zeros((n_int, times.numel()), dtype=initial_u.dtype)
    old_u = initial_u[1:-1].clone()
    old_ux = initial_ux[1:-1].clone()
    old_uxx = initial_uxx[1:-1].clone()
    interior_rollout[:, 0] = old_u

    state = torch.cat([old_u, old_ux], dim=0)
    for step in range(times.numel() - 1):
        previous_u = old_u.clone()
        previous_ux = old_ux.clone()
        previous_uxx = old_uxx.clone()

        result = solve_gauss_newton(
            initial_state=state,
            residual_and_jacobian=lambda z: burgers_residual_and_jacobian(
                z,
                previous_u,
                previous_ux,
                previous_uxx,
                boundary_values,
                config.viscosity,
                config.dt,
            ),
            factor=factor,
            max_iter=gn_step_count,
            damping=gn_damping,
        )
        state = result.state
        old_u = state[:n_int].clone()
        old_ux = state[n_int:].clone()
        old_uxx = 2.0 / config.viscosity * (
            (old_u - previous_u) / config.dt + 0.5 * (old_u * old_ux + previous_u * previous_ux)
        ) - previous_uxx
        interior_rollout[:, step + 1] = old_u

    return interior_rollout


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))


def evaluate_factor(name: str, factor) -> dict[str, float]:
    final_errors = []
    krom_times = []
    full_order_times = []

    for sample in range(num_test):
        full_order_result = benchmark_callable(
            burgers_crank_nicolson_rollout,
            test_u[sample, :, 0],
            config,
            warmup=0,
            repeats=1,
        )
        full_order_times.append(full_order_result.seconds)

        start = time.perf_counter()
        rollout = rollout_burgers_krom(test_u[sample, :, 0], test_ux[sample, :, 0], test_uxx[sample, :, 0], factor)
        krom_times.append(time.perf_counter() - start)
        final_errors.append(relative_l2(rollout[:, -1], test_u[sample, 1:-1, -1]))

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
empirical_rollout = rollout_burgers_krom(test_u[sample_index, :, 0], test_ux[sample_index, :, 0], test_uxx[sample_index, :, 0], empirical_factor)
matern_rollout = rollout_burgers_krom(test_u[sample_index, :, 0], test_ux[sample_index, :, 0], test_uxx[sample_index, :, 0], matern_factor)
truth_rollout = test_u[sample_index, 1:-1, :]

plt.figure(figsize=(10, 4))
plt.plot(x[1:-1], truth_rollout[:, -1], linewidth=2)
plt.plot(x[1:-1], empirical_rollout[:, -1], linestyle="--")
plt.plot(x[1:-1], matern_rollout[:, -1], linestyle=":")
plt.title("Burgers final-time comparison")
plt.xlabel("x")
plt.ylabel("u(x, t_final)")
plt.legend(["Full-order CN", "Empirical KROM", "Matern KROM"])
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.bar(
    ["Empirical", "Matern"],
    [empirical_summary["mean_krom_seconds"], matern_summary["mean_krom_seconds"]],
)
plt.axhline(empirical_summary["mean_full_order_seconds"], color="black", linestyle="--")
plt.ylabel("Seconds per rollout")
plt.title("KROM vs full-order runtime")
plt.tight_layout()
