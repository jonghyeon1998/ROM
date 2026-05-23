from __future__ import annotations

from typing import Iterable

import torch
from scipy.spatial import cKDTree

from .factors import SparseInverseFactor
from .ordering import MeasurementOrdering, build_measurement_ordering


def build_sparsity_pattern(
    ordered_points: torch.Tensor,
    lengthscales: torch.Tensor,
    rho: float,
) -> list[list[int]]:
    ordered_points = ordered_points.detach().cpu().to(dtype=torch.float64)
    lengthscales = lengthscales.detach().cpu().to(dtype=torch.float64)
    tree = cKDTree(ordered_points.numpy())
    pattern: list[list[int]] = []
    for column in range(ordered_points.shape[0]):
        radius = float(rho * lengthscales[column])
        support = [row for row in tree.query_ball_point(ordered_points[column].numpy(), radius) if row <= column]
        support.sort()
        pattern.append(support)
    return pattern


def _normalized_precision_column(
    theta: torch.Tensor,
    support: list[int],
    nugget: float,
) -> torch.Tensor:
    index = torch.tensor(support, dtype=torch.long, device=theta.device)
    block = theta.index_select(0, index).index_select(1, index)
    eye = torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
    jitter = float(nugget)

    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(block + jitter * eye)
        if int(info.item()) == 0:
            rhs = torch.zeros(block.shape[0], dtype=block.dtype, device=block.device)
            rhs[-1] = 1.0
            sol = torch.cholesky_solve(rhs.unsqueeze(-1), chol, upper=False).squeeze(-1)
            pivot = torch.clamp(sol[-1], min=torch.finfo(sol.dtype).eps)
            return sol / torch.sqrt(pivot)
        jitter *= 10.0

    raise torch.linalg.LinAlgError("Sparse Cholesky column block remained indefinite after jitter escalation.")


def sparse_precision_factor(
    theta: torch.Tensor,
    dirac_points: torch.Tensor,
    derivative_point_groups: Iterable[torch.Tensor] | None = None,
    rho: float = 3.0,
    nugget: float = 1e-10,
    ordering: MeasurementOrdering | None = None,
) -> tuple[SparseInverseFactor, MeasurementOrdering]:
    if ordering is None:
        ordering = build_measurement_ordering(dirac_points, derivative_point_groups)

    permutation = ordering.permutation.to(theta.device)
    reordered_theta = theta.index_select(0, permutation).index_select(1, permutation)
    sparsity = build_sparsity_pattern(ordering.ordered_points, ordering.lengthscales, rho)

    row_indices: list[int] = []
    col_indices: list[int] = []
    data_chunks: list[torch.Tensor] = []

    for column, support in enumerate(sparsity):
        values = _normalized_precision_column(reordered_theta, support, nugget)
        row_indices.extend(support)
        col_indices.extend([column] * len(support))
        data_chunks.append(values)

    data = torch.cat(data_chunks, dim=0)
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=theta.device)
    factor = torch.sparse_coo_tensor(indices, data, size=reordered_theta.shape, device=theta.device).coalesce()
    return SparseInverseFactor(factor=factor, permutation=permutation), ordering
