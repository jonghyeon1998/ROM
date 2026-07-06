from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
except Exception:  # pragma: no cover - optional runtime dependency
    cholmod_cholesky = None

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


def _as_numpy_permutation(permutation: torch.Tensor) -> np.ndarray:
    return permutation.detach().cpu().numpy().astype(np.int64, copy=False)


def _batched_normalized_precision_columns(
    theta: torch.Tensor,
    supports: list[list[int]],
    nugget: float,
) -> torch.Tensor:
    """Factor a batch of equal-size column blocks with one vectorized call.

    This is a "supernodal-style" batching of the per-column dense solves in
    ``_normalized_precision_column``: columns whose sparsity support has the
    *same size* are gathered into a single ``(batch, size, size)`` tensor and
    factored together via one batched ``torch.linalg.cholesky_ex`` + one
    batched ``torch.cholesky_solve``, instead of one Python-level call per
    column. This isn't a classical supernode amalgamation (which additionally
    requires matching row *patterns*, not just sizes) — grouping by size is a
    cheap, torch-native proxy that still lets independent small dense
    factorizations run in parallel (especially on GPU) instead of serially
    in a Python loop. The per-column jitter escalation logic of
    ``_normalized_precision_column`` is preserved exactly, just applied
    per-batch-element: any column whose block is still indefinite after
    adding the current jitter is retried (with 10x jitter) in the next round,
    while already-successful columns in the same batch are left alone.

    Returns a ``(batch, size)`` tensor whose row ``b`` equals what
    ``_normalized_precision_column(theta, supports[b], nugget)`` would have
    returned.
    """
    batch = len(supports)
    size = len(supports[0])
    index = torch.tensor(supports, dtype=torch.long, device=theta.device)  # (batch, size)
    block = theta[index.unsqueeze(-1), index.unsqueeze(-2)]  # (batch, size, size); block[b,i,j] = theta[idx[b,i], idx[b,j]]
    eye = torch.eye(size, dtype=block.dtype, device=block.device)

    jitter = torch.full((batch,), float(nugget), dtype=block.dtype, device=block.device)
    result = torch.zeros((batch, size), dtype=block.dtype, device=block.device)
    pending = torch.arange(batch, device=block.device)

    for _ in range(6):
        if pending.numel() == 0:
            break
        sub_block = block.index_select(0, pending)
        sub_jitter = jitter.index_select(0, pending).view(-1, 1, 1)
        chol, info = torch.linalg.cholesky_ex(sub_block + sub_jitter * eye)
        ok = info == 0
        if bool(ok.any()):
            ok_global = pending[ok]
            rhs = torch.zeros((int(ok.sum()), size, 1), dtype=block.dtype, device=block.device)
            rhs[:, -1, 0] = 1.0
            sol = torch.cholesky_solve(rhs, chol[ok], upper=False).squeeze(-1)
            pivot = torch.clamp(sol[:, -1], min=torch.finfo(sol.dtype).eps)
            result[ok_global] = sol / pivot.unsqueeze(-1).sqrt()
        pending = pending[~ok]
        if pending.numel() > 0:
            jitter[pending] = jitter[pending] * 10.0

    if pending.numel() > 0:
        raise torch.linalg.LinAlgError("Sparse Cholesky column block remained indefinite after jitter escalation.")
    return result


def sparse_precision_factor(
    theta: torch.Tensor,
    dirac_points: torch.Tensor,
    derivative_point_groups: Iterable[torch.Tensor] | None = None,
    rho: float = 3.0,
    nugget: float = 1e-10,
    ordering: MeasurementOrdering | None = None,
    backend: str = 'auto',
    k_neighbors: int = 1,
    ordering_variant: str = 'dirac_first_then_unif_scale',
    batch_columns: bool = True,
) -> tuple[SparseInverseFactor, MeasurementOrdering]:
    """Build a sparse Vecchia/KL-minimizing approximate inverse Cholesky factor.

    ``k_neighbors`` and ``ordering_variant`` are forwarded to
    ``build_measurement_ordering`` (and ignored if an explicit ``ordering`` is
    supplied). Matching the official GP-PDEs-SparseCholesky repo's README
    guidance: ``k_neighbors=1`` is the default for a plain point cloud / pure
    Dirac problem; PDE problems with derivative measurements typically use
    ``k_neighbors=3``. ``ordering_variant="follow_diracs"`` requires every
    group in ``derivative_point_groups`` to be co-located 1:1 with
    ``dirac_points``.

    ``batch_columns`` (default ``True``) groups columns that share the same
    sparsity-support size and factors each group with one batched
    ``torch.linalg.cholesky_ex``/``cholesky_solve`` call instead of looping
    over columns one at a time in Python — see
    ``_batched_normalized_precision_columns``. Set to ``False`` to fall back
    to the original strictly-sequential per-column path (useful for
    debugging or exact-call-count comparisons).
    """
    if ordering is None:
        ordering = build_measurement_ordering(
            dirac_points,
            derivative_point_groups,
            k_neighbors=k_neighbors,
            variant=ordering_variant,
        )

    permutation = ordering.permutation.to(theta.device)
    reordered_theta = theta.index_select(0, permutation).index_select(1, permutation)
    sparsity = build_sparsity_pattern(ordering.ordered_points, ordering.lengthscales, rho)

    row_indices: list[int] = []
    col_indices: list[int] = []
    data_chunks: list[torch.Tensor] = []

    if batch_columns:
        size_groups: dict[int, list[int]] = {}
        for column, support in enumerate(sparsity):
            size_groups.setdefault(len(support), []).append(column)

        column_values: list[torch.Tensor | None] = [None] * len(sparsity)
        for size, columns in size_groups.items():
            supports = [sparsity[c] for c in columns]
            batched = _batched_normalized_precision_columns(reordered_theta, supports, nugget)
            for row, column in enumerate(columns):
                column_values[column] = batched[row, :size]
        for column, support in enumerate(sparsity):
            values = column_values[column]
            row_indices.extend(support)
            col_indices.extend([column] * len(support))
            data_chunks.append(values)
    else:
        for column, support in enumerate(sparsity):
            values = _normalized_precision_column(reordered_theta, support, nugget)
            row_indices.extend(support)
            col_indices.extend([column] * len(support))
            data_chunks.append(values)

    data = torch.cat(data_chunks, dim=0).detach().cpu().numpy()
    factor_csc = sp.csc_matrix((data, (row_indices, col_indices)), shape=tuple(reordered_theta.shape))
    permutation_np = _as_numpy_permutation(permutation)

    backend_mode = backend
    if backend_mode == 'auto':
        backend_mode = 'scipy'

    if backend_mode == 'scipy':
        return SparseInverseFactor(factor=factor_csc, permutation=permutation_np, backend_name='scipy_sparse'), ordering

    if backend_mode == 'cholmod':
        if cholmod_cholesky is None:
            raise ImportError('CHOLMOD backend requested but scikit-sparse is not installed.')
        approx_precision = (factor_csc @ factor_csc.transpose()).tocsc()
        cholmod_factor = cholmod_cholesky(approx_precision)
        cholmod_perm = np.asarray(cholmod_factor.P(), dtype=np.int64)
        total_permutation = permutation_np[cholmod_perm]
        return SparseInverseFactor(factor=cholmod_factor.L(), permutation=total_permutation, backend_name='cholmod_sparse'), ordering

    raise ValueError(f'Unsupported sparse backend: {backend}')
