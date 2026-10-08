from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

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

# ---------------------------------------------------------------------------
# Lazy kernel source: avoids forming the full N×N Gram matrix
# ---------------------------------------------------------------------------


@dataclass
class SnapshotKernelSource:
    """Lazy empirical kernel: K[i,j] = scale * snapshots[i] · snapshots[j] + nugget_weights[i] * δ(i,j).

    This allows ``sparse_precision_factor`` to compute only the O(N·ρ^d)
    kernel entries it actually needs, never forming the O(N²) Gram matrix.
    The ``snapshots`` tensor is the stacked feature matrix (N_total × M)
    obtained by concatenating per-measurement-type feature matrices vertically.
    ``scale`` is typically 1/M (normalise by number of snapshots/timesteps).
    ``nugget_weights`` is a 1-D tensor of length N_total with the per-index
    diagonal nugget contribution, computed to reproduce the block-trace-ratio
    weighting used by ``_block_nugget`` in ``workflows.py``.

    Use ``make_snapshot_kernel_source`` to construct this correctly from raw
    feature blocks.
    """

    snapshots: torch.Tensor        # (N_total, M)
    scale: float                   # multiply after dot-product (e.g. 1/M)
    nugget_weights: torch.Tensor   # (N_total,)

    @property
    def device(self) -> torch.device:
        return self.snapshots.device

    @property
    def n_total(self) -> int:
        return self.snapshots.shape[0]

    def get_block(self, index: torch.Tensor) -> torch.Tensor:
        """Return a batched kernel block.

        Parameters
        ----------
        index : Tensor of shape (batch, size) — long indices into the (already
            permutation-reordered) snapshot rows.

        Returns
        -------
        Tensor of shape (batch, size, size): K[index[b, i], index[b, j]] for
        all b, i, j — with the nugget already on the diagonal.
        """
        sub = self.snapshots[index]                           # (batch, size, M)
        block = (sub @ sub.transpose(-1, -2)) * self.scale   # (batch, size, size)
        nw = self.nugget_weights[index]                       # (batch, size)
        return block + torch.diag_embed(nw)

    def get_block_single(self, index: torch.Tensor) -> torch.Tensor:
        """Return a single (size, size) kernel block for the sequential path.

        Parameters
        ----------
        index : 1-D long Tensor of length ``size``.
        """
        sub = self.snapshots[index]                           # (size, M)
        block = (sub @ sub.transpose(0, 1)) * self.scale     # (size, size)
        nw = self.nugget_weights[index]                       # (size,)
        return block + torch.diag(nw)


def make_snapshot_kernel_source(
    feature_blocks: tuple[torch.Tensor, ...],
    nugget: float,
    scale: float = 1.0,
) -> SnapshotKernelSource:
    """Build a :class:`SnapshotKernelSource` from raw feature matrices.

    The resulting source is numerically equivalent to assembling the full N×N
    matrix via::

        theta = _block_nugget(
            mixed_empirical_kernel(feature_blocks),
            block_sizes,
            nugget,
        ) * scale

    without ever allocating that N×N matrix.

    Parameters
    ----------
    feature_blocks
        Tuple of 2-D tensors, each of shape ``(N_k, M)`` — one per
        measurement type (e.g. solution values, x-derivatives, Laplacians).
        All must live on the same device and have the same ``M``.
    nugget
        Structural nugget value, matching the ``nugget`` argument passed to
        ``sparse_precision_factor``.
    scale
        Scalar multiplier applied after the dot product. Typically ``1/M``
        where ``M = feature_blocks[0].shape[1]``.
    """
    snapshots = torch.cat(feature_blocks, dim=0)   # (N_total, M)

    # Reproduce the block-trace-ratio weighting from workflows._block_nugget.
    # trace(block_k * scale) = scale * ||U_k||_F^2
    norms_sq = [float((f * f).sum()) for f in feature_blocks]
    baseline = max(norms_sq[0], float(torch.finfo(snapshots.dtype).eps))

    # nugget_weights[i ∈ block_k] = scale * nugget * norms_sq[k] / baseline
    # matches: scale * _block_nugget(U@U^T, block_sizes, nugget) diagonal entry
    weights_list: list[float] = []
    for norm_sq, f in zip(norms_sq, feature_blocks):
        w = scale * nugget * norm_sq / baseline
        weights_list.extend([w] * f.shape[0])

    nugget_weights = snapshots.new_tensor(weights_list)
    return SnapshotKernelSource(
        snapshots=snapshots,
        scale=scale,
        nugget_weights=nugget_weights,
    )


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

KernelSource = Union[torch.Tensor, SnapshotKernelSource]


# ---------------------------------------------------------------------------
# Sparsity pattern
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-column solvers
# ---------------------------------------------------------------------------


def _normalized_precision_column(
    kernel_source: KernelSource,
    support: list[int],
    nugget: float,
) -> torch.Tensor:
    """Compute one column of the sparse Cholesky factor.

    Accepts either a pre-computed ``theta`` tensor (original path) or a
    :class:`SnapshotKernelSource` (lazy path).  The ``nugget`` is used as the
    initial adaptive jitter; for ``SnapshotKernelSource`` the structural
    nugget is already embedded in the block via ``nugget_weights``.
    """
    index = torch.tensor(support, dtype=torch.long)

    if isinstance(kernel_source, SnapshotKernelSource):
        index = index.to(kernel_source.device)
        block = kernel_source.get_block_single(index)      # (size, size)
    else:
        index = index.to(kernel_source.device)
        block = kernel_source.index_select(0, index).index_select(1, index)

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
    kernel_source: KernelSource,
    supports: list[list[int]],
    nugget: float,
) -> torch.Tensor:
    """Factor a batch of equal-size column blocks with one vectorised call.

    This is a "supernodal-style" batching of the per-column dense solves:
    columns whose sparsity support has the *same size* are gathered into a
    single ``(batch, size, size)`` tensor and factored together with one
    batched ``torch.linalg.cholesky_ex`` + ``torch.cholesky_solve`` call.
    Per-column adaptive jitter escalation is preserved exactly per-element.

    Accepts either a pre-computed ``theta`` tensor (original path) or a
    :class:`SnapshotKernelSource` (lazy path — no N×N matrix needed).

    Returns a ``(batch, size)`` tensor; row ``b`` equals what
    ``_normalized_precision_column(kernel_source, supports[b], nugget)``
    would have returned.
    """
    batch = len(supports)
    size = len(supports[0])

    if isinstance(kernel_source, SnapshotKernelSource):
        device = kernel_source.device
    else:
        device = kernel_source.device

    index = torch.tensor(supports, dtype=torch.long, device=device)  # (batch, size)

    if isinstance(kernel_source, SnapshotKernelSource):
        block = kernel_source.get_block(index)            # (batch, size, size) — includes nugget
    else:
        block = kernel_source[index.unsqueeze(-1), index.unsqueeze(-2)]  # (batch, size, size)

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sparse_precision_factor(
    theta: KernelSource,
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
    """Build a sparse Vecchia/KL-minimising approximate inverse Cholesky factor.

    Parameters
    ----------
    theta
        Kernel source.  Either a pre-computed ``(N, N)`` kernel matrix
        (original interface, both Matérn and empirical) or a
        :class:`SnapshotKernelSource` (lazy empirical path — never forms the
        N×N Gram matrix; O(N·ρ^d·M) block assembly instead of O(N²)).
        Use :func:`make_snapshot_kernel_source` to build the latter from raw
        feature matrices.
    dirac_points, derivative_point_groups, rho, nugget, ordering
        As before.  ``ordering`` is ignored when one is supplied; otherwise
        built from ``k_neighbors`` / ``ordering_variant``.
    backend, k_neighbors, ordering_variant, batch_columns
        As before.

    ``batch_columns`` (default ``True``) groups columns that share the same
    sparsity-support size and factors each group with one batched
    ``torch.linalg.cholesky_ex``/``cholesky_solve`` call.  Set to ``False``
    to fall back to the strictly-sequential per-column path.
    """
    if ordering is None:
        ordering = build_measurement_ordering(
            dirac_points,
            derivative_point_groups,
            k_neighbors=k_neighbors,
            variant=ordering_variant,
        )

    permutation = ordering.permutation

    if isinstance(theta, SnapshotKernelSource):
        # Lazy path: permute snapshot rows and nugget weights (O(N·M)), never
        # form the O(N²) reordered_theta.
        perm_cpu = permutation.cpu()
        kernel_input: KernelSource = SnapshotKernelSource(
            snapshots=theta.snapshots[perm_cpu],
            scale=theta.scale,
            nugget_weights=theta.nugget_weights[perm_cpu],
        )
        n_total = theta.n_total
    else:
        # Original path: permute the pre-computed matrix.
        permutation = permutation.to(theta.device)
        reordered_theta = theta.index_select(0, permutation).index_select(1, permutation)
        kernel_input = reordered_theta
        n_total = theta.shape[0]

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
            batched = _batched_normalized_precision_columns(kernel_input, supports, nugget)
            for row, column in enumerate(columns):
                column_values[column] = batched[row, :size]
        for column, support in enumerate(sparsity):
            values = column_values[column]
            row_indices.extend(support)
            col_indices.extend([column] * len(support))
            data_chunks.append(values)
    else:
        for column, support in enumerate(sparsity):
            values = _normalized_precision_column(kernel_input, support, nugget)
            row_indices.extend(support)
            col_indices.extend([column] * len(support))
            data_chunks.append(values)

    data = torch.cat(data_chunks, dim=0).detach().cpu().numpy()
    factor_csc = sp.csc_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_total, n_total),
    )
    permutation_np = _as_numpy_permutation(ordering.permutation)

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
