from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass
from typing import Iterable

import torch
from scipy.spatial import cKDTree


@dataclass
class MeasurementOrdering:
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    ordered_points: torch.Tensor
    lengthscales: torch.Tensor
    dirac_permutation: torch.Tensor
    ordered_point_indices: torch.Tensor


def _as_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 1:
        return points.unsqueeze(-1)
    return points


def _reverse_maximin(points: torch.Tensor, initial: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    points = _as_points(points).detach().cpu().to(dtype=torch.float64)
    n_points = points.shape[0]
    order = torch.zeros(n_points, dtype=torch.long)
    lengthscales = torch.zeros(n_points, dtype=points.dtype)

    points_np = points.numpy()
    tree = cKDTree(points_np)

    if initial is None or initial.numel() == 0:
        first_index = 0
        current_distances = torch.linalg.norm(points - points[first_index], dim=1).numpy()
        order[-1] = first_index
        lengthscales[-1] = float("inf")
        active = [True] * n_points
        active[first_index] = False
        write_index = n_points - 2
    else:
        initial = _as_points(initial).detach().cpu().to(dtype=points.dtype)
        current_distances, _ = cKDTree(initial.numpy()).query(points_np)
        active = [True] * n_points
        write_index = n_points - 1

    heap = [(-float(distance), point_index) for point_index, distance in enumerate(current_distances)]
    heapq.heapify(heap)

    while write_index >= 0:
        while True:
            neg_distance, candidate = heapq.heappop(heap)
            distance = -neg_distance
            if active[candidate] and distance >= current_distances[candidate] - 1e-15:
                break

        active[candidate] = False
        order[write_index] = candidate
        lengthscales[write_index] = float(distance)

        if distance > 0.0:
            neighbors = tree.query_ball_point(points_np[candidate], distance)
            local_distances = torch.linalg.norm(points[neighbors] - points[candidate], dim=1).numpy()
            for neighbor, neighbor_distance in zip(neighbors, local_distances):
                if active[neighbor] and neighbor_distance < current_distances[neighbor]:
                    current_distances[neighbor] = float(neighbor_distance)
                    heapq.heappush(heap, (-float(neighbor_distance), neighbor))

        write_index -= 1

    return order, lengthscales


def _reverse_maximin_k(
    points: torch.Tensor,
    k_neighbors: int,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse k-maximin ordering: each point's priority is its distance to the
    *k-th* nearest already-selected point (rather than the nearest), matching
    the ``k_neighbors`` option in the official GP-PDEs-SparseCholesky repo's
    ``kolesky/ordering.py``. ``k_neighbors=1`` reduces to plain reverse-maximin.

    Implementation is two-phase to sidestep the "fewer than k points selected
    so far" bootstrap issue:

    1. Bootstrap: run ordinary (1-)maximin for the first ``k_neighbors``
       selections (or until ``initial`` already supplies >= k_neighbors
       reference points), since "k-th nearest" is undefined with fewer than k
       reference points.
    2. Main phase: once >= k_neighbors points have been selected, each active
       point's priority is the k-th smallest of its (capped, sorted) list of
       distances to selected points. A newly selected candidate has the
       maximum such key among active points; any other active point farther
       than that key from the candidate provably cannot have its k-th
       smallest distance reduced, so neighbor updates are restricted to a
       ball of that radius (same pruning argument as plain maximin, applied
       to the k-th order statistic instead of the minimum).
    """
    if k_neighbors <= 1:
        return _reverse_maximin(points, initial)

    points = _as_points(points).detach().cpu().to(dtype=torch.float64)
    n_points = points.shape[0]
    points_np = points.numpy()
    tree = cKDTree(points_np)

    order = torch.zeros(n_points, dtype=torch.long)
    lengthscales = torch.zeros(n_points, dtype=points.dtype)
    active = [True] * n_points
    dist_lists: list[list[float]] = [[] for _ in range(n_points)]

    if initial is not None and initial.numel() > 0:
        initial = _as_points(initial).detach().cpu().to(dtype=torch.float64)
        kk = min(k_neighbors, initial.shape[0])
        distances, _ = cKDTree(initial.numpy()).query(points_np, k=kk)
        if kk == 1:
            distances = distances.reshape(-1, 1)
        for p in range(n_points):
            dist_lists[p] = sorted(float(d) for d in distances[p])
        write_index = n_points - 1
        selected_count = 0
    else:
        first_index = 0
        order[-1] = first_index
        lengthscales[-1] = float("inf")
        active[first_index] = False
        write_index = n_points - 2
        selected_count = 1
        d0 = torch.linalg.norm(points - points[first_index], dim=1).numpy()
        for p in range(n_points):
            if active[p]:
                dist_lists[p].append(float(d0[p]))

    def key(point_index: int) -> float:
        values = dist_lists[point_index]
        if len(values) < k_neighbors:
            return float("inf")
        return values[k_neighbors - 1]

    # --- bootstrap phase: brute-force update (no radius pruning) until every
    # active point has accumulated k_neighbors reference distances. ---
    while selected_count < k_neighbors and write_index >= 0:
        best_index, best_key = None, -1.0
        for p in range(n_points):
            if active[p] and key(p) > best_key:
                best_key = key(p)
                best_index = p
        candidate = best_index
        active[candidate] = False
        order[write_index] = candidate
        lengthscales[write_index] = best_key
        d_candidate = torch.linalg.norm(points - points[candidate], dim=1).numpy()
        for p in range(n_points):
            if active[p]:
                bisect.insort(dist_lists[p], float(d_candidate[p]))
                if len(dist_lists[p]) > k_neighbors:
                    dist_lists[p].pop()
        write_index -= 1
        selected_count += 1

    # --- main phase: heap + radius-pruned updates, same style as 1-maximin. ---
    heap = [(-key(p), p) for p in range(n_points) if active[p]]
    heapq.heapify(heap)

    while write_index >= 0:
        while True:
            neg_key, candidate = heapq.heappop(heap)
            current_key = -neg_key
            if active[candidate] and current_key >= key(candidate) - 1e-15:
                break

        active[candidate] = False
        order[write_index] = candidate
        lengthscales[write_index] = current_key

        if current_key > 0.0 and current_key < float("inf"):
            neighbors = tree.query_ball_point(points_np[candidate], current_key)
            local_distances = torch.linalg.norm(points[neighbors] - points[candidate], dim=1).numpy()
            for neighbor, neighbor_distance in zip(neighbors, local_distances):
                if not active[neighbor]:
                    continue
                values = dist_lists[neighbor]
                if len(values) < k_neighbors or neighbor_distance < values[-1]:
                    bisect.insort(values, float(neighbor_distance))
                    if len(values) > k_neighbors:
                        values.pop()
                    heapq.heappush(heap, (-key(neighbor), neighbor))

        write_index -= 1

    return order, lengthscales


def maximin(
    points: torch.Tensor,
    initial: torch.Tensor | None = None,
    k_neighbors: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    reverse_order, reverse_lengths = _reverse_maximin_k(points, k_neighbors, initial)
    return torch.flip(reverse_order, dims=[0]), torch.flip(reverse_lengths, dims=[0]).to(points.dtype)


def build_measurement_ordering(
    dirac_points: torch.Tensor,
    derivative_point_groups: Iterable[torch.Tensor] | None = None,
    k_neighbors: int = 1,
    variant: str = "dirac_first_then_unif_scale",
) -> MeasurementOrdering:
    """Build a maximin ordering over a Dirac (point-evaluation) measurement
    set plus zero or more co-located derivative-measurement groups.

    ``k_neighbors`` selects 1-maximin (default, ``k_neighbors=1``) or
    k-maximin (``k_neighbors>1``), matching the official GP-PDEs-SparseCholesky
    repo's ``maximin_ordering(..., k_neighbors=...)``.

    ``variant`` selects between the two ordering patterns described in that
    repo's README:

    * ``"dirac_first_then_unif_scale"`` (default, original behavior here):
      order all Dirac points first, then append every derivative-measurement
      group afterwards, all sharing the single finest Dirac lengthscale.
      This is the variant the official paper's theory is stated for.
    * ``"follow_diracs"``: insert each derivative measurement immediately
      after its co-located Dirac point (so they land in the same supernode),
      using that point's own per-step lengthscale rather than one global
      finest scale. Requires every group in ``derivative_point_groups`` to be
      co-located 1:1 with ``dirac_points`` (i.e. ``group == arange(N)``).
    """
    dirac_points = _as_points(dirac_points)
    derivative_point_groups = tuple(derivative_point_groups or ())

    dirac_permutation, dirac_lengthscales = maximin(dirac_points, k_neighbors=k_neighbors)
    total_dirac = dirac_points.shape[0]

    if variant == "dirac_first_then_unif_scale":
        permutation_parts = [dirac_permutation]
        point_index_parts = [dirac_permutation]
        next_offset = total_dirac

        for group in derivative_point_groups:
            group = torch.as_tensor(group, dtype=torch.long)
            permutation_parts.append(torch.arange(next_offset, next_offset + group.numel(), dtype=torch.long))
            point_index_parts.append(group)
            next_offset += group.numel()

        permutation = torch.cat(permutation_parts, dim=0)
        ordered_point_indices = torch.cat(point_index_parts, dim=0)
        ordered_points = dirac_points.index_select(0, ordered_point_indices.to(dirac_points.device))

        total_derivatives = ordered_point_indices.numel() - total_dirac
        if total_derivatives > 0:
            derivative_length = dirac_lengthscales[-1].expand(total_derivatives)
            lengthscales = torch.cat([dirac_lengthscales, derivative_length.to(dirac_lengthscales.dtype)], dim=0)
        else:
            lengthscales = dirac_lengthscales

    elif variant == "follow_diracs":
        if not derivative_point_groups:
            raise ValueError("follow_diracs variant requires at least one derivative_point_group.")
        identity = torch.arange(total_dirac, dtype=torch.long)
        for group in derivative_point_groups:
            group = torch.as_tensor(group, dtype=torch.long)
            if group.numel() != total_dirac or not torch.equal(group, identity):
                raise ValueError(
                    "follow_diracs variant requires every derivative_point_group to be co-located "
                    "1:1 with dirac_points (group == arange(len(dirac_points)))."
                )

        n_groups = len(derivative_point_groups)
        block = n_groups + 1
        total = total_dirac * block

        ordered_point_indices = torch.empty(total, dtype=torch.long)
        permutation = torch.empty(total, dtype=torch.long)
        lengthscales = torch.empty(total, dtype=dirac_lengthscales.dtype)

        slot_index = torch.arange(block, dtype=torch.long)
        for i in range(total_dirac):
            p = int(dirac_permutation[i])
            base = i * block
            ordered_point_indices[base : base + block] = p
            # slot 0 = the dirac block [0, total_dirac); slot g (1-indexed) =
            # derivative group g-1's block [g*total_dirac, (g+1)*total_dirac).
            permutation[base : base + block] = slot_index * total_dirac + p
            lengthscales[base : base + block] = dirac_lengthscales[i]

        ordered_points = dirac_points.index_select(0, ordered_point_indices.to(dirac_points.device))

    else:
        raise ValueError(f"Unknown ordering variant: {variant!r}")

    inverse_permutation = torch.argsort(permutation)

    return MeasurementOrdering(
        permutation=permutation.to(dirac_points.device),
        inverse_permutation=inverse_permutation.to(dirac_points.device),
        ordered_points=ordered_points.to(dirac_points.device),
        lengthscales=lengthscales.to(dirac_points.device, dtype=dirac_points.dtype),
        dirac_permutation=dirac_permutation.to(dirac_points.device),
        ordered_point_indices=ordered_point_indices.to(dirac_points.device),
    )
