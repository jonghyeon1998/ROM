from __future__ import annotations

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


def maximin(points: torch.Tensor, initial: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    reverse_order, reverse_lengths = _reverse_maximin(points, initial)
    return torch.flip(reverse_order, dims=[0]), torch.flip(reverse_lengths, dims=[0]).to(points.dtype)


def build_measurement_ordering(
    dirac_points: torch.Tensor,
    derivative_point_groups: Iterable[torch.Tensor] | None = None,
) -> MeasurementOrdering:
    dirac_points = _as_points(dirac_points)
    derivative_point_groups = tuple(derivative_point_groups or ())

    dirac_permutation, dirac_lengthscales = maximin(dirac_points)
    total_dirac = dirac_points.shape[0]

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

    inverse_permutation = torch.argsort(permutation)

    return MeasurementOrdering(
        permutation=permutation.to(dirac_points.device),
        inverse_permutation=inverse_permutation.to(dirac_points.device),
        ordered_points=ordered_points.to(dirac_points.device),
        lengthscales=lengthscales.to(dirac_points.device, dtype=dirac_points.dtype),
        dirac_permutation=dirac_permutation.to(dirac_points.device),
        ordered_point_indices=ordered_point_indices.to(dirac_points.device),
    )
