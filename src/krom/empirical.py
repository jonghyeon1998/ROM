from __future__ import annotations

from typing import Iterable

import torch


def linear_empirical_kernel(
    features_x: torch.Tensor,
    features_y: torch.Tensor | None = None,
    normalize_by: float | None = None,
) -> torch.Tensor:
    features_y = features_x if features_y is None else features_y
    kernel = features_x.matmul(features_y.transpose(0, 1))
    if normalize_by is not None:
        kernel = kernel / normalize_by
    return kernel


def mixed_empirical_kernel(
    feature_blocks: Iterable[torch.Tensor],
    normalize_by: float | None = None,
) -> torch.Tensor:
    feature_blocks = tuple(feature_blocks)
    rows = []
    for left in feature_blocks:
        row_blocks = [linear_empirical_kernel(left, right, normalize_by) for right in feature_blocks]
        rows.append(torch.cat(row_blocks, dim=1))
    return torch.cat(rows, dim=0)


def temporal_feature_matrix(
    trajectories: torch.Tensor,
    burnin: int = 0,
    stride: int = 1,
    temporal_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = trajectories[..., burnin::stride]
    if temporal_weights is not None:
        weights = temporal_weights.to(device=selected.device, dtype=selected.dtype)
        selected = selected * weights.view(*((1,) * (selected.ndim - 1)), -1)
    return selected.permute(1, 0, 2).reshape(selected.shape[1], -1)


def frobenius_rbf_kernel(trajectories: torch.Tensor, sigma: float) -> torch.Tensor:
    n_samples, n_space, n_time = trajectories.shape
    flattened = trajectories.reshape(n_samples, n_space * n_time)
    sq_norms = flattened.square().sum(dim=1, keepdim=True)
    distances = torch.clamp(sq_norms + sq_norms.transpose(0, 1) - 2.0 * flattened.matmul(flattened.transpose(0, 1)), min=0.0)
    return torch.exp(-distances / (2.0 * n_space * n_time * sigma * sigma))


def polynomial_empirical_kernel(
    features: torch.Tensor,
    degree: int = 2,
    bias: float = 1.0,
) -> torch.Tensor:
    return (features.matmul(features.transpose(0, 1)) + bias) ** degree


def additive_kernel(
    linear_kernel: torch.Tensor,
    nonlinear_kernel: torch.Tensor,
    linear_weight: float = 1.0,
    nonlinear_weight: float = 1.0,
) -> torch.Tensor:
    return linear_weight * linear_kernel + nonlinear_weight * nonlinear_kernel


def block_trace_ratios(kernel: torch.Tensor, block_sizes: Iterable[int]) -> list[float]:
    block_sizes = tuple(int(size) for size in block_sizes)
    ratios: list[float] = []
    start = 0
    traces = []
    for block_size in block_sizes:
        block = kernel[start : start + block_size, start : start + block_size]
        traces.append(float(torch.trace(block).detach().cpu()))
        start += block_size
    baseline = traces[0]
    for trace in traces:
        ratios.append(trace / baseline if baseline != 0.0 else 1.0)
    return ratios
