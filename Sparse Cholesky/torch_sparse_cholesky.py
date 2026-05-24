from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.krom.factors import SparseInverseFactor
from src.krom.ordering import MeasurementOrdering, build_measurement_ordering, maximin
from src.krom.sparse_cholesky import build_sparsity_pattern, sparse_precision_factor


def sparse_cholesky_from_measurements(
    theta: torch.Tensor,
    dirac_points: torch.Tensor,
    derivative_point_groups: tuple[torch.Tensor, ...] = (),
    rho: float = 3.0,
    nugget: float = 1e-10,
) -> tuple[SparseInverseFactor, MeasurementOrdering]:
    return sparse_precision_factor(
        theta=theta,
        dirac_points=dirac_points,
        derivative_point_groups=derivative_point_groups,
        rho=rho,
        nugget=nugget,
    )


__all__ = [
    'MeasurementOrdering',
    'SparseInverseFactor',
    'build_measurement_ordering',
    'build_sparsity_pattern',
    'maximin',
    'sparse_cholesky_from_measurements',
]
