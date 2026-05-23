from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DenseCholeskyFactor:
    chol: torch.Tensor

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol, vec.unsqueeze(-1), upper=False).squeeze(-1)

    def apply_jacobian(self, jacobian: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol, jacobian, upper=False)


@dataclass
class SparseInverseFactor:
    factor: torch.Tensor
    permutation: torch.Tensor

    def __post_init__(self) -> None:
        self.factor = self.factor.coalesce()
        self.permutation = self.permutation.to(dtype=torch.long, device=self.factor.device)
        self.transpose_factor = self.factor.transpose(0, 1).coalesce()
        self.inverse_permutation = torch.argsort(self.permutation)

    @property
    def nnz(self) -> int:
        return int(self.factor._nnz())

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        permuted = vec.index_select(0, self.permutation)
        return torch.sparse.mm(self.transpose_factor, permuted.unsqueeze(-1)).squeeze(-1)

    def apply_jacobian(self, jacobian: torch.Tensor) -> torch.Tensor:
        permuted = jacobian.index_select(0, self.permutation)
        return torch.sparse.mm(self.transpose_factor, permuted)
