from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class DenseCholeskyFactor:
    chol: torch.Tensor

    @property
    def nnz(self) -> int:
        return int(self.chol.numel())

    @property
    def backend(self) -> str:
        return 'torch_dense'

    @property
    def backend_name(self) -> str:
        return 'torch_dense'

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol, vec.unsqueeze(-1), upper=False).squeeze(-1)

    def apply_transpose(self, vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol.transpose(0, 1), vec.unsqueeze(-1), upper=True).squeeze(-1)

    def apply_jacobian(self, jacobian: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol, jacobian, upper=False)


@dataclass
class SparseInverseFactor:
    factor: sp.csc_matrix
    permutation: np.ndarray | torch.Tensor
    backend_name: str = 'scipy_sparse'

    def __post_init__(self) -> None:
        self.factor = self.factor.tocsc()
        self.transpose_factor = self.factor.transpose().tocsr()
        if isinstance(self.permutation, torch.Tensor):
            self.permutation = self.permutation.detach().cpu().numpy()
        self.permutation = np.asarray(self.permutation, dtype=np.int64)
        self.inverse_permutation = np.argsort(self.permutation)

    @property
    def nnz(self) -> int:
        return int(self.factor.nnz)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().contiguous().numpy()

    def _from_numpy(self, array: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        out = torch.from_numpy(np.asarray(array))
        if out.dtype != dtype:
            out = out.to(dtype=dtype)
        if device.type != 'cpu':
            out = out.to(device=device)
        return out

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        vec_np = self._to_numpy(vec)
        result = self.transpose_factor.dot(vec_np[self.permutation])
        return self._from_numpy(result, dtype=vec.dtype, device=vec.device)

    def apply_transpose(self, vec: torch.Tensor) -> torch.Tensor:
        vec_np = self._to_numpy(vec)
        ordered = self.factor.dot(vec_np)
        return self._from_numpy(ordered[self.inverse_permutation], dtype=vec.dtype, device=vec.device)

    def apply_jacobian(self, jacobian: torch.Tensor) -> torch.Tensor:
        jac_np = self._to_numpy(jacobian)
        result = self.transpose_factor.dot(jac_np[self.permutation, :])
        return self._from_numpy(result, dtype=jacobian.dtype, device=jacobian.device)


def dense_precision_factor(theta: torch.Tensor, nugget: float = 1e-10) -> DenseCholeskyFactor:
    eye = torch.eye(theta.shape[0], dtype=theta.dtype, device=theta.device)
    jitter = max(float(nugget), 0.0)
    if jitter == 0.0:
        chol, info = torch.linalg.cholesky_ex(theta)
        if int(info.item()) == 0:
            return DenseCholeskyFactor(chol=chol)
        jitter = float(torch.finfo(theta.dtype).eps)

    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(theta + jitter * eye)
        if int(info.item()) == 0:
            return DenseCholeskyFactor(chol=chol)
        jitter *= 10.0

    raise torch.linalg.LinAlgError('Dense kernel matrix remained indefinite after jitter escalation.')
