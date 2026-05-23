import sys
from dataclasses import dataclass
from scipy.spatial import KDTree  # Importing KDTree for efficient nearest-neighbor search
import matplotlib.pyplot as plt
import torch

@dataclass
class Measurement:
    index: int
    l: float


class MaxHeap:

    def __init__(self, distances):
        self.maxsize = len(distances)
        self.size = self.maxsize
        self.Heap = [Measurement(index=-1, l=sys.maxsize)] \
                    + [Measurement(index=int(i) + 1, l=float(d)) for i, d in zip(range(self.maxsize), distances)]
        self.ids = [i for i in range(self.maxsize + 1)]  # the ith element's value represent the index of the original ith element in the current heap, 0 means deleted
        self.FRONT = 1
        self._build_max_heap()

    # Function to return the position of the parent for the node currently at pos
    def parent(self, pos):
        return pos // 2

    # Function to return the position of the left child for the node currently at pos
    def leftChild(self, pos):
        return 2 * pos

    # Function to return the position of the right child for the node currently at pos
    def rightChild(self, pos):
        return (2 * pos) + 1

    # Function that returns True if the passed node is a leaf node
    def isLeaf(self, pos):
        return pos > (self.size // 2) and pos <= self.size

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):
        i, j = self.Heap[fpos].index, self.Heap[spos].index
        self.Heap[fpos], self.Heap[spos], self.ids[i], self.ids[j] = self.Heap[spos], self.Heap[fpos], spos, fpos

    # Function to heapify the node at pos
    def _heapify_down(self, pos):
        # If the node is a non-leaf node and smaller than any of its children
        if not self.isLeaf(pos):
            left = self.leftChild(pos)
            right = self.rightChild(pos)
            largest = pos

            if left <= self.size and self.Heap[left].l > self.Heap[largest].l:
                largest = left
            if right <= self.size and self.Heap[right].l > self.Heap[largest].l:
                largest = right

            # Swap and continue heapifying if the current node is not the largest
            if largest != pos:
                self.swap(pos, largest)
                self._heapify_down(largest)



    # Function to remove and return the maximum element from the heap
    def pop(self):
        if self.size == 0:
            raise None
        popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.ids[popped.index] = -1  # Remove this term
        self.size -= 1
        self._heapify_down(self.FRONT)
        return popped

    def decrease_key(self, k, new_distance):
        if k == -1:
            return 0 # the node specified in index is removed

        if self.Heap[k].l > new_distance:
            self.Heap[k].l = new_distance
            self._heapify_down(k)
        return 1

    def _build_max_heap(self):
        for i in range(self.size // 2, 0, -1):
            self._heapify_down(i)



def __reverse_maximin(x, initial = None):
    """Return the reverse maximin ordering and length scales."""
    n = x.size(0)
    indexes = torch.zeros(n, dtype=torch.int64)
    lengths = torch.zeros(n)

    # Arbitrarily select the first point
    if initial is None or initial.size(0) == 0:
        k = 0
        dists = torch.cdist(x, x[k : k + 1], p=2).flatten()
        indexes[-1] = k
        lengths[-1] = float('inf')
        start = n - 2
    else:
        initial_tree = KDTree(initial.numpy())
        dists, _ = initial_tree.query(x.numpy())
        dists = torch.tensor(dists)
        start = n - 1

    # Initialize tree and heap
    tree = KDTree(x.numpy())
    heap = MaxHeap(dists)

    for i in range(start, -1, -1):
        # Select point with the largest minimum distance
        popped = heap.pop()
        k = popped.index - 1
        lk = popped.l
        indexes[i] = k
        lengths[i] = lk
        # Update distances to all points within the distance `lk`
        js = tree.query_ball_point(x[k].numpy(), lk)
        dists = torch.cdist(x[js], x[k:k + 1], p=2).flatten()
        for index, j in enumerate(js):
            heap.decrease_key(heap.ids[j+1], dists[index].item())

    return indexes, lengths


def maximin(x, initial = None):
    indices, lengths = __reverse_maximin(x, initial)
    return torch.flip(indices, dims=[0]), torch.flip(lengths, dims=[0])


def sparsity_pattern(x, lengths, rho):
    """Compute the sparity pattern given the ordered x."""
    # O(n log^2 n + n s)
    tree, offset, length_scale = KDTree(x.numpy()), 0, lengths[0]
    sparsity = {}
    for j in range(len(x)):
        sparsity[j] = [offset + i for i in tree.query_ball_point(x[j], rho * lengths[j]) if offset + i <= j]
    return sparsity


def __col(Theta, s, nugget = 1e-12):
    """
    compute \Theta_{s, s}^{-1}e_j / \sqrt{e_j^T\Theta_{s, s}^{-1}e_j}

    A simple implementation is to invert \Theta_{s, s} directly using the following code

    m = torch.inverse(Theta[s][:, s])
    return m[:, -1] / torch.sqrt(m[-1, -1])

    However, it has some stability issues. When, \Theta_{s, s} is ill-conditioned, torch.inverse(Theta[s, s]) may not
    give us the accurate solution and e_j^T\Theta_{s, s}^{-1}e_j may be negative and torch.sqrt(m[-1, -1]) produces NaN value.

    Thus, instead, we use standard Cholesky directly on \Theta_{s, s}^{-1} and also adding a small nugget to prevent numerical issue
    """
    try:
        m = Theta[s][:, s] + nugget * torch.eye(len(s))
        L = torch.linalg.cholesky(m)
        ej = torch.zeros(len(s))  # Create a tensor of zeros
        ej[-1] = 1
        v = torch.cholesky_solve(ej.unsqueeze(-1), L, upper=False).squeeze(-1)
        # print(v.shape)
        if v[-1] < 0:
            raise ValueError(f"Negative value encountered for square root: {v[-1]}")
        vl = torch.sqrt(v[-1])
        return v / vl
    except torch.linalg.LinAlgError as e:
        # Handle Cholesky decomposition failure
        print("When doing sparse Cholesky decomposition, Cholesky decomposition of the submatrix \Theta_{s, s} " + f"failed: {e}")
        raise  # Optionally re-raise the exception after logging
    except ValueError as e:
        # Handle negative square root or other value issues
        print("When doing sparse Cholesky decomposition, \Theta_{s, s}[-1, -1] is negative. " + f"Value error encountered: {e}")
        raise  # Optionally re-raise the exception


def __cholesky(Theta, sparsity):
    n = Theta.size(0)
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    # Prepare storage for sparse matrix components
    data = torch.zeros(total_nonzeros, dtype=torch.float64)
    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        col_data = __col(Theta, s)
        start, end = indptr[i], indptr[i + 1]

        data[start:end] = col_data
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    sparse_cholesky = torch.sparse_coo_tensor(indices, data, size=(n, n))
    return sparse_cholesky

def non_zeros(n, sparsity):
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        start, end = indptr[i], indptr[i + 1]
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    return indices


def sparse_cholesky(Theta, Perm, sparsity) -> torch.sparse_coo_tensor:
    reordered_Theta = Theta[Perm][:, Perm]
    return __cholesky(reordered_Theta, sparsity)


if __name__ == '__main__':
    import torch
    torch.set_default_dtype(torch.float64)

    A = torch.tensor([[4.0, 1.0, 0.5, 0.0, 0.0],
                      [1.0, 3.0, 0.5, 0.0, 0.0],
                      [0.5, 0.5, 2.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0, 3.0, 0.5],
                      [0.0, 0.0, 0.0, 0.5, 1.0]])
    Theta = A @ A.T  # Ensure positive definiteness

    # Input points (not used in this example, but required for function signature)
    x = torch.linspace(0, 1, 5)[1:][:, None]
    
    print(x)

    # Sparsity control parameter
    rho = 3

    # Compute sparse Cholesky factor

    Perm, lengths = maximin(x)
    sparsity = sparsity_pattern(x[Perm], lengths, rho)
    U_sparse = sparse_cholesky(Theta, Perm, sparsity)

    # Convert sparse tensor to dense for validation
    U_dense = U_sparse.to_dense()

    invPerm = torch.argsort(Perm)
    # Validate L_sparse satisfies Theta^-1 = LL^T
    Theta_inverse_reconstructed = (U_dense @ U_dense.T)[invPerm][:, invPerm]
    Theta_inverse = torch.inverse(Theta)

    # Condition number before preconditioning
    cond_before = torch.linalg.cond(Theta)

    # Precondition Theta
    Theta_preconditioned = U_dense.T @ Theta[Perm][:, Perm] @ U_dense #L_dense @ Theta @ L_dense.T

    # Condition number after preconditioning
    cond_after = torch.linalg.cond(Theta_preconditioned)

    # Output results
    print("Original Theta^-1:\n", Theta_inverse)
    print("\nReconstructed Theta^-1:\n", Theta_inverse_reconstructed)
    print("\nSparse Cholesky Factor (U):\n", U_dense)
    print("\nCondition number before preconditioning:", cond_before.item())
    print("Condition number after preconditioning:", cond_after.item())

    # Check correctness
    reconstruction_error = torch.linalg.norm(Theta_inverse - Theta_inverse_reconstructed, ord='fro')
    print("\nReconstruction error of Theta^-1:", reconstruction_error.item())
