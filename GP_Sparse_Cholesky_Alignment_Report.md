# GP-PDEs-SparseCholesky Alignment — Final Report

This covers all work done against the official repo
(`https://github.com/yifanc96/GP-PDEs-SparseCholesky`) and the underlying paper
(Chen, Owhadi, Schäfer, *"Sparse Cholesky Factorization for Solving Nonlinear
PDEs via Gaussian Processes"*, Math. Comp. 2025, arXiv:2304.01294).

## 1. Ordering variants (`src/krom/ordering.py`)

- **k-maximin ordering** — added, matching the official repo's coarse-to-fine
  multiresolution ordering scheme that the paper's complexity proofs depend on.
- **`follow_diracs` variant** — added. It requires every entry in
  `derivative_point_groups` to be co-located 1:1 with `dirac_points`
  (`group.numel() == total_dirac_points` and `torch.equal(group, arange(...))`).
  I checked this structurally against every PDE experiment file. Only
  `navier_stokes.py` (and the related `Kolmogorov_ROM_Comparison.py`, see §3)
  satisfy it — every other PDE (Burgers, Darcy, Elliptic, Allen-Cahn,
  Moving-Domain-Heat) has dedicated boundary Dirac points with *no* derivative
  measurements, so `follow_diracs` is not structurally usable there. Those
  files stay on the default `dirac_first_then_unif_scale` variant.

## 2. Matérn 7/2 kernel (`src/krom/kernels.py`)

Added the full family of 2D Matérn-7/2 derivative kernels (value, first/second
derivatives, Laplacians, mixed Laplacian-derivative terms) plus the 1D
derivative orders, following the exact code style of the pre-existing
Matérn-5/2 functions. All 16 2D functions and 9 1D derivative orders were
verified to machine precision (1e-13 to 1e-16 relative error) against exact
sympy symbolic derivatives — not finite differences, which would only catch
errors above ~1e-4. A `MATERN_2D_KERNELS` registry dict was added as an
additive convenience layer exposing both "5/2" and "7/2" families by string
key; no existing call site was forced to switch kernels, since every PDE file
currently hardcodes `matern52_2d*` by name.

## 3. Hyperparameter alignment

Audited every `rho` / `lengthscale` / `nugget` / `k_neighbors` default
against the official README's stated guidance. Findings: `rho` (3.0–4.0,
the README's "sweet spot" is 3.0) and `lengthscale`/`nugget` were already
well-aligned everywhere — no changes needed. `k_neighbors` was the one clear
gap: the README states `k_neighbors=1` is the default for a plain point
cloud, but `k_neighbors=3` is the usual default for PDE problems with
derivative measurements. The codebase implicitly used 1 everywhere. I added
a threaded `k_neighbors: int = 3` default to:

- `src/krom/sparse_cholesky.py` (`sparse_precision_factor`)
- `src/krom/moving_domain.py` (`build_moving_heat_factors`)
- `experiments/Burgers_clean.py`, `Allen_Cahn_clean.py`, `Darcy_Flow_clean.py`,
  `Elliptic_Nonlinear_PDE_clean.py`, `Moving_Domain_Heat_clean.py`

**Deliberately left untouched:** `src/krom/navier_stokes.py`,
`experiments/NS_Vorticity_clean.py`, and
`experiments/Kolmogorov_ROM_Comparison.py`. The last one isn't explicitly
named "NS" in your instructions, but it imports directly from
`navier_stokes.py` (`build_navier_stokes_empirical_theta`,
`solve_streamfunction_with_fft`) and has the identical structural pattern —
`dirac_points = points` with no separate boundary set, so it is in fact
eligible for `follow_diracs` the same way `navier_stokes.py` is. Given its
content is squarely Navier-Stokes/Kolmogorov-flow, I treated it as part of
the protected NS family and left its `k_neighbors`/ordering defaults alone.
Flagging this explicitly in case you'd prefer it aligned like the other PDE
files instead — it's a one-line change if so.

## 4. Supernodal-style batched Cholesky (`src/krom/sparse_cholesky.py`)

The original factorization looped over columns one at a time in Python, each
doing its own dense `cholesky_ex` + `cholesky_solve` on its local sparsity
block. Added `_batched_normalized_precision_columns`, which groups columns by
**support size** (a cheap, vectorizable proxy for true supernode grouping,
which additionally requires matching row *patterns*, not just sizes) and
factors each size-group with a single batched `torch.linalg.cholesky_ex` +
batched `torch.cholesky_solve` call, instead of one Python call per column.
Per-column adaptive jitter escalation (retry with 10x jitter on failure) is
preserved exactly, applied per-batch-element.

This is wired in as `sparse_precision_factor(..., batch_columns=True)`
(default on); `batch_columns=False` recovers the exact original sequential
path for debugging.

**Correctness:** since torch can't execute in this sandbox, I built a
pure-numpy mirror of both the sequential and batched algorithms and ran 200
random trials plus grouped multi-column batches — max absolute difference
was exactly `0.0`. The batched path is provably the same computation, just
vectorized; it changes wall-clock performance (via GPU/vectorization
parallelism across same-size columns) but **not** asymptotic complexity or
numerical results.

## 5. JAX+GPU vs. scipy investigation

Checked the current (June 2026) status of `jax.experimental.sparse`: it
remains explicitly experimental — the JAX team's own docs describe it as
"reference implementations not recommended for performance-critical
applications," with development effectively frozen (existing features are
supported, but no active investment). So your original reasoning for
rejecting JAX still holds; nothing material has changed there.

That said, the factorization's actual bottleneck is **dense** linear algebra
on small local blocks (the per-column Cholesky), not genuinely sparse GPU
kernels — the sparsity only shows up in *which* points are gathered into each
block, and in the final CSC assembly/triangular solves. Both PyTorch and JAX
have mature, near-identical batched dense GPU Cholesky support
(`torch.linalg.cholesky_ex` with a batch dim vs. `jax.vmap(jnp.linalg.cholesky)`),
so switching frameworks wouldn't unlock anything beyond what the
torch-native batching in §4 already captures. The genuinely sparse
operations downstream (CSC assembly, `scipy`/CHOLMOD triangular solves) stay
CPU-bound either way, since neither vanilla PyTorch nor JAX's sparse module
wraps a GPU sparse direct solver (e.g. NVIDIA cuDSS) — that would require a
dedicated GPU sparse-solver binding regardless of which array framework
drives it. **Recommendation: stay on torch; the batched-column change in §4
is the practical GPU win available without taking on JAX's still-immature
sparse stack.**

## 6. Complexity-claim parity with the paper

The paper proves the approximate inverse Cholesky factor can be computed in
**O(N log^d(N/ε)) space** and **O(N log^{2d}(N/ε)) time**, where N is the
number of measurements and d the spatial dimension, *provided* the ordering
is the multiresolution maximin-type scheme and the sparsity pattern is the
ρ-ball pattern under that ordering — exactly the algorithm structure already
implemented here (`build_measurement_ordering` with maximin/k-maximin +
`build_sparsity_pattern`'s `cKDTree.query_ball_point`, followed by per-column
dense local solves). This is the same "KL-minimization via local Cholesky
updates" scheme from Schäfer–Owhadi's near-linear-complexity dense-kernel
papers that the PDE paper builds on, and its near-linear bound specifically
relies on the *hierarchical* shrinking of block sizes at finer resolutions —
not on every block being the same fixed size — so the implementation's
per-column dense `cholesky_ex` (cost cubic in *that column's* support size)
is consistent with, not a deviation from, the paper's bound.

What this means concretely: the implementation is algorithmically the right
shape to inherit the paper's complexity, but I can't empirically *confirm*
the asymptotic scaling here, since doing so requires running factorizations
across a range of N on real torch/scipy and timing/measuring memory — and
this sandbox cannot execute torch or scipy (see caveat in §7). The
supernodal batching in §4 does **not** change this asymptotic picture either
way; it's a constant-factor wall-clock optimization, not an algorithmic
complexity change. For the Gauss-Newton PDE solve specifically, the paper's
near-linear claim composes the per-iteration factorization cost with a
bounded GN iteration count — the experiment files already use small fixed
`gn_steps` (1, 2, or 4), consistent with that assumption.

If you want empirical confirmation rather than a structural argument, the
natural next step is a small benchmark script (varying N, plotting factor
build time and nnz against N log^d(N) / N log^{2d}(N)) — I can write that,
but it needs to be run on your machine since this sandbox has no working
torch/scipy.

## 7. Verification caveat

All verification this session used pure numpy/sympy mirrors of the torch
logic (machine-precision exact for the kernels, bit-identical for the
batched-vs-sequential Cholesky), because this sandbox cannot run torch or
scipy — not even via your own pre-built venv at `/Users/jonghyeonlee/ROM/krom/`,
since its compiled binaries are macOS Mach-O and this sandbox is Linux.

**Recommended one-time smoke test on your Mac:**

```bash
/Users/jonghyeonlee/ROM/krom/bin/python3 -c "
import torch
from src.krom.kernels import matern72_2d, MATERN_2D_KERNELS
from src.krom.ordering import build_measurement_ordering
from src.krom.sparse_cholesky import sparse_precision_factor
print('imports ok')
"
```
run from `/Users/jonghyeonlee/ROM`, plus running one of the edited experiment
files (e.g. `Burgers_clean.py`) end-to-end to confirm no runtime regressions
from the `k_neighbors` threading or the new batched Cholesky path.
