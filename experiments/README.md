# Clean experiment entry points

These scripts are the cleaned, cell-based entry points for the main KROM experiments.

- `Burgers_clean.py`
  Uses Crank-Nicolson full-order snapshots, separate empirical and Matérn solver cells, sparse derivative-aware precision factors, and batch evaluation on multiple unseen initial conditions.
- `Allen_Cahn_clean.py`
  Uses Crank-Nicolson full-order snapshots, separate empirical and Matérn solver cells, sparse derivative-aware precision factors, and batch evaluation on multiple unseen initial conditions.
- `Elliptic_Nonlinear_PDE_clean.py`
  Uses the shared Laplacian-based Matérn assembly and Gauss-Newton residual/Jacobian code for a cleaner nonlinear elliptic setup.

The shared reusable code now lives in `src/krom/`. The legacy notebooks are still present for reference, but the cleaned scripts are the intended place to compare kernels without manually toggling cells.
