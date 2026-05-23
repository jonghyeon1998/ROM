# KROM Follow-Up Notes

## Empirical kernels for time-dependent PDEs

- Replace the current time-average kernel with a time-lagged feature map. Stack `u(t)`, `u(t-\tau)`, `u(t-2\tau)`, and operator features such as `u_x`, `u_xx`, or `\Delta u` before forming the Gram matrix. This lets the kernel encode local dynamics rather than only trajectory means.
- Use exponentially decaying temporal weights instead of uniform averaging when forming empirical features. This gives more emphasis to near-future dynamics and usually stabilizes long rollouts.
- Keep separate empirical kernels for state and operator features, then combine them additively with tuned weights. This is often easier to calibrate than one monolithic block kernel.
- Add a nonlinear empirical component, such as a Frobenius RBF kernel on trajectory windows or a low-degree polynomial kernel on snapshot features, and compare it against the current linear kernel and an additive linear-plus-nonlinear kernel.
- For Burgers, Allen-Cahn, and Navier-Stokes, consider windowed kernels that only use a short temporal stencil. This is usually more faithful to Markovian time stepping than a full-history average.

## Stronger generalization tests

- Test on a batch of unseen initial conditions and forcing terms rather than a single held-out case. Report mean, median, and worst-case relative errors over the full test batch.
- Keep training and test distributions matched. If the forcing terms come from a Gaussian process prior or a bounded Fourier family, sample both train and test from the same family first, then add distribution shift later as a separate stress test.
- Once the initial-condition and forcing generalization study is stable, add parameter generalization one axis at a time, for example viscosity in Burgers or interface width in Allen-Cahn.

## Fair ROM comparisons

- Compare wall-clock online solve time at matched accuracy, not only fixed reduced dimension. KROM does not use POD truncation, so the fairest comparison is error-versus-runtime and error-versus-memory.
- Report the offline and online stages separately for every method. KROM has kernel assembly and sparse factorization costs; POD-ROMs have snapshot generation, SVD, operator projection, and hyper-reduction costs.
- Include at least one POD-Galerkin baseline, one POD-DEIM or POD-ECSW style hyper-reduced baseline, and one operator-learning baseline if you want a modern ML comparison.
- Match the same training snapshots, test set, and stopping tolerances across methods. Otherwise the comparison is easy to overstate in either direction.
- For nonlinear elliptic and Darcy examples, compare against Newton solves in the reduced coordinates with the same number of retained snapshots or basis vectors. This isolates the benefit of sparse precision-factor solves from the benefit of different trial spaces.

## 3D Navier-Stokes benchmarks

- 3D Taylor-Green vortex is the cleanest first benchmark because it has a standard setup, published reference curves, and transition-to-turbulence behavior.
- 3D lid-driven cavity flow is a good steady or weakly unsteady benchmark for boundary-driven dynamics and pressure-velocity coupling.
- Flow past a sphere or cylinder at moderate Reynolds number is strong if you want wake dynamics and force coefficients that are easy to compare.
- 3D channel flow or periodic box decaying turbulence is a good choice if you want to emphasize long-time rollout fidelity and energy-spectrum preservation.
- Rayleigh-Bénard convection in 3D is attractive if you want a coupled multiphysics benchmark with rich coherent structures.
