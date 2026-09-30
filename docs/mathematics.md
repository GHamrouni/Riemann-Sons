# Mathematics and discretisations

This note records exactly what Riemann & Sons computes, so that results can be trusted (or distrusted)
for the right reasons.

## Metric, inverse, volume element

A metric is any callable `g : (..., n) → (..., n, n)` returning SPD matrices. Inverse, determinant and
`√|g|` are obtained with `torch.linalg`. Learnable metrics keep SPD-ness by construction through an
`SPDParameterization` applied *after* interpolation:

| parameterisation | map | raw parameters |
|---|---|---|
| Cholesky | `g = L Lᵀ + ε I`, `diag L = exp(raw)` | `n(n+1)/2` |
| log-Euclidean | `g = expm(S)`, `S` symmetric | `n(n+1)/2` |
| diagonal | `g = diag(exp(raw))` | `n` |
| conformal | `g = e^{2u} I` | `1` |
| diagonal + low-rank | `g = diag(exp(d)) + U Uᵀ` | `n + n r` |

`GridMetric` stores raw parameters at the nodes of a regular grid and interpolates them with a uniform
cubic B-spline (C², clamped or periodic). Node values are converted to spline coefficients with an exact
separable prefilter, so `from_tensor_field` reproduces the given matrices at the nodes
when the parameterization can represent them and its inverse is exact. Diagonal and
conformal parameterizations project general inputs; the low-rank inverse can approximate them.

## Christoffel symbols and curvature

With `∂g` from `torch.func.jacfwd`,

```
Γ^k_ij = ½ g^{kl} (∂_i g_lj + ∂_j g_li − ∂_l g_ij)
R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
Ric_σν = R^ρ_σρν,   R = g^{σν} Ric_σν,   K = R/2 in 2-D.
```

`∂Γ` is a second `jacfwd`, batched with `vmap` (chunked). Because these are ordinary autograd
operations, `∂R/∂θ` for metric parameters `θ` is available. Sign check: the unit sphere gives `R = +2`,
the Poincaré half-plane `R = −2`.

Limitations: for a B-spline `GridMetric`, curvature is the curvature of the interpolant. It is exact
for the spline but only approximates the intended smooth metric at scales above the grid spacing.

## Laplace–Beltrami operator

```
Δ_g f = |g|^{-1/2} ∂_i ( |g|^{1/2} g^{ij} ∂_j f )
```

is assembled as a multilinear (Q1) finite-element operator on the grid cells with a lumped mass matrix:

```
K_ij = Σ_cells Σ_q w_q ∇φ_i(ξ_q) · A_cell ∇φ_j(ξ_q),   A = |g|^{1/2} g^{-1} at the cell centre,
M    = diag(|g|^{1/2}(node) · w_node),
L    = −M^{-1} K.
```

Here `w_node = Π h` on periodic grids. On a bounded box it is halved for each
axis on whose boundary the node lies (tensor-product trapezoidal quadrature).
`Field.integral` and `Field.riemannian_integral` use these same weights.
`Geometry.total_volume` instead samples cell midpoints and averages the volume
element over the box.

Because every quadrature term is `Bᵀ A B` with `A` SPD and positive weight, `K = Kᵀ ⪰ 0`. Hence
`M L = −K` is symmetric (**weighted self-adjoint**), `L` has nonpositive eigenvalues, `L 1 = 0`, the
Riemannian mass `1ᵀ M f` is conserved by `∂_t f = L f`, and the Dirichlet energy `½ fᵀ K f` is
dissipated. These are tested separately from accuracy in `tests/test_review_fixes.py`.

Two quadratures are available. `"nodal"` (default) is the corner/trapezoidal rule: for `g = I` in 2-D it
reduces to the classical 5-point interior stencil and has no checkerboard null mode.
`"gauss"` uses two-point Gauss quadrature per axis, exact for the element integrands with
the cell-centred coefficient held constant. The scheme is exact for
quadratic `f` and constant `g` in the interior. Boundary faces carry zero flux (natural Neumann
condition) unless the domain is periodic. Neither quadrature guarantees a discrete
maximum principle for arbitrary anisotropic metrics; stable diffusion can still
produce overshoots or negative values.

The operator coefficients keep their autograd history unless `detach=True`, so a loss on a diffused
field differentiates with respect to the metric parameters (checked against finite differences).
`diffuse` builds the operator once per call and reuses it across time steps. It bounds
the absolute row sums of `M^{-1}K` by summing absolute element contributions. If their
maximum is `b`, the automatic Euler step is `dt = 2 · safety / b` (default `safety = 0.4`).
This conservatively bounds the spectral stability limit of the assembled operator;
the final step size is adjusted to reach the requested time exactly. A caller-supplied
`dt` must be positive and finite, but its stability is the caller's responsibility.

## Geodesics

* **Initial value problem.** `ẍ^k + Γ^k_ij ẋ^i ẋ^j = 0` with RK4. The continuous equation
  conserves speed `‖ẋ‖_g`; the discrete integration has a time-step-dependent error.
* **Boundary value problem.** The polyline `x_0 … x_{K−1}` with fixed ends minimises the discrete energy
  `E = (K−1) Σ_k Δx_kᵀ g(m_k) Δx_k` (`m_k` = segment midpoint) with L-BFGS and strong-Wolfe line search.
  This approximates the continuous energy problem, whose minimizers are constant-speed geodesics.
  For the discrete midpoint lengths, `L² ≤ E`, with equality only when all segment lengths
  are equal; even a converged discrete solve need not satisfy equality. The problem is non-convex
  in the presence of obstacles; `init="graph"` starts from a Dijkstra path.
* **Distance fields.** Dijkstra on the grid graph with edge weight `½(‖d‖_{g(p)} + ‖d‖_{g(q)})` and, in 2-D,
  16-connectivity (8 neighbours + knight moves). Sources are snapped to nodes; errors depend
  on grid resolution, direction, and metric variation. The predecessor tree yields shortest
  paths on this discrete graph. Periodic distances wrap, but path coordinates remain in
  the original box; unwrap seam crossings before interpreting ordinary polyline lengths.
* **Parallel transport.** `v̇^k = −Γ^k_ij ẋ^i v^j` with RK4 along each polyline segment; the norm is
  approximately preserved. The analytical holonomy around a latitude circle of the sphere
  is `2π cos θ₀` (mod 2π), which provides a reference for the tests.

## Distances for learning

* `straight_line_distance(g, x, y) = ∫₀¹ ‖y−x‖_{g(x + s(y−x))} ds` (midpoint rule). Cheap, batched,
  and differentiable. The exact integral is an upper bound on `d_g`; the numerical
  quadrature estimate need not preserve that bound.
* `geodesic_distance` runs a batched variational solve and returns the length of the (detached) path
  evaluated with a fresh autograd graph: the **frozen-path gradient**. When the inner problem is at a
  stationary point of the discrete energy `E` and the polyline is constant-speed, `L² = E` and the
  envelope theorem for `E` transfers to `L`, so the frozen-path gradient is exact to first order.
  An incomplete inner solve can give a poor gradient. `Path.info["grad_norm"]` exposes
  stationarity of the returned path; the tests compare a converged solve with finite
  differences of re-solved paths. Inner optimization does not accumulate gradients on
  metric parameters or endpoints. Endpoint gradients through the solver are unsupported.

## Metric flows

* `RicciFlow` updates node metrics `g ← g − 2 dt Ric` (optionally `+ dt (2r/n) g` with `r` the
  volume-weighted mean scalar curvature) and re-encodes them as spline coefficients (collocation).
  Eigenvalues are clamped from below to keep SPD-ness. Explicit stability in 2-D follows from writing the
  flow as `∂_t u = e^{−2u} Δu` for `g = e^{2u} I`: `dt ≲ h² λ_min(g) / (2n)` is the heuristic
  used for adaptive substeps, capped by `max_substeps`. This is not a general stability guarantee. Coordinates are held fixed
  (no DeTurck trick). In two dimensions `Ric = (R/2) g`, so the flow is a pointwise conformal rescaling for
  the continuous equation. Discrete eigenvalue clipping, parameterization, and interpolation
  can change this behavior; the tests compare Cholesky and conformal grids on a smooth case.
* `GradientFlow` is `dθ/dt = −∇_θ E(g_θ)` by explicit Euler in parameter space.
* `HybridFlow` applies its components sequentially (Lie splitting), each with `weight · dt`.
* `integrate` steps to exactly `t_end` (shortening the last step) and treats `t_end ≤ 0` as a no-op.
* All flows are forward simulations: parameters are updated in place under `no_grad` and trajectory
  snapshots are detached. Learning metric parameters is supported; differentiating through an entire
  metric evolution is not.

## Retrieval graph

`GeodesicRetriever(method="graph")` builds the kNN graph once with **unique undirected edges**
(reciprocal neighbours are inserted once; SciPy would otherwise sum duplicate COO entries and double
those edge costs) and links each query to its `k_graph` Euclidean neighbours at query time. Graph
weights and cached distance fields are a snapshot of the metric; `refresh()` rebuilds them.
Disconnected graphs return infinite distances; requesting a path to an unreachable hit
raises an error. Retrieval paths run from query to retrieved point.

## Transformations and Jacobians

`J_Φ = ∂Φ/∂x` by `vmap(jacfwd(Φ))`. Inverses are computed by Newton iteration (fixed-point iteration first
for displacement fields). `warp` samples `I(Φ^{-1}(x))` with multilinear interpolation; points outside the
image domain take the clamped boundary value or `fill` with the default linear interpolation.
Periodic images wrap. Numerical inversion returns its final iterate without guaranteeing
convergence; inspect the residual and Jacobians for difficult deformations. `warp` computes
sampling coordinates without autograd, preserving gradients only with respect to image values.
Folds are flagged where `det J ≤ 0`.
The thin-plate spline uses the kernel `r² log r` in 2-D and `r` in 3-D, written in terms of the squared
distance so that forward-mode AD works (`torch.cdist` does not support it).
