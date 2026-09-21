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
separable prefilter, so `from_tensor_field` reproduces the given matrices at the nodes.

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
M    = diag(|g|^{1/2}(node) · Π h),
L    = −M^{-1} K.
```

Because every quadrature term is `Bᵀ A B` with `A` SPD and positive weight, `K = Kᵀ ⪰ 0`. Hence
`M L = −K` is symmetric (**weighted self-adjoint**), `L` is negative semi-definite, `L 1 = 0`, the
Riemannian mass `1ᵀ M f` is conserved by `∂_t f = L f`, and the Dirichlet energy `½ fᵀ K f` is
dissipated. These are tested separately from accuracy in `tests/test_review_fixes.py`.

Two quadratures are available. `"nodal"` (default) is the corner/trapezoidal rule: for `g = I` it
reduces to the classical 5-point stencil, it has no checkerboard null mode, and on our tests it is
2–4× more accurate than `"gauss"` (2-point Gauss per axis, exact for the element integrands). Both
are second order: on a smoothly varying anisotropic periodic metric the max error against a 256²
reference decreases as 4.5e-2 → 1.2e-2 → 2.8e-3 for 16², 32², 64² (nodal). The scheme is exact for
quadratic `f` and constant `g` in the interior. Boundary faces carry zero flux (natural Neumann
condition) unless the domain is periodic.

The operator coefficients keep their autograd history unless `detach=True`, so a loss on a diffused
field differentiates with respect to the metric parameters (checked against finite differences).
`diffuse` builds the operator once per call and reuses it across time steps; its explicit Euler step is
`dt = 0.4 · h_min² / (2 n λ_max(g^{-1}))`.

## Geodesics

* **Initial value problem.** `ẍ^k + Γ^k_ij ẋ^i ẋ^j = 0` with RK4; speed `‖ẋ‖_g` is conserved to
  round-off on the sphere test.
* **Boundary value problem.** The polyline `x_0 … x_K` with fixed ends minimises the discrete energy
  `E = (K−1) Σ_k Δx_kᵀ g(m_k) Δx_k` (`m_k` = segment midpoint) with L-BFGS and strong-Wolfe line search.
  Minimisers of `E` are constant-speed geodesics, and `L² = E` at the optimum. The problem is non-convex
  in the presence of obstacles; `init="graph"` starts from a Dijkstra path.
* **Distance fields.** Dijkstra on the grid graph with edge weight `½(‖d‖_{g(p)} + ‖d‖_{g(q)})` and, in 2-D,
  16-connectivity (8 neighbours + knight moves). The metrication error is about 1–2 %; the predecessor
  tree yields discrete shortest paths.
* **Parallel transport.** `v̇^k = −Γ^k_ij ẋ^i v^j` with RK4 along each polyline segment; the norm is
  preserved and the holonomy around a latitude circle of the sphere is `2π cos θ₀` (mod 2π).

## Distances for learning

* `straight_line_distance(g, x, y) = ∫₀¹ ‖y−x‖_{g(x + s(y−x))} ds` (midpoint rule). Cheap, batched,
  differentiable, and an **upper bound** on `d_g`.
* `geodesic_distance` runs a batched variational solve and returns the length of the (detached) path
  evaluated with a fresh autograd graph: the **frozen-path gradient**. When the inner problem is at a
  stationary point of the discrete energy `E` and the polyline is constant-speed, `L² = E` and the
  envelope theorem for `E` transfers to `L`, so the frozen-path gradient is exact to first order.
  Measured on a 6×6 grid metric against central finite differences of re-solved geodesics: 1.7e-4
  relative error at convergence (`grad_norm ≈ 1e-5`), but about 65 % error after 10 L-BFGS iterations.
  `Path.info["grad_norm"]` exposes the stationarity of the returned path.

## Metric flows

* `RicciFlow` updates node metrics `g ← g − 2 dt Ric` (optionally `+ dt (2r/n) g` with `r` the
  volume-weighted mean scalar curvature) and re-encodes them as spline coefficients (collocation).
  Eigenvalues are clamped from below to keep SPD-ness. Explicit stability in 2-D follows from writing the
  flow as `∂_t u = e^{−2u} Δu` for `g = e^{2u} I`: `dt ≲ h² λ_min(g) / (2n)`. Coordinates are held fixed
  (no DeTurck trick). In two dimensions `Ric = (R/2) g`, so the flow is a pointwise conformal rescaling for
  *any* parameterisation (checked numerically: Cholesky and conformal grids evolve identically).
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

## Transformations and Jacobians

`J_Φ = ∂Φ/∂x` by `vmap(jacfwd(Φ))`. Inverses are computed by Newton iteration (fixed-point iteration first
for displacement fields). `warp` samples `I(Φ^{-1}(x))` with multilinear interpolation; points outside the
image domain take the clamped boundary value or `fill`. Folds are flagged where `det J ≤ 0`.
The thin-plate spline uses the kernel `r² log r` in 2-D and `r` in 3-D, written in terms of the squared
distance so that forward-mode AD works (`torch.cdist` does not support it).
