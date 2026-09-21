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

is discretised in flux form on a staggered grid: for each axis `i`, `∂_i f` is a forward difference at
the face `i + ½`; cross derivatives `∂_j f` (`j ≠ i`) are central differences at the nodes averaged to
the face; the coefficients `|g|^{1/2} g^{ij}` are averaged to the face; the flux is differenced back to
the nodes and divided by `|g|^{1/2}`. Boundary faces carry zero flux (Neumann) unless the domain is
periodic. The scheme is exact for quadratic `f` and constant `g` in the interior and conserves the
Riemannian mass `∫ f √|g| dx`.

`diffuse` integrates `∂_t f = Δ_g f` with explicit Euler and the step
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
* `geodesic_distance` runs a batched variational solve for a few iterations and returns the length of the
  (detached) path evaluated with a fresh autograd graph. By the envelope theorem this gradient equals the
  gradient of the true geodesic distance to first order.

## Metric flows

* `RicciFlow` updates node metrics `g ← g − 2 dt Ric` (optionally `+ dt (2r/n) g` with `r` the
  volume-weighted mean scalar curvature) and re-encodes them as spline coefficients (collocation).
  Eigenvalues are clamped from below to keep SPD-ness. Explicit stability in 2-D follows from writing the
  flow as `∂_t u = e^{−2u} Δu` for `g = e^{2u} I`: `dt ≲ h² λ_min(g) / (2n)`. Coordinates are held fixed
  (no DeTurck trick). In two dimensions `Ric = (R/2) g`, so the flow is a pointwise conformal rescaling for
  *any* parameterisation (checked numerically: Cholesky and conformal grids evolve identically).
* `GradientFlow` is `dθ/dt = −∇_θ E(g_θ)` by explicit Euler in parameter space.
* `HybridFlow` applies its components sequentially (Lie splitting), each with `weight · dt`.

## Transformations and Jacobians

`J_Φ = ∂Φ/∂x` by `vmap(jacfwd(Φ))`. Inverses are computed by Newton iteration (fixed-point iteration first
for displacement fields). `warp` samples `I(Φ^{-1}(x))` with multilinear interpolation; points outside the
image domain take the clamped boundary value or `fill`. Folds are flagged where `det J ≤ 0`.
The thin-plate spline uses the kernel `r² log r` in 2-D and `r` in 3-D, written in terms of the squared
distance so that forward-mode AD works (`torch.cdist` does not support it).
