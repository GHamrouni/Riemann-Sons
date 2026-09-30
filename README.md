# Riemann & Sons

A PyTorch toolkit for experimenting with **how geometry changes distances, diffusion, and retrieval**.

A metric tensor `g(x)` assigns a cost to movement at each point. Keep your field, image, or point cloud fixed; change the metric to study how operations on it change. Metrics can be defined by hand or learned from data.

![The same points under four different geometries](docs/images/hero.png)

This is a research library for small, low-dimensional experiments, with visualization tools for 2-D. It is useful for checking geometric ideas and learning metrics; it is not a scalable vector-search backend. The API is experimental.

## Install

Requires Python 3.13 or newer. From a checkout:

```bash
git clone https://github.com/GHamrouni/Riemann-Sons.git
cd Riemann-Sons
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Dependencies: PyTorch, NumPy, SciPy, and Matplotlib. The examples run on CPU; SciPy-based graph and retrieval operations use CPU arrays. GPU support is not tested end to end.

## Quick start

This complete example makes horizontal movement twice as expensive as vertical movement, finds a path, and diffuses a signal under that geometry:

```python
import torch
import riemann_and_sons as rn

# Set precision before creating tensors, domains, or metrics.
# Importing the library does not change PyTorch's default dtype.
torch.set_default_dtype(torch.float64)

domain = rn.Plane()  # the unit square [0, 1] × [0, 1]
metric = rn.ConstantMetric(torch.diag(torch.tensor([4.0, 1.0])))
geometry = rn.Geometry(domain, metric)

a = torch.tensor([0.2, 0.5])
b = torch.tensor([0.8, 0.5])
path = rn.geodesic(geometry, a, b, n_points=16)
print(f"Path length: {path.length(geometry).item():.3f}")  # 1.200

signal = rn.Field.from_function(
    domain, 32, lambda x: torch.exp(-((x - 0.5) ** 2).sum(-1) / 0.01)
)
smoothed = rn.diffuse(signal, geometry, t=0.002)

# Compare with ordinary Euclidean diffusion of the same signal.
euclidean = geometry.with_metric(rn.EuclideanMetric())
baseline = rn.diffuse(signal, euclidean, t=0.002)

fig = rn.Inspector(geometry, field=smoothed).show(metric=True)
fig.savefig("geometry.png")
```

`Geometry` pairs a domain with a metric. `Field` stores values on a regular grid; `Image` adds image loading and coordinate conversion. Operations take the geometry explicitly. Use float64 for curvature and other precision-sensitive calculations, and keep inputs and metrics on compatible dtypes and devices.

## What you can do

| Task | Main API | Example |
|---|---|---|
| Define a geometry | `Box`, `Plane`, `Geometry`; constant, conformal, function, grid, and pullback metrics | [Manual geometry](examples/demo_a_manual_geometry.py) |
| Derive geometry from an image | `Image.from_file`, `image_induced_metric` | [Image geometry](examples/demo_b_image_geometry.py) |
| Compute paths and curvature | `geodesic`, `exp_map`, `distance_field`, `geometry.scalar_curvature` | [Manual geometry](examples/demo_a_manual_geometry.py) |
| Diffuse a field | `LaplaceBeltrami`, `diffuse` | [Diffusion](examples/demo_c_diffusion.py) |
| Deform an image | `Affine`, `Displacement`, `ThinPlateSpline`, `warp`, `jacobian_stats` | [Deformation](examples/demo_d_deformation.py) |
| Learn a metric | `GridMetric`, `learning.fit_metric` | [Triplet learning](examples/demo_e_metric_learning.py) |
| Compare retrieval methods | Euclidean, cosine, Mahalanobis, and `GeodesicRetriever` | [Held-out retrieval](examples/demo_f_retrieval_lab.py) |
| Evolve a metric | `RicciFlow`, `GradientFlow`, `HybridFlow` | [Metric flows](examples/demo_g_ricci_and_task_flows.py) |
| Learn from diffusion observations | Differentiate through `diffuse` | [Diffusion learning](examples/demo_h_learn_geometry_from_diffusion.py) |

Run an example from the repository root after installing:

```bash
python examples/demo_a_manual_geometry.py
```

Figures and GIFs go to `examples/output/`. See the [example guide](examples/README.md) for the full set. Learning and flow demos take longer than the introductory examples; their results are experiments, not general performance guarantees.

## Correctness and limits

The tests check sphere and hyperbolic curvature, known geodesics, transport, diffusion mass conservation and energy dissipation, and gradients against finite differences. These checks cover specific analytical cases and discrete invariants; they do not establish accuracy for every metric or resolution.

- **Paths are numerical approximations.** The variational solver can find local minima. Use `init="graph"` for obstacles and inspect `path.info["grad_norm"]`. Grid and graph distances depend on sampling and connectivity; there is no universal percentage error bound.
- **Straight-line distance is a path-length estimate.** The exact line integral bounds geodesic distance from above, but finite quadrature does not guarantee a bound.
- **Differentiation has limits.** Metric quantities and diffusion support parameter gradients. Optimized path points are detached; `path.length(geometry)` differentiates the metric along the frozen path. This approximates a distance gradient only when the inner solve is sufficiently converged. SciPy graph distances are not differentiable.
- **Resolution matters.** `GridMetric` curvature is the curvature of its spline interpolant. Check convergence as you refine spatial grids and time steps. The anisotropic diffusion scheme does not guarantee positivity or a discrete maximum principle.
- **Metric flows are forward simulations.** They update parameters in place and detach snapshots. Ricci flow here is intended for small 2-D experiments, especially periodic domains.
- **Retrieval caches geometry.** After changing a metric, call `GeodesicRetriever.refresh()` to rebuild its snapshot. Graph connectivity can leave some distances infinite.
- **Periodic distances need care.** Grid distance fields wrap, but returned paths use coordinates in the original box. A path crossing a seam is not a continuous polyline in those coordinates; its plotted segments and `Path.length` need unwrapping.

## Conventions and details

Points have shape `(..., n)` and metrics return `(..., n, n)` symmetric positive-definite matrices. Grid axis `k` corresponds to coordinate `k`. Fields store spatial dimensions last; `Image.from_array` converts conventional image rows and columns to this layout.

Metric ellipses show `vᵀ g v = 1`: a shorter axis means more expensive movement. The curvature sign convention gives scalar curvature `+2` on the unit sphere and `−2` on the Poincaré half-plane.

See [mathematics and discretizations](docs/mathematics.md) for formulas, gradient semantics, and solver assumptions. The [roadmap](docs/roadmap.md) describes future work.

MIT licensed. See [LICENSE](LICENSE).
