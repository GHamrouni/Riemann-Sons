# Demos

Install the package using the [root README](../README.md), then run any script
from the repository root. Figures and GIFs land in `examples/output/`.
The shared setup uses float64 and a noninteractive Matplotlib backend, so a
display server is unnecessary. Start with A–D; E–H include longer optimization
or simulation runs.

| Script | Demo | What you see |
|---|---|---|
| `demo_a_manual_geometry.py` | A | Same scene, three hand-built geometries: metric ellipses, geodesics, distance fields, curvature |
| `demo_b_image_geometry.py` | B | Image → structure-tensor metric → geodesics → distance field → edge-aware diffusion |
| `demo_c_diffusion.py` | C | `∂ₜI = ΔI` vs `∂ₜI = Δ_g I` on the same image under three geometries |
| `demo_d_deformation.py` | D | Landmark-driven smooth transformations, warped images and grids, `det J_Φ` with fold detection |
| `demo_e_metric_learning.py` | E | A 2-D metric learned from triplets, epoch by epoch, plus how nearest neighbours change (GIF) |
| `demo_f_retrieval_lab.py` | F | Euclidean vs learned Mahalanobis vs learned-geodesic retrieval on two moons, held-out queries over 3 seeds: top-K, paths, contours, precision@K |
| `demo_g_ricci_and_task_flows.py` | G | Ricci flow on a torus (Gauss–Bonnet, volume, smoothing), task-driven and hybrid metric flows (GIF) |
| `demo_h_learn_geometry_from_diffusion.py` | H | Learn a metric from diffusion of the same inputs on two grid resolutions; evaluate held-out patterns and times, metric shape, and scale |

```bash
python examples/demo_a_manual_geometry.py
```

The images in [`docs/images/`](../docs/images/) illustrate earlier runs. Rerun
the scripts to evaluate the current implementation; figures and numerical
results can change with solver corrections and dependency versions.
