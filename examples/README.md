# Demos

Run any script from this directory; figures and GIFs land in `examples/output/`.

| Script | Demo | What you see |
|---|---|---|
| `demo_a_manual_geometry.py` | A | Same scene, three hand-built geometries: metric ellipses, geodesics, distance fields, curvature |
| `demo_b_image_geometry.py` | B | Image → structure-tensor metric → geodesics → distance field → edge-aware diffusion |
| `demo_c_diffusion.py` | C | `∂ₜI = ΔI` vs `∂ₜI = Δ_g I` on the same image under three geometries |
| `demo_d_deformation.py` | D | Landmark-driven smooth transformations, warped images and grids, `det J_Φ` with fold detection |
| `demo_e_metric_learning.py` | E | A 2-D metric learned from triplets, epoch by epoch, plus how nearest neighbours change (GIF) |
| `demo_f_retrieval_lab.py` | F | Euclidean vs learned-geodesic retrieval on two moons: top-K, paths, contours, precision@K |
| `demo_g_ricci_and_task_flows.py` | G | Ricci flow on a torus (Gauss–Bonnet, volume, smoothing), task-driven and hybrid metric flows (GIF) |

```bash
cd examples
python demo_a_manual_geometry.py
```
