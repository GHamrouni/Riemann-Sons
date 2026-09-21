"""Demo B -- image-induced geometry.

    Image -> metric field -> geodesics -> distance field -> geometric operation

The metric is the structure-tensor metric

    g_I(x) = I + λ ∇I_σ(x) ∇I_σ(x)ᵀ

(I_σ: image blurred with a Gaussian of σ pixels; see `image_induced_metric`).
Crossing an edge costs √(1 + λ|∇I|²) per unit length; sliding along it costs 1.
The image defines the geometry, and the geometry then acts back on the image
(here: Riemannian diffusion that does not leak across edges).
"""

import torch

from _common import plt, rn, save, timer
from riemann_and_sons import plot, synthetic

dom = rn.Plane()
scene = synthetic.shapes_scene(dom, 128, soft=0.008)
with timer("metric from image"):
    metric = rn.image_induced_metric(scene, lam=0.02, sigma=1.5)
geo = rn.Geometry(dom, metric)
euclid = rn.Geometry(dom)

pairs = [((0.3, 0.7), (0.72, 0.3)), ((0.1, 0.9), (0.9, 0.1)), ((0.15, 0.5), (0.9, 0.5))]
with timer("geodesics"):
    riem = [rn.geodesic(geo, torch.tensor(a), torch.tensor(b), n_points=96, init="graph", graph_resolution=128) for a, b in pairs]
    eucl = [rn.geodesic(euclid, torch.tensor(a), torch.tensor(b), n_points=8) for a, b in pairs]
    for (a, b), p in zip(pairs, riem):
        straight = rn.learning.straight_line_distance(metric, torch.tensor(a), torch.tensor(b), samples=400).item()
        print(f"    {a}->{b}: d_g = {p.length(geo).item():.3f}   straight segment under g: {straight:.3f}   Euclidean: {p.euclidean_length().item():.3f}")

fig, axes = plt.subplots(1, 5, figsize=(24, 5))
plot.image(scene, ax=axes[0], title="1. image  I : M → R")
plot.image(scene, ax=axes[1], alpha=0.4)
plot.metric_field(geo, ax=axes[1], resolution=(20, 20), color_by="anisotropy", normalize="local", title="2. metric field  g = I + λ∇I∇Iᵀ\n(unit balls, coloured by λmax/λmin)")
plot.image(scene, ax=axes[2], alpha=0.4)
plot.geodesics(eucl, ax=axes[2], color="C0", linewidth=1.2, linestyle="--", endpoints=False, label="Euclidean")
plot.geodesics(riem, ax=axes[2], color="C3", label="Riemannian")
axes[2].legend(loc="lower left")
axes[2].set_title("3. geodesics")
with timer("distance field"):
    df = rn.distance_field(geo, torch.tensor([[0.3, 0.7]]), 128)
plot.distance_field(df, ax=axes[3], contours=24, title="4. distance field from ★ (inside the disk)")
with timer("Riemannian diffusion"):
    blurred = rn.diffuse(scene, geo, 0.0025)
plot.image(blurred, ax=axes[4], title="5. operation: Δ_g diffusion, t = 0.0025\n(edges act as walls)")
save(fig, "demo_b_image_geometry.png")
