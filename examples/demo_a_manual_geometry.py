"""Demo A -- construct a geometry by hand.

Same points.  Same coordinates.  Different geometry.  Different notion of distance.

A synthetic scene (circles, rectangles) is the *field*.  Three geometries are
placed on the very same plane:

1. Euclidean          g = I                         -> straight geodesics
2. Conformal obstacles g = λ(x) I, λ ≫ 1 on shapes  -> geodesics bend around them
3. Anisotropic        g = I + (κ−1) t tᵀ           -> radial travel is √κ× more
   expensive than circling, so geodesics spiral.  In polar coordinates this is
   g = κ dr² + r² dθ², a *cone*: flat everywhere except at the apex, which is
   why its curvature panel is identically zero.

For every geometry we show metric ellipses + geodesics, a geodesic distance
field, and the scalar curvature.
"""

import torch

from _common import plt, rn, save, timer
from riemann_and_sons import plot, synthetic

dom = rn.Plane()
scene = synthetic.shapes_scene(dom, 128)
pairs = [((0.08, 0.5), (0.92, 0.5)), ((0.12, 0.12), (0.88, 0.88)), ((0.5, 0.06), (0.5, 0.94))]

geometries = {
    "Euclidean   g = I": rn.Geometry(dom),
    "Conformal obstacles   g = λ(x) I": rn.Geometry(
        dom, synthetic.bump_conformal_metric(centers=[[0.3, 0.7], [0.72, 0.3], [0.72, 0.72]], radii=[0.14, 0.11, 0.12], heights=[60, 60, 60])
    ),
    "Anisotropic   g = I + 15 t tᵀ  (t radial)": rn.Geometry(dom, synthetic.anisotropic_flow_metric(ratio=16)),
}

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
for row, (name, geo) in enumerate(geometries.items()):
    with timer(name):
        paths = [rn.geodesic(geo, torch.tensor(a), torch.tensor(b), n_points=80, init="graph", graph_resolution=96) for a, b in pairs]
        ax = axes[row, 0]
        plot.image(scene, ax=ax, alpha=0.35)
        plot.metric_field(geo, ax=ax, resolution=(16, 16), color="C1")
        plot.geodesics(paths, ax=ax)
        ax.set_title(f"{name}\nunit balls vᵀg v = 1  +  geodesics")
        df = rn.distance_field(geo, torch.tensor([[0.08, 0.5]]), 128)
        plot.distance_field(df, ax=axes[row, 1], contours=18, title="geodesic distance from ★")
        plot.geodesics(paths[:1], ax=axes[row, 1], color="white")
        plot.curvature(geo, ax=axes[row, 2], resolution=72)
        for p, (a, b) in zip(paths, pairs):
            print(f"    {a} -> {b}: length {p.length(geo).item():.3f}   (Euclidean {p.euclidean_length().item():.3f})")
save(fig, "demo_a_manual_geometry.png")
