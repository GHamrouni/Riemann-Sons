"""Demo C -- geometry-dependent diffusion.

The same image, the same times, three geometries:

    ∂_t I = Δ I          (Euclidean)
    ∂_t I = Δ_g I        (g from the image itself: edge-aware diffusion)
    ∂_t I = Δ_g I        (g = conformal wall: information cannot cross x = 0.5)

Changing g changes how information propagates.  Nothing about the image changed.
"""

from _common import plt, rn, save, timer
from riemann_and_sons import plot, synthetic

dom = rn.Plane()
scene = synthetic.shapes_scene(dom, 96, soft=0.008)
times = [0.0, 0.0005, 0.002, 0.006]

geometries = {
    "Euclidean  Δ": rn.Geometry(dom),
    "image metric  Δ_g,  g = I + λ∇I∇Iᵀ": rn.Geometry(dom, rn.image_induced_metric(scene, lam=0.02, sigma=1.5)),
    "wall metric  Δ_g,  g = λ_wall(x) I": rn.Geometry(dom, synthetic.wall_metric(cost=400.0, gap=(0.62, 0.72))),
}

fig, axes = plt.subplots(len(geometries), len(times), figsize=(4.2 * len(times), 4.2 * len(geometries)))
for row, (name, geo) in enumerate(geometries.items()):
    with timer(name):
        f = scene
        t_prev = 0.0
        for col, t in enumerate(times):
            if t > t_prev:
                f = rn.diffuse(f, geo, t - t_prev)
                t_prev = t
            plot.image(f, ax=axes[row, col], title=f"{name}\nt = {t:g}" if col == 0 else f"t = {t:g}")
            if col == 0 and row > 0:
                plot.metric_field(geo, ax=axes[row, col], resolution=(12, 12), color="C1", alpha=0.5, normalize="local")
        print(f"    Riemannian mass: {scene.riemannian_integral(geo).item():.5f} -> {f.riemannian_integral(geo).item():.5f}")
save(fig, "demo_c_diffusion.png")
