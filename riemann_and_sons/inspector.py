"""The visual geometry inspector: one call to look at everything a 2-D geometry has to say."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from . import plot
from .fields import Field, Image
from .flows import MetricTrajectory, diffuse
from .geometry import Geometry
from .paths import Path, distance_field, geodesic

__all__ = ["Inspector"]


class Inspector:
    """Multi-panel inspection of a 2-D :class:`Geometry`, optionally with a field living on it.

    Example::

        insp = Inspector(geometry, field=image)
        insp.show(metric=True, curvature=True, distance=sources, geodesics=[(a, b)])
        insp.animate(trajectory, panels=("metric", "curvature"), path="flow.gif")
    """

    def __init__(self, geometry: Geometry, field: Field | None = None, resolution: int = 96, ellipses=(14, 14)):
        if geometry.dim != 2:
            raise ValueError("Inspector is a 2-D tool")
        self.geometry = geometry
        self.field = field
        self.resolution = resolution
        self.ellipses = ellipses

    # ----------------------------------------------------------------- panels
    def _panels(
        self,
        metric=True,
        det=False,
        eigenvalues=False,
        condition=False,
        volume=False,
        curvature=False,
        christoffel=False,
        distance: Tensor | None = None,
        geodesics: Sequence | None = None,
        field=None,
        diffusion: float | None = None,
        jacobian=None,
        geodesic_kw: dict | None = None,
    ) -> list[tuple[str, Callable]]:
        geo = self.geometry
        dom = geo.domain
        show_field = self.field is not None if field is None else field
        panels: list[tuple[str, Callable]] = []
        paths: list[Path] = []
        if geodesics:
            for item in geodesics:
                if isinstance(item, Path):
                    paths.append(item)
                else:
                    a, b = item
                    paths.append(geodesic(geo, torch.as_tensor(a), torch.as_tensor(b), **(geodesic_kw or {"init": "graph"})))

        def bg(ax):
            if show_field and self.field is not None:
                (plot.image if isinstance(self.field, Image) else lambda f, ax: plot.scalar_field(f, ax=ax, colorbar=False, cmap="gray"))(self.field, ax=ax)

        if show_field and self.field is not None:
            panels.append(("field", lambda ax: (bg(ax), ax.set_title("field"))))
        if metric:
            def draw_metric(ax):
                bg(ax)
                plot.metric_field(geo, ax=ax, resolution=self.ellipses, color="C1" if show_field else "C0", title="metric unit balls  vᵀg v = 1")
                if paths:
                    plot.geodesics(paths, ax=ax)
            panels.append(("metric", draw_metric))
        if det:
            panels.append(("det", lambda ax: plot.metric_scalar(geo, "logdet", ax=ax, resolution=self.resolution)))
        if volume:
            panels.append(("volume", lambda ax: plot.metric_scalar(geo, "volume", ax=ax, resolution=self.resolution)))
        if eigenvalues:
            panels.append(("lmin", lambda ax: plot.metric_scalar(geo, "lmin", ax=ax, resolution=self.resolution, log=True, title="log10 λ_min")))
            panels.append(("lmax", lambda ax: plot.metric_scalar(geo, "lmax", ax=ax, resolution=self.resolution, log=True, title="log10 λ_max")))
        if condition:
            panels.append(("condition", lambda ax: plot.metric_scalar(geo, "condition", ax=ax, resolution=self.resolution, cmap="magma")))
        if christoffel:
            def draw_gamma(ax):
                pts = dom.grid(self.resolution // 2)
                with torch.no_grad():
                    G = geo.christoffel(pts, chunk=2048)
                plot.heatmap((G**2).sum(dim=(-1, -2, -3)).sqrt(), dom, ax=ax, cmap="magma", title="‖Γ‖")
            panels.append(("christoffel", draw_gamma))
        if curvature:
            panels.append(("curvature", lambda ax: plot.curvature(geo, ax=ax, resolution=self.resolution // 2 * 2 // 2 if self.resolution > 64 else self.resolution)))
        if distance is not None:
            df = distance_field(geo, torch.as_tensor(distance), self.resolution)
            def draw_dist(ax):
                plot.distance_field(df, ax=ax)
                if paths:
                    plot.geodesics(paths, ax=ax, color="white")
            panels.append(("distance", draw_dist))
        if geodesics and not metric and distance is None:
            def draw_geo(ax):
                bg(ax)
                plot.geodesics(paths, ax=ax)
                ax.set_title("geodesics")
            panels.append(("geodesics", draw_geo))
        if diffusion is not None and self.field is not None:
            out = diffuse(self.field, geo, diffusion)
            panels.append(("diffusion", lambda ax: (plot.image if isinstance(out, Image) else plot.scalar_field)(out, ax=ax, title=f"Δ_g diffusion, t={diffusion:g}")))
        if jacobian is not None:
            panels.append(("jacobian", lambda ax: plot.jacobian_map(jacobian, dom, ax=ax, resolution=self.resolution)))
            panels.append(("grid", lambda ax: plot.deformation_grid(jacobian, dom, ax=ax, title="deformed grid")))
        return panels

    def show(self, ncols: int = 3, panel_size: float = 4.2, **kw):
        panels = self._panels(**kw)
        if not panels:
            raise ValueError("nothing to show")
        n = len(panels)
        ncols = min(ncols, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for (name, draw), ax in zip(panels, axes.flat):
            draw(ax)
        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------- animation
    def animate(self, trajectory: MetricTrajectory, panels: Sequence[str] = ("metric", "curvature"), path: str | None = None, fps: int = 6, panel_size: float = 4.2, panel_kw: dict | None = None, **kw):
        """Animate a :class:`MetricTrajectory` through the requested panels; saves a GIF when ``path`` is given."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        panel_kw = panel_kw or {}
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(panel_size * n, panel_size), squeeze=False)
        base_geo = self.geometry

        def frame(i):
            for ax in axes.flat:
                ax.clear()
            geo = Geometry(base_geo.domain, trajectory.at(i))
            insp = Inspector(geo, self.field, self.resolution, self.ellipses)
            flags = {name: True for name in panels}
            flags.update(panel_kw)
            drawn = dict(insp._panels(**{k: v for k, v in flags.items() if k != "field"}, field=("field" in panels)))
            for name, ax in zip(panels, axes.flat):
                drawn[name](ax)
            fig.suptitle(f"t = {trajectory.times[i]:.3f}")

        anim = FuncAnimation(fig, frame, frames=len(trajectory), interval=1000 / fps)
        if path is not None:
            anim.save(path, writer=PillowWriter(fps=fps))
        return anim
