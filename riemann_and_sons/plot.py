"""Matplotlib helpers -- the 2-D visual laboratory.

Conventions: fields are stored as ``(N_x, N_y)`` with coordinate ``x_0``
along array axis 0; for ``imshow`` we therefore transpose and use
``origin="lower"`` with the domain extent, so that plots are in *domain
coordinates* and everything (points, geodesics, ellipses) overlays correctly.
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.patches import Ellipse
from torch import Tensor

from .fields import Field, Image

__all__ = [
    "heatmap",
    "scalar_field",
    "image",
    "metric_field",
    "ellipse_parameters",
    "geodesics",
    "points",
    "distance_field",
    "curvature",
    "metric_scalar",
    "deformation_grid",
    "jacobian_map",
    "vector_field",
]


def _ax(ax=None, figsize=(5, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _extent(domain):
    return domain.extent


# ------------------------------------------------------------------ scalars
def heatmap(values: Tensor, domain, ax=None, cmap="viridis", colorbar=True, title=None, vmin=None, vmax=None, alpha=1.0, contours=None, contour_kw=None, symmetric=False, log=False, robust: float | None = None):
    """Plot a ``(N_x, N_y)`` array in domain coordinates.

    ``robust=q`` clips the colour range to the ``q``-quantile of ``|values|`` so a
    few singular pixels do not wash out the picture.
    """
    ax = _ax(ax)
    v = _np(values)
    if log:
        v = np.log10(np.clip(v, 1e-300, None))
    finite_abs = np.abs(v[np.isfinite(v)])
    if robust is not None and finite_abs.size and vmax is None:
        vmax = float(np.quantile(finite_abs, robust))
        if not symmetric and vmin is None:
            vmin = float(np.quantile(v[np.isfinite(v)], 1 - robust))
    if symmetric:
        m = (np.nanmax(finite_abs) if finite_abs.size else 1.0) if vmax is None else vmax
        vmin, vmax = -m, m
    im = ax.imshow(v.T, origin="lower", extent=_extent(domain), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, aspect="equal", interpolation="nearest")
    if contours:
        xs = np.linspace(domain.lo[0].item(), domain.hi[0].item(), v.shape[0])
        ys = np.linspace(domain.lo[1].item(), domain.hi[1].item(), v.shape[1])
        kw = {"colors": "white", "linewidths": 0.6, "alpha": 0.8}
        kw.update(contour_kw or {})
        finite = v[np.isfinite(v)]
        if finite.size:
            levels = contours if not isinstance(contours, int) else np.linspace(finite.min(), finite.max(), contours + 2)[1:-1]
            ax.contour(xs, ys, v.T, levels=levels, **kw)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_xlim(domain.lo[0].item(), domain.hi[0].item())
    ax.set_ylim(domain.lo[1].item(), domain.hi[1].item())
    return im


def scalar_field(field: Field, ax=None, **kw):
    vals = field.values
    if vals.ndim > field.dim:
        vals = vals.reshape(-1, *field.resolution).mean(0)
    return heatmap(vals, field.domain, ax=ax, **kw)


def image(img: Field, ax=None, title=None, cmap="gray", vmin=0.0, vmax=1.0, alpha=1.0):
    """Show an :class:`Image` (or single-channel field) in domain coordinates."""
    ax = _ax(ax)
    vals = img.values
    if vals.ndim == img.dim:
        vals = vals[None]
    arr = _np(vals.permute(2, 1, 0))  # (H, W, C) with y upwards
    if arr.shape[-1] == 1:
        ax.imshow(arr[..., 0], origin="lower", extent=_extent(img.domain), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, aspect="equal")
    else:
        ax.imshow(np.clip(arr[..., :3], 0, 1), origin="lower", extent=_extent(img.domain), alpha=alpha, aspect="equal")
    if title:
        ax.set_title(title)
    return ax


# ------------------------------------------------------------------ metric
def ellipse_parameters(g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Unit ball ``vᵀ g v = 1`` of a batch of 2×2 SPD matrices -> (semi-axis a, semi-axis b, angle in degrees).

    Semi-axes are ``1/√λ`` along the eigenvectors: a *large* eigenvalue means the
    direction is expensive, so the unit ball is *short* along it.
    """
    evals, evecs = torch.linalg.eigh(g)
    a = 1 / torch.sqrt(evals[..., 0])  # along the eigenvector of the smallest eigenvalue (long axis)
    b = 1 / torch.sqrt(evals[..., 1])
    ang = torch.atan2(evecs[..., 1, 0], evecs[..., 0, 0]) * 180 / math.pi
    return a, b, ang


def metric_field(
    geometry,
    ax=None,
    resolution=(15, 15),
    scale: float = 0.42,
    normalize: str = "global",
    color="C0",
    edgecolor=None,
    alpha: float = 0.9,
    linewidth: float = 1.0,
    fill: bool = False,
    color_by: str | None = None,
    cmap="viridis",
    background=None,
    title=None,
):
    """Draw the local unit balls ``vᵀ g(x) v = 1`` of a 2-D metric as ellipses.

    * ``normalize="global"``: one scale for the whole plot -- ellipse *size* is
      meaningful (small = expensive region, large = cheap region).
    * ``normalize="local"``: every ellipse is scaled to its cell -- shows only
      anisotropy and orientation.
    ``color_by`` may be ``"anisotropy"``, ``"det"`` or ``"logdet"`` to colour the ellipses.
    """
    ax = _ax(ax)
    dom = geometry.domain
    if dom.dim != 2:
        raise ValueError("metric_field only plots 2-D geometries")
    res = dom._shape(resolution)
    # cell-centred sample points
    h = dom.size / torch.as_tensor(res, dtype=dom.lo.dtype)
    axes = [dom.lo[k] + h[k] * (torch.arange(res[k], dtype=dom.lo.dtype) + 0.5) for k in range(2)]
    pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        g = geometry.g(pts)
    a, b, ang = ellipse_parameters(g)
    cell = float(h.min())
    if normalize == "global":
        s = scale * cell / float(a.max())
        widths, heights = 2 * a * s, 2 * b * s
    elif normalize == "local":
        widths, heights = 2 * a / a * scale * cell, 2 * b / a * scale * cell
    else:
        raise ValueError("normalize must be 'global' or 'local'")
    if background is not None:
        if isinstance(background, Image) or background.values.ndim > 2:
            image(background, ax=ax)
        else:
            scalar_field(background, ax=ax, colorbar=False, cmap="gray")
    kwargs = dict(units="xy", offsets=_np(pts), transOffset=ax.transData, linewidths=linewidth, alpha=alpha)
    if color_by is not None:
        with torch.no_grad():
            ev = torch.linalg.eigvalsh(g)
            val = {"anisotropy": ev[:, 1] / ev[:, 0], "det": ev.prod(-1), "logdet": torch.log(ev.prod(-1))}[color_by]
        coll = EllipseCollection(_np(widths), _np(heights), _np(ang), facecolors="none" if not fill else None, **kwargs)
        coll.set_array(_np(val))
        coll.set_cmap(cmap)
        if fill:
            coll.set_edgecolors("none")
        else:
            coll.set_facecolors("none")
        ax.add_collection(coll)
        plt.colorbar(coll, ax=ax, fraction=0.046, pad=0.04, label=color_by)
    else:
        coll = EllipseCollection(_np(widths), _np(heights), _np(ang), facecolors=color if fill else "none", edgecolors=edgecolor or color, **kwargs)
        ax.add_collection(coll)
    ax.set_xlim(dom.lo[0].item(), dom.hi[0].item())
    ax.set_ylim(dom.lo[1].item(), dom.hi[1].item())
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return coll


def metric_scalar(geometry, quantity: str = "logdet", ax=None, resolution=96, **kw):
    """Heatmap of a pointwise metric invariant: ``det``, ``logdet``, ``volume`` (√det), ``lmin``, ``lmax``, ``condition``."""
    dom = geometry.domain
    pts = dom.grid(resolution)
    with torch.no_grad():
        g = geometry.g(pts)
        ev = torch.linalg.eigvalsh(g)
        vals = {
            "det": ev.prod(-1),
            "logdet": torch.log(ev.prod(-1)),
            "volume": torch.sqrt(ev.prod(-1)),
            "lmin": ev[..., 0],
            "lmax": ev[..., -1],
            "condition": ev[..., -1] / ev[..., 0],
        }[quantity]
    kw.setdefault("title", {"det": "det g", "logdet": "log det g", "volume": "√det g", "lmin": "λ_min(g)", "lmax": "λ_max(g)", "condition": "λ_max/λ_min"}[quantity])
    return heatmap(vals, dom, ax=ax, **kw)


def curvature(geometry, ax=None, resolution=64, cmap="RdBu_r", chunk=2048, title="scalar curvature R", **kw):
    """Heatmap of the scalar curvature, symmetric colour scale around zero (robust to isolated spikes)."""
    pts = geometry.domain.grid(resolution)
    with torch.no_grad():
        R = geometry.scalar_curvature(pts, chunk=chunk)
    kw.setdefault("symmetric", True)
    if float(R.abs().max()) < 1e-8:  # numerically flat: show a clean zero instead of round-off noise
        kw.setdefault("vmax", 1.0)
        kw.setdefault("robust", None)
    else:
        kw.setdefault("robust", 0.98)
    return heatmap(R, geometry.domain, ax=ax, cmap=cmap, title=title, **kw)


# ------------------------------------------------------------------ paths & points
def geodesics(paths, ax=None, color="C3", linewidth=1.8, alpha=1.0, endpoints=True, label=None, **kw):
    ax = _ax(ax)
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    segs = []
    for p in paths:
        pts = _np(p.points if hasattr(p, "points") else p)
        segs.append(pts)
    lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, label=label, **kw)
    ax.add_collection(lc)
    if endpoints:
        starts = np.array([s[0] for s in segs])
        ends = np.array([s[-1] for s in segs])
        ax.scatter(starts[:, 0], starts[:, 1], c="white", edgecolors="black", s=30, zorder=5)
        ax.scatter(ends[:, 0], ends[:, 1], c="black", edgecolors="white", s=30, zorder=5)
    ax.set_aspect("equal")
    return lc


def points(x: Tensor, ax=None, labels=None, s=18, cmap="tab10", alpha=0.9, **kw):
    ax = _ax(ax)
    x = _np(x)
    if labels is not None:
        return ax.scatter(x[:, 0], x[:, 1], c=_np(labels), cmap=cmap, s=s, alpha=alpha, **kw)
    return ax.scatter(x[:, 0], x[:, 1], s=s, alpha=alpha, **kw)


def distance_field(df, ax=None, contours=15, cmap="viridis", title="geodesic distance", **kw):
    """Heatmap + iso-distance contours + sources."""
    ax = _ax(ax)
    vals = df.field.values.clone()
    vals[~torch.isfinite(vals)] = float("nan")
    im = heatmap(vals, df.field.domain, ax=ax, cmap=cmap, contours=contours, title=title, **kw)
    src = _np(df.sources)
    ax.scatter(src[:, 0], src[:, 1], marker="*", s=140, c="white", edgecolors="black", zorder=6)
    return im


# ------------------------------------------------------------------ deformation
def deformation_grid(transformation, domain, ax=None, lines=16, samples=120, color="C0", linewidth=0.7, alpha=0.9, title=None):
    """Draw the image of a regular coordinate grid under ``Φ``."""
    ax = _ax(ax)
    lo, hi = domain.lo, domain.hi
    xs = torch.linspace(lo[0].item(), hi[0].item(), lines)
    ys = torch.linspace(lo[1].item(), hi[1].item(), lines)
    t0 = torch.linspace(lo[0].item(), hi[0].item(), samples)
    t1 = torch.linspace(lo[1].item(), hi[1].item(), samples)
    segs = []
    with torch.no_grad():
        for x in xs:  # vertical lines x = const
            line = torch.stack([x.expand(samples), t1], -1)
            segs.append(_np(transformation(line)))
        for y in ys:  # horizontal lines
            line = torch.stack([t0, y.expand(samples)], -1)
            segs.append(_np(transformation(line)))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha))
    ax.set_xlim(lo[0].item(), hi[0].item())
    ax.set_ylim(lo[1].item(), hi[1].item())
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def jacobian_map(transformation, domain, ax=None, resolution=128, quantity="det", highlight_folds=True, cmap="RdBu_r", title=None, **kw):
    """Heatmap of ``det J``, ``cond J``, or singular values; folds (``det ≤ 0``) hatched in black."""
    from .deformation import jacobian_stats

    ax = _ax(ax)
    pts = domain.grid(resolution)
    with torch.no_grad():
        st = jacobian_stats(transformation, pts)
    if quantity == "det":
        vals = st.det
        kw.setdefault("vmin", 0.0)
        kw.setdefault("vmax", float(2 * torch.quantile(vals.abs().flatten(), 0.5)))
        title = title or "det J_Φ"
    elif quantity == "cond":
        vals = st.condition_number
        cmap = "magma"
        title = title or "condition number of J_Φ"
    elif quantity == "smax":
        vals = st.singular_values[..., 0]
        cmap = "magma"
        title = title or "σ_max(J_Φ)"
    elif quantity == "smin":
        vals = st.singular_values[..., -1]
        cmap = "magma"
        title = title or "σ_min(J_Φ)"
    else:
        raise ValueError("quantity must be det, cond, smax or smin")
    im = heatmap(vals, domain, ax=ax, cmap=cmap, title=title, **kw)
    if highlight_folds and st.has_folds:
        mask = _np(st.orientation_reversed).astype(float)
        xs = np.linspace(domain.lo[0].item(), domain.hi[0].item(), mask.shape[0])
        ys = np.linspace(domain.lo[1].item(), domain.hi[1].item(), mask.shape[1])
        ax.contourf(xs, ys, mask.T, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(xs, ys, mask.T, levels=[0.5], colors="black", linewidths=1.2)
    return im


def vector_field(values: Tensor, domain, ax=None, step=6, color="white", scale=None, **kw):
    """Quiver plot of a gridded vector field ``(N_x, N_y, 2)``."""
    ax = _ax(ax)
    v = _np(values)
    pts = _np(domain.grid(v.shape[:2]))
    sl = (slice(None, None, step), slice(None, None, step))
    ax.quiver(pts[sl][..., 0], pts[sl][..., 1], v[sl][..., 0], v[sl][..., 1], color=color, scale=scale, **kw)
    return ax
