"""``Geometry = (Domain, Metric)`` -- the central object of Riemann & Sons."""

from __future__ import annotations

import torch
from torch import Tensor

from .domains import Box
from .metrics import EuclideanMetric, Metric

__all__ = ["Geometry"]


class Geometry:
    """A Riemannian manifold ``(M, g)`` given by a coordinate :class:`Box` and a :class:`Metric`.

    The class is deliberately thin: it bundles the two ingredients and offers
    convenience wrappers around the differential-geometry primitives.  All
    heavy lifting lives in :mod:`curvature`, :mod:`paths`, :mod:`operators`
    and :mod:`flows`, which accept a ``Geometry`` as an argument so that the
    same field / image / point cloud can be re-interpreted under a different
    geometry without being touched.
    """

    def __init__(self, domain: Box, metric: Metric | None = None):
        self.domain = domain
        self.metric = metric if metric is not None else EuclideanMetric()
        if self.metric.dim is not None and self.metric.dim != domain.dim:
            raise ValueError(f"metric dimension {self.metric.dim} != domain dimension {domain.dim}")

    @property
    def dim(self) -> int:
        return self.domain.dim

    def with_metric(self, metric: Metric) -> "Geometry":
        """Same domain, different geometry."""
        return Geometry(self.domain, metric)

    # ---- pointwise quantities -------------------------------------------
    def g(self, x: Tensor) -> Tensor:
        return self.metric(x)

    def g_inv(self, x: Tensor) -> Tensor:
        return self.metric.inverse(x)

    def volume_element(self, x: Tensor) -> Tensor:
        return self.metric.volume_element(x)

    def norm(self, x: Tensor, v: Tensor) -> Tensor:
        return self.metric.norm(x, v)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return self.metric.inner(x, u, v)

    def christoffel(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return self.metric.christoffel(x, chunk=chunk)

    def riemann(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return self.metric.riemann(x, chunk=chunk)

    def ricci(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return self.metric.ricci(x, chunk=chunk)

    def scalar_curvature(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return self.metric.scalar_curvature(x, chunk=chunk)

    def gauss_curvature(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return self.metric.gauss_curvature(x, chunk=chunk)

    # ---- grid helpers -----------------------------------------------------
    def grid(self, resolution) -> Tensor:
        return self.domain.grid(resolution)

    def total_volume(self, resolution=64) -> Tensor:
        """Riemannian volume ``∫ √|g| dx`` using ``resolution`` midpoint cells per axis."""
        shape = self.domain._shape(resolution)
        lo, size = self.domain.lo, self.domain.size
        axes = [lo[k] + (torch.arange(m, dtype=lo.dtype, device=lo.device) + 0.5) * size[k] / m
                for k, m in enumerate(shape)]
        pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return self.volume_element(pts).mean() * size.prod()

    # ---- paths & distances (thin wrappers) ---------------------------------
    def geodesic(self, x0: Tensor, x1: Tensor, **kw):
        from .paths import geodesic

        return geodesic(self, x0, x1, **kw)

    def exp(self, x: Tensor, v: Tensor, **kw) -> Tensor:
        from .paths import exp_map

        return exp_map(self, x, v, **kw)

    def distance_field(self, sources: Tensor, resolution=128, **kw):
        from .paths import distance_field

        return distance_field(self, sources, resolution, **kw)

    def __repr__(self) -> str:
        return f"Geometry(domain={self.domain!r}, metric={self.metric})"
