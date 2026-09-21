r"""Differential operators for fields on a geometry (finite differences on the grid).

For a scalar field ``f`` on ``(M, g)``:

* Euclidean gradient       ``∂_i f``                  -- :func:`gradient`
* Riemannian gradient      ``(∇f)^i = g^{ij} ∂_j f``  -- :func:`riemannian_gradient`
* divergence               ``div V = |g|^{-1/2} ∂_i (|g|^{1/2} V^i)`` -- :func:`divergence`
* Laplace–Beltrami         ``Δ_g f = |g|^{-1/2} ∂_i (|g|^{1/2} g^{ij} ∂_j f)`` -- :func:`laplace_beltrami`

The Laplace–Beltrami operator is discretised in *conservative* (flux) form on
a staggered grid: fluxes ``|g|^{1/2} g^{ij} ∂_j f`` are evaluated at cell
faces and differenced back to the nodes.  This avoids the odd/even decoupling
of a naive central-difference ``div(grad)`` and yields a symmetric operator
with zero-flux (Neumann) boundaries on boxes, or periodic wrap-around.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .fields import Field

__all__ = ["gradient", "riemannian_gradient", "divergence", "laplace_beltrami", "LaplaceBeltrami", "structure_tensor"]


def _central(f: Tensor, axis: int, h: float, periodic: bool) -> Tensor:
    if periodic:
        return (torch.roll(f, -1, axis) - torch.roll(f, 1, axis)) / (2 * h)
    return torch.gradient(f, spacing=float(h), dim=axis, edge_order=1)[0]


def gradient(field: Field) -> Tensor:
    """Euclidean gradient ``∂_i f`` with shape ``(*channels, *res, n)``."""
    f = field.values
    n = field.dim
    h = field.spacing
    comps = [_central(f, f.ndim - n + i, h[i].item(), field.domain.periodic) for i in range(n)]
    return torch.stack(comps, dim=-1)


def riemannian_gradient(field: Field, geometry) -> Tensor:
    """``(∇f)^i = g^{ij} ∂_j f`` with shape ``(*channels, *res, n)``."""
    df = gradient(field)
    ginv = geometry.g_inv(field.points)
    return torch.einsum("...ij,...j->...i", ginv, df)


def divergence(vector_values: Tensor, geometry, domain=None) -> Field:
    """Riemannian divergence of a contravariant vector field given on the grid, ``(*res, n) -> Field``."""
    from .domains import Box

    dom = domain if domain is not None else geometry.domain
    n = dom.dim
    res = tuple(vector_values.shape[-n - 1 : -1])
    pts = dom.grid(res)
    sqrtg = geometry.volume_element(pts)
    h = dom.spacing(res)
    lead = vector_values.ndim - n - 1
    out = torch.zeros_like(vector_values[..., 0])
    for i in range(n):
        Fi = sqrtg * vector_values[..., i]
        out = out + _central(Fi, lead + i, h[i].item(), dom.periodic)
    return Field(out / sqrtg, dom)


class LaplaceBeltrami:
    r"""``Δ_g`` on a fixed grid with the metric coefficients precomputed.

    Build once, apply many times (this is what :func:`flows.diffuse` does)::

        op = LaplaceBeltrami(geometry, field.resolution, periodic=field.domain.periodic)
        lap = op(field)
    """

    def __init__(self, geometry, resolution, domain=None):
        self.domain = domain if domain is not None else geometry.domain
        self.resolution = self.domain._shape(resolution)
        self.periodic = self.domain.periodic
        self.h = self.domain.spacing(self.resolution)
        pts = self.domain.grid(self.resolution)
        with torch.no_grad():
            g = geometry.g(pts)
            self.ginv = torch.linalg.inv(g)
            self.sqrtg = torch.sqrt(torch.linalg.det(g))
            self.A = self.sqrtg[..., None, None] * self.ginv  # |g|^{1/2} g^{ij}

    def __call__(self, field: Field) -> Field:
        if field.resolution != self.resolution:
            raise ValueError("field resolution does not match the operator's grid")
        return field.like(self.apply(field.values))

    def apply(self, f: Tensor) -> Tensor:
        n = len(self.resolution)
        res = self.resolution
        periodic = self.periodic
        h = self.h
        A = self.A
        lead = f.ndim - n
        central = [_central(f, lead + j, h[j].item(), periodic) for j in range(n)]
        result = torch.zeros_like(f)
        for i in range(n):
            ax = lead + i
            hi = h[i].item()
            if periodic:
                fwd = (torch.roll(f, -1, ax) - f) / hi

                def to_face(t, axis_offset=0, i=i):
                    a = axis_offset + i
                    return 0.5 * (torch.roll(t, -1, a) + t)
            else:
                Ni = res[i]
                fwd = (f.narrow(ax, 1, Ni - 1) - f.narrow(ax, 0, Ni - 1)) / hi

                def to_face(t, axis_offset=0, i=i):
                    a = axis_offset + i
                    m = t.shape[a]
                    return 0.5 * (t.narrow(a, 1, m - 1) + t.narrow(a, 0, m - 1))
            flux = torch.zeros_like(fwd)
            for j in range(n):
                coef = to_face(A[..., i, j])
                dj = fwd if j == i else to_face(central[j], lead)
                flux = flux + coef * dj
            if periodic:
                div_i = (flux - torch.roll(flux, 1, ax)) / hi
            else:
                pad = [0] * (2 * f.ndim)
                k = f.ndim - 1 - ax  # F.pad lists dims from the last one backwards
                pad[2 * k] = 1
                pad[2 * k + 1] = 1
                div_i = torch.diff(torch.nn.functional.pad(flux, pad), dim=ax) / hi
            result = result + div_i
        return result / self.sqrtg


def laplace_beltrami(field: Field, geometry) -> Field:
    r"""``Δ_g f = |g|^{-1/2} ∂_i(|g|^{1/2} g^{ij} ∂_j f)`` on the grid (conservative staggered scheme)."""
    return LaplaceBeltrami(geometry, field.resolution, field.domain)(field)


def structure_tensor(field: Field) -> Tensor:
    """``Σ_c ∂_i f_c ∂_j f_c`` with shape ``(*res, n, n)``."""
    df = gradient(field)
    if df.ndim == field.dim + 1:
        df = df.unsqueeze(0)
    df = df.reshape(-1, *field.resolution, field.dim)
    return torch.einsum("c...i,c...j->...ij", df, df)
