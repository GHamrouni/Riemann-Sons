r"""Differential operators for fields on a geometry (finite differences on the grid).

For a scalar field ``f`` on ``(M, g)``:

* Euclidean gradient       ``∂_i f``                  -- :func:`gradient`
* Riemannian gradient      ``(∇f)^i = g^{ij} ∂_j f``  -- :func:`riemannian_gradient`
* divergence               ``div V = |g|^{-1/2} ∂_i (|g|^{1/2} V^i)`` -- :func:`divergence`
* Laplace–Beltrami         ``Δ_g f = |g|^{-1/2} ∂_i (|g|^{1/2} g^{ij} ∂_j f)`` -- :func:`laplace_beltrami`

The Laplace–Beltrami operator is assembled as a Q1 finite-element operator
with lumped mass (see :class:`LaplaceBeltrami`): ``L = −M⁻¹K`` with ``K``
symmetric positive semi-definite, so ``M L`` is symmetric, constants are
annihilated, Riemannian mass is conserved, and the discrete heat flow
dissipates the Dirichlet energy.  Zero-flux (Neumann) boundaries on boxes,
or periodic wrap-around.  The operator is differentiable with respect to the
metric parameters.
"""

from __future__ import annotations

import contextlib
import itertools
import math

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
    r"""``Δ_g`` on a fixed grid, assembled as a **finite-element** operator with cached coefficients.

    Discretisation.  Multilinear (Q1) elements on the grid cells, the metric
    evaluated at cell centres, and a ``2^n``-point tensor quadrature give the
    stiffness matrix ``K = Σ_cells Σ_q w_q ∇φ_i(ξ_q) · (|g|^{1/2} g^{-1}) ∇φ_j(ξ_q)``;
    the mass matrix is lumped, ``M = diag(|g|^{1/2} Π h)`` at the nodes.
    ``quadrature="nodal"`` (default, corner/trapezoidal rule) reduces to the
    classical 5-point stencil when ``g = I`` and is the more accurate choice on
    our tests; ``"gauss"`` integrates the element integrands exactly.  Both are
    second-order accurate and yield the same structure: the operator is

    .. math::  L = -M^{-1} K,\qquad K = K^T \succeq 0,

    so it is *weighted self-adjoint* (``M L = (M L)^T``), negative semi-definite,
    annihilates constants, conserves the Riemannian mass ``Σ M f`` under
    ``∂_t f = L f``, is exact for quadratic ``f`` and constant ``g`` in the interior,
    and has no spurious checkerboard null mode.  Boundary faces carry zero flux
    (natural Neumann condition) unless the domain is periodic.

    Differentiability.  Unless ``detach=True`` the coefficients keep their autograd
    history, so a loss on a diffused field back-propagates to the metric
    parameters.  Build the operator once per forward pass and reuse it across
    time steps; :func:`flows.diffuse` does exactly that.
    """

    def __init__(self, geometry, resolution, domain=None, detach: bool = False, quadrature: str = "nodal"):
        self.domain = domain if domain is not None else geometry.domain
        self.quadrature = quadrature
        self.resolution = self.domain._shape(resolution)
        self.periodic = self.domain.periodic
        self.h = self.domain.spacing(self.resolution)
        n = self.n = len(self.resolution)
        res = self.resolution
        self.cell_shape = tuple(res) if self.periodic else tuple(r - 1 for r in res)
        lo = self.domain.lo
        # cell centres and node points
        idx = torch.stack(torch.meshgrid(*[torch.arange(m, dtype=lo.dtype, device=lo.device) for m in self.cell_shape], indexing="ij"), -1)
        centers = lo + (idx + 0.5) * self.h
        nodes = self.domain.grid(res)
        ctx = torch.no_grad() if detach else contextlib.nullcontext()
        with ctx:
            g_c = geometry.g(centers)
            A = torch.sqrt(torch.linalg.det(g_c))[..., None, None] * torch.linalg.inv(g_c)  # |g|^{1/2} g^{-1} per cell
            self.sqrtg = torch.sqrt(torch.linalg.det(geometry.g(nodes)))  # (*res)
        # reference element: corners in {0,1}^n, 2-point Gauss rule per axis
        self.offsets = list(itertools.product((0, 1), repeat=n))
        if quadrature == "gauss":  # exact for the bilinear element integrands
            xi_1d = torch.tensor([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)], dtype=lo.dtype, device=lo.device)
        elif quadrature == "nodal":  # trapezoidal rule at the corners: reduces to the 5-point stencil for g = I
            xi_1d = torch.tensor([0.0, 1.0], dtype=lo.dtype, device=lo.device)
        else:
            raise ValueError("quadrature must be 'gauss' or 'nodal'")
        gauss = list(itertools.product(range(2), repeat=n))
        B = torch.zeros(len(gauss), n, len(self.offsets), dtype=lo.dtype, device=lo.device)  # B[q, d, c] = ∂_d φ_c(ξ_q)
        for qi, q in enumerate(gauss):
            xi = xi_1d[list(q)]
            for ci, o in enumerate(self.offsets):
                for d in range(n):
                    val = (1.0 if o[d] == 1 else -1.0) / self.h[d]
                    for e in range(n):
                        if e != d:
                            val = val * (xi[e] if o[e] == 1 else 1 - xi[e])
                    B[qi, d, ci] = val
        w = torch.full((len(gauss),), float(torch.prod(self.h)) / len(gauss), dtype=lo.dtype, device=lo.device)
        # element stiffness K_e = Σ_q w_q B_qᵀ A B_q  -> (*cells, C, C)
        self.K = torch.einsum("q,qdc,...de,qef->...cf", w, B, A, B)
        self.mass = self.sqrtg * torch.prod(self.h)

    # ------------------------------------------------------------------ apply
    def __call__(self, field: Field) -> Field:
        if field.resolution != self.resolution:
            raise ValueError("field resolution does not match the operator's grid")
        return field.like(self.apply(field.values))

    def _corner(self, f: Tensor, o: tuple[int, ...], lead: int) -> Tensor:
        t = f
        for d in range(self.n):
            ax = lead + d
            t = torch.roll(t, -o[d], ax) if self.periodic else t.narrow(ax, o[d], self.resolution[d] - 1)
        return t

    def _scatter(self, y: Tensor, o: tuple[int, ...], lead: int) -> Tensor:
        if self.periodic:
            for d in range(self.n):
                y = torch.roll(y, o[d], lead + d)
            return y
        pad = [0] * (2 * y.ndim)
        for d in range(self.n):
            k = y.ndim - 1 - (lead + d)  # F.pad lists dims from the last one backwards
            pad[2 * k] = o[d]
            pad[2 * k + 1] = 1 - o[d]
        return torch.nn.functional.pad(y, pad)

    def stiffness_apply(self, f: Tensor) -> Tensor:
        """``K f`` with shape ``(*channels, *res)``."""
        lead = f.ndim - self.n
        F = torch.stack([self._corner(f, o, lead) for o in self.offsets], dim=-1)  # (*ch, *cells, C)
        Y = (self.K @ F.unsqueeze(-1)).squeeze(-1)  # K broadcasts over the channel axes
        out = torch.zeros_like(f)
        for ci, o in enumerate(self.offsets):
            out = out + self._scatter(Y[..., ci], o, lead)
        return out

    def apply(self, f: Tensor) -> Tensor:
        """``Δ_g f = −M^{-1} K f``."""
        return -self.stiffness_apply(f) / self.mass

    def energy(self, f: Tensor) -> Tensor:
        """Dirichlet energy ``½ fᵀ K f = ½ ∫ |∇f|²_g √|g| dx`` (per channel)."""
        return 0.5 * (f * self.stiffness_apply(f)).sum(dim=tuple(range(-self.n, 0)))

    def matrix(self) -> Tensor:
        """Dense ``L`` as an ``(N, N)`` matrix -- for small grids and tests."""
        N = math.prod(self.resolution)
        eye = torch.eye(N, dtype=self.mass.dtype, device=self.mass.device)
        cols = [self.apply(eye[i].reshape(self.resolution)).reshape(-1) for i in range(N)]
        return torch.stack(cols, dim=1)


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
