"""Christoffel symbols and curvature tensors by automatic differentiation.

All functions take a *metric callable* ``metric(x) -> g`` accepting a single
point ``(n,)`` and return batched tensors for ``x`` of shape ``(..., n)``.
Derivatives are computed with ``torch.func.jacfwd`` (forward mode), nested
once for curvature, and batched with ``torch.func.vmap``.  Because these are
ordinary autograd operations, gradients also flow back to metric parameters.

Index conventions
-----------------
* ``metric_derivative``: ``dg[..., i, j, k] = ∂_k g_ij``
* ``christoffel``:       ``Γ[..., k, i, j] = Γ^k_ij``
* ``riemann``:           ``R[..., ρ, σ, μ, ν] = R^ρ_{σμν}``
  with ``R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}``
* ``ricci``:             ``Ric_{σν} = R^ρ_{σρν}``
* ``scalar_curvature``:  ``R = g^{σν} Ric_{σν}``  (sphere of radius 1: ``R = +2``).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch.func import jacfwd, vmap

__all__ = [
    "metric_derivative",
    "christoffel",
    "riemann",
    "ricci",
    "scalar_curvature",
    "gauss_curvature",
    "batched",
]


def batched(fn: Callable[[Tensor], Tensor], x: Tensor, chunk: int | None = None) -> Tensor:
    """Apply a single-point function to ``x`` of shape ``(..., n)`` with ``vmap``, optionally in chunks."""
    n = x.shape[-1]
    batch = x.shape[:-1]
    flat = x.reshape(-1, n)
    # PyTorch's chunked vmap cannot infer an output from zero chunks.
    out = vmap(fn, chunk_size=chunk if flat.shape[0] else None)(flat)
    return out.reshape(*batch, *out.shape[1:])


# ---------------------------------------------------------------- single point
def _metric_derivative_single(metric, p: Tensor) -> Tensor:
    return jacfwd(metric)(p)  # (n, n, n): [i, j, k] = ∂_k g_ij


def _christoffel_single(metric, p: Tensor) -> Tensor:
    g = metric(p)
    dg = jacfwd(metric)(p)  # dg[a, b, c] = ∂_c g_ab
    ginv = torch.linalg.inv(g)
    # Γ^k_ij = ½ g^{kl} (∂_i g_lj + ∂_j g_li − ∂_l g_ij)
    a = dg.permute(0, 2, 1)  # a[l, i, j] = ∂_i g_lj
    b = dg  # b[l, i, j] = ∂_j g_li
    c = dg.permute(2, 0, 1)  # c[l, i, j] = ∂_l g_ij
    return 0.5 * torch.einsum("kl,lij->kij", ginv, a + b - c)


def _riemann_single(metric, p: Tensor) -> Tensor:
    def gamma_with_aux(q):
        G = _christoffel_single(metric, q)
        return G, G

    dG, G = jacfwd(gamma_with_aux, has_aux=True)(p)  # dG[k, i, j, m] = ∂_m Γ^k_ij
    t1 = dG.permute(0, 2, 3, 1)  # [ρ, σ, μ, ν] = ∂_μ Γ^ρ_νσ
    t2 = dG.permute(0, 2, 1, 3)  # [ρ, σ, μ, ν] = ∂_ν Γ^ρ_μσ
    t3 = torch.einsum("rml,lns->rsmn", G, G)  # Γ^ρ_μλ Γ^λ_νσ
    t4 = torch.einsum("rnl,lms->rsmn", G, G)  # Γ^ρ_νλ Γ^λ_μσ
    return t1 - t2 + t3 - t4


def _ricci_single(metric, p: Tensor) -> Tensor:
    R = _riemann_single(metric, p)
    return torch.einsum("rsrn->sn", R)


def _scalar_single(metric, p: Tensor) -> Tensor:
    ginv = torch.linalg.inv(metric(p))
    return torch.einsum("sn,sn->", ginv, _ricci_single(metric, p))


# -------------------------------------------------------------------- batched
def metric_derivative(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """``∂_k g_ij`` as ``(..., n, n, n)`` indexed ``[i, j, k]``."""
    return batched(lambda p: _metric_derivative_single(metric, p), x, chunk)


def christoffel(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Christoffel symbols of the second kind, ``(..., n, n, n)`` indexed ``[k, i, j]``."""
    return batched(lambda p: _christoffel_single(metric, p), x, chunk)


def riemann(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Riemann tensor ``R^ρ_{σμν}`` as ``(..., n, n, n, n)``."""
    return batched(lambda p: _riemann_single(metric, p), x, chunk)


def ricci(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Ricci tensor ``Ric_{σν}`` as ``(..., n, n)``."""
    return batched(lambda p: _ricci_single(metric, p), x, chunk)


def scalar_curvature(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Scalar curvature ``R`` as ``(...)``."""
    return batched(lambda p: _scalar_single(metric, p), x, chunk)


def gauss_curvature(metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """``K = R/2`` -- the Gaussian curvature of a 2-D metric."""
    if x.shape[-1] != 2:
        raise ValueError("Gaussian curvature is only defined here for two-dimensional metrics")
    return 0.5 * scalar_curvature(metric, x, chunk)
