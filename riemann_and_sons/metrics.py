"""Riemannian metrics ``g(x)`` and safe SPD parameterizations.

A :class:`Metric` is a ``torch.nn.Module`` mapping points ``x`` of shape
``(..., n)`` to symmetric positive-definite matrices of shape ``(..., n, n)``.
Everything downstream (Christoffel symbols, curvature, geodesics, operators)
is derived from this single callable with autograd, so any metric written
here is automatically differentiable with respect to its parameters.

Metrics available
-----------------
* :class:`EuclideanMetric`      -- ``g = I``.
* :class:`ConstantMetric`       -- ``g = A`` (global Mahalanobis).
* :class:`ConformalMetric`      -- ``g = λ(x) I`` from a callable ``λ``.
* :class:`FunctionMetric`       -- arbitrary analytical ``g(x)``.
* :class:`GridMetric`           -- learnable SPD field stored on a grid and
  interpolated with cubic B-splines (the workhorse for 2-D experiments).
* :class:`PullbackMetric`       -- ``g = J_F^T J_F + εI`` for a map ``F``.
* :class:`DiagonalLowRankMetric`-- ``g = D(x) + U(x)U(x)^T`` from callables.
* :func:`image_induced_metric`  -- structure-tensor metric of an image.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn

from . import curvature as _curv
from ._interp import bspline_prefilter, interpolate
from .domains import Box

__all__ = [
    "Metric",
    "EuclideanMetric",
    "ConstantMetric",
    "ConformalMetric",
    "FunctionMetric",
    "GridMetric",
    "PullbackMetric",
    "DiagonalLowRankMetric",
    "SPDParameterization",
    "CholeskyParameterization",
    "LogEuclideanParameterization",
    "DiagonalParameterization",
    "ConformalParameterization",
    "DiagonalLowRankParameterization",
    "image_induced_metric",
]


def _eye_like(x: Tensor, n: int) -> Tensor:
    return torch.eye(n, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], n, n)


# ---------------------------------------------------------------------------
# SPD parameterizations
# ---------------------------------------------------------------------------
class SPDParameterization(nn.Module):
    """Map an unconstrained vector of size ``n_params(n)`` to an SPD ``n × n`` matrix."""

    def n_params(self, n: int) -> int:
        raise NotImplementedError

    def forward(self, raw: Tensor, n: int) -> Tensor:
        raise NotImplementedError

    def inverse(self, g: Tensor) -> Tensor:
        """Best-effort inverse map SPD -> raw (used to initialise from a known metric)."""
        raise NotImplementedError(f"{type(self).__name__} has no inverse map")

    def identity_raw(self, n: int, dtype=None, device=None) -> Tensor:
        return self.inverse(torch.eye(n, dtype=dtype, device=device))


class CholeskyParameterization(SPDParameterization):
    r"""``g = L L^T + ε I`` with ``L`` lower-triangular and ``diag(L) = exp(raw_diag)``."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def n_params(self, n: int) -> int:
        return n * (n + 1) // 2

    def forward(self, raw: Tensor, n: int) -> Tensor:
        rows, cols = torch.tril_indices(n, n, device=raw.device)
        L = raw.new_zeros(*raw.shape[:-1], n, n)
        L[..., rows, cols] = raw
        diag = torch.exp(torch.diagonal(L, dim1=-2, dim2=-1))
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return L @ L.transpose(-1, -2) + self.eps * torch.eye(n, dtype=raw.dtype, device=raw.device)

    def inverse(self, g: Tensor) -> Tensor:
        n = g.shape[-1]
        eye = torch.eye(n, dtype=g.dtype, device=g.device)
        L = torch.linalg.cholesky(g - self.eps * eye)
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(
            torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
        )
        rows, cols = torch.tril_indices(n, n, device=g.device)
        return L[..., rows, cols]


class LogEuclideanParameterization(SPDParameterization):
    r"""``g = expm(S)`` with ``S`` symmetric (log-Euclidean coordinates)."""

    def n_params(self, n: int) -> int:
        return n * (n + 1) // 2

    def forward(self, raw: Tensor, n: int) -> Tensor:
        rows, cols = torch.tril_indices(n, n, device=raw.device)
        S = raw.new_zeros(*raw.shape[:-1], n, n)
        S[..., rows, cols] = raw
        S = S + S.transpose(-1, -2) - torch.diag_embed(torch.diagonal(S, dim1=-2, dim2=-1))
        return torch.linalg.matrix_exp(S)

    def inverse(self, g: Tensor) -> Tensor:
        n = g.shape[-1]
        evals, evecs = torch.linalg.eigh(g)
        S = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.transpose(-1, -2)
        rows, cols = torch.tril_indices(n, n, device=g.device)
        return S[..., rows, cols]


class DiagonalParameterization(SPDParameterization):
    """``g = diag(exp(raw))``."""

    def n_params(self, n: int) -> int:
        return n

    def forward(self, raw: Tensor, n: int) -> Tensor:
        return torch.diag_embed(torch.exp(raw))

    def inverse(self, g: Tensor) -> Tensor:
        return torch.log(torch.diagonal(g, dim1=-2, dim2=-1))


class ConformalParameterization(SPDParameterization):
    """``g = exp(2u) I`` -- a single scalar ``u`` per point."""

    def n_params(self, n: int) -> int:
        return 1

    def forward(self, raw: Tensor, n: int) -> Tensor:
        scale = torch.exp(2 * raw[..., 0])
        return scale[..., None, None] * torch.eye(n, dtype=raw.dtype, device=raw.device)

    def inverse(self, g: Tensor) -> Tensor:
        n = g.shape[-1]
        return 0.5 * torch.log(torch.diagonal(g, dim1=-2, dim2=-1).mean(-1, keepdim=True))


class DiagonalLowRankParameterization(SPDParameterization):
    r"""``g = diag(exp(d)) + U U^T`` with ``U ∈ R^{n×r}`` -- ``n + n r`` parameters.

    Intended for high-dimensional spaces where a full ``n × n`` field is too
    expensive.  The inverse map is approximate (top-``r`` eigen-directions).
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def n_params(self, n: int) -> int:
        return n + n * self.rank

    def forward(self, raw: Tensor, n: int) -> Tensor:
        d = torch.exp(raw[..., :n])
        U = raw[..., n:].reshape(*raw.shape[:-1], n, self.rank)
        return torch.diag_embed(d) + U @ U.transpose(-1, -2)

    def inverse(self, g: Tensor) -> Tensor:
        n = g.shape[-1]
        evals, evecs = torch.linalg.eigh(g)
        top_vals = evals[..., -self.rank :]
        top_vecs = evecs[..., :, -self.rank :]
        rest = evals[..., : n - self.rank].mean(-1, keepdim=True).expand(*g.shape[:-2], n)
        U = top_vecs * torch.sqrt((top_vals - rest[..., :1]).clamp_min(1e-8))[..., None, :]
        return torch.cat([torch.log(rest), U.reshape(*g.shape[:-2], -1)], dim=-1)


# ---------------------------------------------------------------------------
# Metric base class
# ---------------------------------------------------------------------------
class Metric(nn.Module):
    """Base class: ``forward(x)`` returns ``g(x)`` with shape ``(..., n, n)``.

    Subclasses must broadcast over arbitrary leading batch dimensions *and*
    accept a single point of shape ``(n,)`` -- the latter is what
    ``torch.func`` transforms feed in when differentiating.
    """

    dim: int | None = None

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- algebraic quantities -------------------------------------------
    def inverse(self, x: Tensor) -> Tensor:
        return torch.linalg.inv(self(x))

    def det(self, x: Tensor) -> Tensor:
        return torch.linalg.det(self(x))

    def volume_element(self, x: Tensor) -> Tensor:
        r"""``√|g(x)|``."""
        return torch.sqrt(self.det(x))

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return torch.einsum("...i,...ij,...j->...", u, self(x), v)

    def quadratic_form(self, x: Tensor, v: Tensor) -> Tensor:
        """``vᵀ g(x) v``.  Structured metrics override this to avoid materialising ``g``."""
        return self.inner(x, v, v)

    def norm(self, x: Tensor, v: Tensor) -> Tensor:
        return torch.sqrt(self.quadratic_form(x, v).clamp_min(0))

    def logdet(self, x: Tensor) -> Tensor:
        return torch.logdet(self(x))

    def eigenvalues(self, x: Tensor) -> Tensor:
        return torch.linalg.eigvalsh(self(x))

    def condition_number(self, x: Tensor) -> Tensor:
        ev = self.eigenvalues(x)
        return ev[..., -1] / ev[..., 0]

    def anisotropy(self, x: Tensor) -> Tensor:
        """Same as :meth:`condition_number`: ``λ_max / λ_min``."""
        return self.condition_number(x)

    # ---- differential quantities (autograd through ``curvature``) -------
    def christoffel(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return _curv.christoffel(self, x, chunk=chunk)

    def riemann(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return _curv.riemann(self, x, chunk=chunk)

    def ricci(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return _curv.ricci(self, x, chunk=chunk)

    def scalar_curvature(self, x: Tensor, chunk: int | None = None) -> Tensor:
        return _curv.scalar_curvature(self, x, chunk=chunk)

    def gauss_curvature(self, x: Tensor, chunk: int | None = None) -> Tensor:
        """``K = R / 2`` (only meaningful in two dimensions)."""
        return 0.5 * self.scalar_curvature(x, chunk=chunk)

    def metric_gradient(self, x: Tensor, chunk: int | None = None) -> Tensor:
        """``∂_k g_ij`` with shape ``(..., n, n, n)`` indexed ``[i, j, k]``."""
        return _curv.metric_derivative(self, x, chunk=chunk)

    # ---- misc ----------------------------------------------------------------
    def _check_dim(self, x: Tensor) -> None:
        if self.dim is not None and x.shape[-1] != self.dim:
            raise ValueError(f"metric expects points of dimension {self.dim}, got {x.shape[-1]}")


# ---------------------------------------------------------------------------
# Concrete metrics
# ---------------------------------------------------------------------------
class EuclideanMetric(Metric):
    """``g(x) = I``."""

    def forward(self, x: Tensor) -> Tensor:
        return _eye_like(x, x.shape[-1])


class ConstantMetric(Metric):
    """``g(x) = A`` for a fixed SPD matrix ``A`` (global Mahalanobis geometry)."""

    def __init__(self, A: Tensor, learnable: bool = False, parameterization: SPDParameterization | None = None):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.get_default_dtype())
        self.dim = A.shape[-1]
        self.param = parameterization or CholeskyParameterization()
        raw = self.param.inverse(A)
        self.raw = nn.Parameter(raw, requires_grad=learnable)

    @property
    def matrix(self) -> Tensor:
        return self.param(self.raw, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        return self.matrix.expand(*x.shape[:-1], self.dim, self.dim)


class ConformalMetric(Metric):
    """``g(x) = λ(x) I`` for a positive callable ``λ : (..., n) -> (...)``."""

    def __init__(self, scale: Callable[[Tensor], Tensor], dim: int | None = None):
        super().__init__()
        self.scale = scale
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        lam = self.scale(x)
        return lam[..., None, None] * _eye_like(x, x.shape[-1])


class FunctionMetric(Metric):
    """Wrap an arbitrary callable ``g : (..., n) -> (..., n, n)``."""

    def __init__(self, fn: Callable[[Tensor], Tensor], dim: int | None = None):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        return self.fn(x)


class GridMetric(Metric):
    """Learnable SPD field stored on a regular grid.

    Raw (unconstrained) parameters live at the grid nodes; they are interpolated
    with a cubic B-spline and mapped through an :class:`SPDParameterization`
    *after* interpolation, so ``g(x)`` is SPD everywhere and ``C^2`` in ``x``.

    Parameters
    ----------
    domain : Box
    resolution : int or tuple -- number of nodes per axis.
    parameterization : how raw parameters become SPD matrices (default Cholesky).
    mode : ``"cubic"`` (C^2, default) or ``"linear"`` (C^0, no curvature).
    """

    def __init__(
        self,
        domain: Box,
        resolution,
        parameterization: SPDParameterization | None = None,
        mode: str = "cubic",
        init: str | Tensor = "euclidean",
    ):
        super().__init__()
        self.domain = domain
        self.dim = domain.dim
        self.resolution = domain._shape(resolution)
        self.param = parameterization or CholeskyParameterization()
        self.mode = mode
        self.periodic = domain.periodic
        n_p = self.param.n_params(self.dim)
        if isinstance(init, str):
            if init != "euclidean":
                raise ValueError("init must be 'euclidean' or a tensor of node metrics")
            base = self.param.identity_raw(self.dim, dtype=domain.lo.dtype, device=domain.lo.device)
            raw = base.expand(*self.resolution, n_p).clone()
        else:
            raw = self._encode(init)
        self.raw = nn.Parameter(raw)

    # ------------------------------------------------------------------ helpers
    def _encode(self, g_nodes: Tensor) -> Tensor:
        raw = self.param.inverse(g_nodes)
        if self.mode == "cubic":
            raw = bspline_prefilter(raw, self.periodic)
        return raw

    @classmethod
    def from_tensor_field(cls, domain: Box, g_nodes: Tensor, parameterization=None, mode: str = "cubic") -> "GridMetric":
        """Build a grid metric that interpolates the SPD matrices ``g_nodes`` ``(*res, n, n)``."""
        return cls(domain, tuple(g_nodes.shape[:-2]), parameterization, mode, init=g_nodes)

    @classmethod
    def from_function(cls, domain: Box, resolution, fn: Callable[[Tensor], Tensor], parameterization=None, mode: str = "cubic") -> "GridMetric":
        pts = domain.grid(resolution)
        with torch.no_grad():
            g = fn(pts)
        return cls.from_tensor_field(domain, g, parameterization, mode)

    @property
    def nodes(self) -> Tensor:
        return self.domain.grid(self.resolution)

    def node_metric(self) -> Tensor:
        """``g`` evaluated at the grid nodes, shape ``(*res, n, n)``."""
        return self(self.nodes)

    @torch.no_grad()
    def set_node_metric(self, g_nodes: Tensor) -> None:
        self.raw.copy_(self._encode(g_nodes))

    def raw_at(self, x: Tensor) -> Tensor:
        return interpolate(self.raw, x, self.domain.lo, self.domain.hi, self.mode, self.periodic)

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        return self.param(self.raw_at(x), self.dim)

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, param={type(self.param).__name__}, mode={self.mode}"


class PullbackMetric(Metric):
    r"""``g(x) = J_F(x)^T J_F(x) + ε I`` for a differentiable map ``F : R^n -> R^k``.

    With ``ε = 0`` this is a true Riemannian metric only where ``J_F`` has full
    column rank (``k ≥ n`` and injective differential); otherwise it is merely
    positive semi-definite.  The ``ε I`` term restores definiteness.
    """

    def __init__(self, F: Callable[[Tensor], Tensor] | nn.Module, eps: float = 1e-3, dim: int | None = None):
        super().__init__()
        self.F = F
        self.eps = eps
        self.dim = dim

    def jacobian(self, x: Tensor) -> Tensor:
        single = torch.func.jacfwd(self.F)
        batch = x.shape[:-1]
        J = torch.func.vmap(single)(x.reshape(-1, x.shape[-1]))
        return J.reshape(*batch, *J.shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        J = self.jacobian(x)
        return J.transpose(-1, -2) @ J + self.eps * _eye_like(x, x.shape[-1])


class DiagonalLowRankMetric(Metric):
    r"""``g(x) = D(x) + U(x) U(x)^T`` with callables ``D : x -> (..., n)`` (positive) and ``U : x -> (..., n, r)``."""

    def __init__(self, diag_fn: Callable[[Tensor], Tensor], lowrank_fn: Callable[[Tensor], Tensor], dim: int | None = None):
        super().__init__()
        self.diag_fn = diag_fn
        self.lowrank_fn = lowrank_fn
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        U = self.lowrank_fn(x)
        return torch.diag_embed(self.diag_fn(x)) + U @ U.transpose(-1, -2)

    def quadratic_form(self, x: Tensor, v: Tensor) -> Tensor:
        """``Σ_i D_i v_i² + ‖Uᵀ v‖²`` in ``O(n r)`` without forming the ``n × n`` matrix."""
        U = self.lowrank_fn(x)
        return (self.diag_fn(x) * v * v).sum(-1) + (torch.einsum("...ir,...i->...r", U, v) ** 2).sum(-1)


# ---------------------------------------------------------------------------
# Image-induced metric
# ---------------------------------------------------------------------------
def image_induced_metric(image, lam: float = 10.0, sigma: float = 1.0, structure_sigma: float = 0.0, mode: str = "cubic") -> GridMetric:
    r"""Structure-tensor metric of an image (Demo B).

    .. math::

        g_I(x) = I + λ \, ∇I_σ(x) ∇I_σ(x)^T,

    where ``I_σ`` is the image blurred with a Gaussian of width ``sigma`` (in
    grid cells) and, for multi-channel images, the outer products are summed
    over channels.  If ``structure_sigma > 0`` the structure tensor itself is
    additionally blurred.  Moving *across* an edge (along ``∇I``) costs
    ``√(1 + λ|∇I|²)`` per unit length; moving *along* the edge costs ``1``.

    Returns a :class:`GridMetric` whose nodes coincide with the image pixels.
    """
    from .fields import Field, gaussian_blur
    from .operators import gradient

    field: Field = image
    smooth = gaussian_blur(field, sigma) if sigma > 0 else field
    grad = gradient(smooth)  # (C, *res, n)
    if grad.ndim == field.dim + 1:
        grad = grad.unsqueeze(0)
    struct = torch.einsum("c...i,c...j->...ij", grad, grad)
    if structure_sigma > 0:
        n = field.dim
        flat = Field(struct.reshape(*struct.shape[:-2], n * n).movedim(-1, 0), field.domain)
        struct = gaussian_blur(flat, structure_sigma).values.movedim(0, -1).reshape(*struct.shape)
    eye = torch.eye(field.dim, dtype=struct.dtype, device=struct.device)
    g_nodes = eye + lam * struct
    return GridMetric.from_tensor_field(field.domain, g_nodes, mode=mode)
