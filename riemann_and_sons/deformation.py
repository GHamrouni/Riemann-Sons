r"""Transformations ``Φ : M → M``, their Jacobians, image warping and fitting energies.

Deformation is *one application* of geometry: a transformation is a smooth
map of the coordinate patch, its Jacobian ``J_Φ = ∂Φ/∂x`` is first-class, and
an image is warped by inverse sampling ``I'(x) = I(Φ^{-1}(x))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import jacfwd, vmap

from ._interp import interpolate
from .domains import Box
from .fields import Field

__all__ = [
    "Transformation",
    "Identity",
    "Affine",
    "Displacement",
    "ThinPlateSpline",
    "Composed",
    "FunctionTransformation",
    "JacobianStats",
    "jacobian_stats",
    "warp",
    "bending_energy",
    "fold_penalty",
    "landmark_loss",
    "fit_transformation",
]


class Transformation(nn.Module):
    """Base class.  ``forward(x)`` maps ``(..., n) -> (..., n)`` and must accept a single point."""

    dim: int | None = None

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, x: Tensor, chunk: int | None = None) -> Tensor:
        """``J_Φ(x)`` of shape ``(..., n, n)`` with ``J[i, j] = ∂Φ^i/∂x^j``."""
        n = x.shape[-1]
        flat = x.reshape(-1, n)
        fn = jacfwd(self.forward)
        if chunk is None or flat.shape[0] <= chunk:
            J = vmap(fn)(flat)
        else:
            J = torch.cat([vmap(fn)(part) for part in flat.split(chunk)])
        return J.reshape(*x.shape[:-1], n, n)

    def inverse(self, y: Tensor, iterations: int = 30, tol: float = 1e-10, x0: Tensor | None = None) -> Tensor:
        """Solve ``Φ(x) = y`` by damped Newton iteration (batched)."""
        x = y.clone() if x0 is None else x0.clone()
        for _ in range(iterations):
            r = self(x) - y
            if r.abs().max() < tol:
                break
            J = self.jacobian(x)
            step = torch.linalg.solve(J, r.unsqueeze(-1)).squeeze(-1)
            x = x - step
        return x

    def compose(self, inner: "Transformation") -> "Composed":
        """``self ∘ inner``."""
        return Composed(self, inner)

    def __matmul__(self, inner: "Transformation") -> "Composed":
        return self.compose(inner)


class Identity(Transformation):
    def forward(self, x: Tensor) -> Tensor:
        return x


class FunctionTransformation(Transformation):
    def __init__(self, fn: Callable[[Tensor], Tensor], dim: int | None = None):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class Affine(Transformation):
    """``Φ(x) = A x + b``."""

    def __init__(self, A: Tensor, b: Tensor | None = None, learnable: bool = False):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.get_default_dtype())
        b = torch.zeros(A.shape[0], dtype=A.dtype) if b is None else torch.as_tensor(b, dtype=A.dtype)
        self.A = nn.Parameter(A, requires_grad=learnable)
        self.b = nn.Parameter(b, requires_grad=learnable)
        self.dim = A.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.A.transpose(-1, -2) + self.b

    def inverse(self, y: Tensor, **kw) -> Tensor:
        return (y - self.b) @ torch.linalg.inv(self.A).transpose(-1, -2)


class Composed(Transformation):
    def __init__(self, outer: Transformation, inner: Transformation):
        super().__init__()
        self.outer = outer
        self.inner = inner
        self.dim = outer.dim or inner.dim

    def forward(self, x: Tensor) -> Tensor:
        return self.outer(self.inner(x))

    def inverse(self, y: Tensor, **kw) -> Tensor:
        return self.inner.inverse(self.outer.inverse(y, **kw), **kw)


class Displacement(Transformation):
    """``Φ(x) = x + u(x)`` with the displacement ``u`` stored on a grid (cubic B-spline by default)."""

    def __init__(self, domain: Box, resolution, mode: str = "cubic", init: Tensor | None = None):
        super().__init__()
        self.domain = domain
        self.dim = domain.dim
        self.resolution = domain._shape(resolution)
        self.mode = mode
        raw = torch.zeros(*self.resolution, self.dim, dtype=domain.lo.dtype, device=domain.lo.device) if init is None else init.clone()
        self.raw = nn.Parameter(raw)

    def displacement(self, x: Tensor) -> Tensor:
        return interpolate(self.raw, x, self.domain.lo, self.domain.hi, self.mode, self.domain.periodic)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.displacement(x)

    def inverse(self, y: Tensor, iterations: int = 50, tol: float = 1e-10, x0: Tensor | None = None) -> Tensor:
        """Fixed-point iteration ``x ← y − u(x)`` (converges for ``‖∇u‖ < 1``), polished with Newton."""
        x = y.clone() if x0 is None else x0.clone()
        with torch.no_grad():
            for _ in range(iterations):
                x_new = y - self.displacement(x)
                if (x_new - x).abs().max() < tol:
                    x = x_new
                    break
                x = x_new
        return Transformation.inverse(self, y, iterations=5, tol=tol, x0=x)

    @property
    def nodes(self) -> Tensor:
        return self.domain.grid(self.resolution)


class ThinPlateSpline(Transformation):
    r"""Thin-plate spline interpolating ``sources → targets`` (2-D kernel ``r² log r``, 3-D kernel ``r``).

    Minimises the bending energy ``∫ ‖∇² Φ‖²`` among all interpolants; ``reg > 0``
    trades exact interpolation for smoothness.
    """

    def __init__(self, sources: Tensor, targets: Tensor, reg: float = 0.0):
        super().__init__()
        sources = torch.as_tensor(sources, dtype=torch.get_default_dtype())
        targets = torch.as_tensor(targets, dtype=sources.dtype)
        n, k = sources.shape[1], sources.shape[0]
        self.dim = n
        K = self._kernel(((sources[:, None, :] - sources[None, :, :]) ** 2).sum(-1))
        P = torch.cat([torch.ones(k, 1, dtype=sources.dtype), sources], dim=1)
        L = torch.zeros(k + n + 1, k + n + 1, dtype=sources.dtype)
        L[:k, :k] = K + reg * torch.eye(k, dtype=sources.dtype)
        L[:k, k:] = P
        L[k:, :k] = P.T
        rhs = torch.cat([targets, torch.zeros(n + 1, n, dtype=sources.dtype)], dim=0)
        sol = torch.linalg.solve(L, rhs)
        self.register_buffer("sources", sources)
        self.register_buffer("W", sol[:k])
        self.register_buffer("A", sol[k:])  # (n+1, n): affine part [b; M^T]

    def _kernel(self, r2: Tensor) -> Tensor:
        """Radial basis as a function of the *squared* distance (forward-AD friendly)."""
        if self.dim == 2:
            return 0.5 * r2 * torch.log(r2.clamp_min(1e-300))  # r² log r
        return torch.sqrt(r2 + 1e-300)  # r

    def forward(self, x: Tensor) -> Tensor:
        r2 = ((x[..., None, :] - self.sources) ** 2).sum(-1)  # (..., k)
        U = self._kernel(r2)
        affine = self.A[0] + x @ self.A[1:]
        return affine + U @ self.W


# ---------------------------------------------------------------------------
# Jacobian analysis
# ---------------------------------------------------------------------------
@dataclass
class JacobianStats:
    jacobian: Tensor  # (..., n, n)
    det: Tensor  # (...)
    singular_values: Tensor  # (..., n) descending
    condition_number: Tensor  # (...)
    orientation_reversed: Tensor  # bool (...)
    volume_change: Tensor  # |det|

    @property
    def fold_fraction(self) -> float:
        return float(self.orientation_reversed.double().mean())

    @property
    def has_folds(self) -> bool:
        return bool(self.orientation_reversed.any())

    def summary(self) -> dict[str, float]:
        det = self.det.detach()
        return {
            "det_min": float(det.min()),
            "det_max": float(det.max()),
            "cond_max": float(self.condition_number.detach().max()),
            "fold_fraction": self.fold_fraction,
        }


def jacobian_stats(transformation: Transformation, points: Tensor, chunk: int | None = 4096) -> JacobianStats:
    J = transformation.jacobian(points, chunk=chunk)
    det = torch.linalg.det(J)
    sv = torch.linalg.svdvals(J)
    return JacobianStats(J, det, sv, sv[..., 0] / sv[..., -1].clamp_min(1e-300), det <= 0, det.abs())


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------
def warp(
    image: Field,
    transformation: Transformation,
    inverse: Transformation | None = None,
    resolution=None,
    mode: str = "linear",
    fill: float | None = None,
    **inverse_kw,
) -> Field:
    r"""Push an image forward through ``Φ`` by inverse sampling ``I'(x) = I(Φ^{-1}(x))``.

    ``inverse`` may be supplied when ``Φ^{-1}`` is known analytically; otherwise
    it is computed numerically (Newton / fixed point).  Points that land outside
    the image domain are filled with ``fill`` (or the clamped boundary value).
    """
    dom = image.domain
    res = image.resolution if resolution is None else dom._shape(resolution)
    x = dom.grid(res)
    with torch.no_grad():
        src = inverse(x) if inverse is not None else transformation.inverse(x, **inverse_kw)
    vals = image.sample(src, mode=mode)  # (*res, *channels)
    if vals.ndim > image.dim:
        vals = vals.movedim(-1, 0) if len(image.channels) == 1 else vals.permute(*range(image.dim, vals.ndim), *range(image.dim))
    if fill is not None:
        inside = dom.contains(src, tol=1e-9)
        vals = torch.where(inside, vals, torch.full_like(vals, fill))
    return image.like(vals)


# ---------------------------------------------------------------------------
# Energies & fitting
# ---------------------------------------------------------------------------
def bending_energy(displacement: Displacement) -> Tensor:
    """Discrete ``∫ ‖∇² u‖² dx`` over the displacement's control grid."""
    u = displacement.raw  # (*res, n)
    h = displacement.domain.spacing(displacement.resolution)
    n = displacement.dim
    total = u.new_zeros(())
    for i in range(n):
        for j in range(n):
            d = u
            if i == j:
                d = (d.narrow(i, 2, d.shape[i] - 2) - 2 * d.narrow(i, 1, d.shape[i] - 2) + d.narrow(i, 0, d.shape[i] - 2)) / h[i] ** 2
            else:
                d = (d.narrow(i, 1, d.shape[i] - 1) - d.narrow(i, 0, d.shape[i] - 1)) / h[i]
                d = (d.narrow(j, 1, d.shape[j] - 1) - d.narrow(j, 0, d.shape[j] - 1)) / h[j]
            total = total + (d**2).sum()
    return total * torch.prod(h)


def fold_penalty(transformation: Transformation, points: Tensor, margin: float = 0.1) -> Tensor:
    """Penalise ``det J ≤ margin`` (squared hinge)."""
    det = torch.linalg.det(transformation.jacobian(points))
    return torch.relu(margin - det).pow(2).mean()


def landmark_loss(transformation: Transformation, sources: Tensor, targets: Tensor, geometry=None) -> Tensor:
    """Mean squared landmark error; with a geometry, uses the metric at the target (first-order Riemannian distance)."""
    diff = transformation(sources) - targets
    if geometry is None:
        return (diff**2).sum(-1).mean()
    return geometry.inner(targets, diff, diff).mean()


def fit_transformation(
    transformation: Transformation,
    sources: Tensor,
    targets: Tensor,
    bending: float = 1e-3,
    fold: float = 1.0,
    fold_margin: float = 0.1,
    iterations: int = 300,
    lr: float = 0.02,
    sample_points: Tensor | None = None,
    geometry=None,
    callback: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Fit a learnable transformation to landmark constraints with smoothness and fold penalties (Adam)."""
    params = [p for p in transformation.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for it in range(iterations):
        opt.zero_grad()
        terms = {"landmarks": landmark_loss(transformation, sources, targets, geometry)}
        if bending > 0 and isinstance(transformation, Displacement):
            terms["bending"] = bending * bending_energy(transformation)
        if fold > 0 and sample_points is not None:
            terms["fold"] = fold * fold_penalty(transformation, sample_points, fold_margin)
        loss = sum(terms.values())
        loss.backward()
        opt.step()
        rec = {k: float(v.detach()) for k, v in terms.items()}
        rec["total"] = float(loss.detach())
        history.append(rec)
        if callback is not None:
            callback(it, rec)
    return history
