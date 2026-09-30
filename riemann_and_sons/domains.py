"""Domains: the coordinate patch ``M ⊂ R^n`` on which metrics and fields live."""

from __future__ import annotations

import operator

import torch
from torch import Tensor

from ._interp import grid_points, grid_spacing

__all__ = ["Box", "Plane"]


class Box:
    """Axis-aligned box ``[lo_0, hi_0] × ... × [lo_{n-1}, hi_{n-1}]``.

    Coordinate ``k`` of a point corresponds to array axis ``k`` of any grid
    built on the box (``ij`` indexing).  The 2-D plotting helpers take care of
    the transposition needed for ``imshow``.
    """

    def __init__(self, lo, hi, periodic: bool = False):
        lo = torch.as_tensor(lo)
        hi = torch.as_tensor(hi, device=lo.device)
        dtype = torch.promote_types(lo.dtype, hi.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        lo, hi = lo.to(dtype=dtype), hi.to(dtype=dtype)
        if lo.ndim == 0:
            lo = lo.reshape(1)
        if hi.ndim == 0:
            hi = hi.reshape(1)
        if lo.ndim != 1 or lo.numel() == 0 or lo.shape != hi.shape:
            raise ValueError("lo and hi must be nonempty vectors with the same shape")
        if not torch.all(torch.isfinite(lo) & torch.isfinite(hi) & (hi > lo)):
            raise ValueError("bounds must be finite and hi must exceed lo along every axis")
        self.lo = lo
        self.hi = hi
        self.periodic = periodic

    # ------------------------------------------------------------------ basics
    @property
    def dim(self) -> int:
        return self.lo.numel()

    @property
    def size(self) -> Tensor:
        return self.hi - self.lo

    @property
    def center(self) -> Tensor:
        return 0.5 * (self.lo + self.hi)

    @property
    def extent(self) -> list[float]:
        """``[x0_min, x0_max, x1_min, x1_max]`` for matplotlib (2-D only)."""
        if self.dim != 2:
            raise ValueError("extent is only defined for 2-D boxes")
        return [self.lo[0].item(), self.hi[0].item(), self.lo[1].item(), self.hi[1].item()]

    def to(self, device=None, dtype=None) -> "Box":
        b = Box.__new__(Box)
        b.lo = self.lo.to(device=device, dtype=dtype)
        b.hi = self.hi.to(device=device, dtype=dtype)
        b.periodic = self.periodic
        return b

    # -------------------------------------------------------------------- grid
    def _shape(self, resolution) -> tuple[int, ...]:
        try:
            shape = (operator.index(resolution),) * self.dim
        except TypeError:
            try:
                shape = tuple(operator.index(r) for r in resolution)
            except TypeError as exc:
                raise ValueError("resolution must contain positive integers") from exc
        if len(shape) != self.dim:
            raise ValueError(f"resolution has {len(shape)} entries, expected {self.dim}")
        if any(r < 1 for r in shape):
            raise ValueError("resolution must contain positive integers")
        return shape

    def grid(self, resolution) -> Tensor:
        """Node coordinates, shape ``(*resolution, n)``."""
        return grid_points(self.lo, self.hi, self._shape(resolution), self.periodic)

    def spacing(self, resolution) -> Tensor:
        return grid_spacing(self.lo, self.hi, self._shape(resolution), self.periodic)

    def axes(self, resolution) -> list[Tensor]:
        """One 1-D coordinate tensor per axis."""
        shape = self._shape(resolution)
        pts = self.grid(shape)
        out = []
        for k in range(self.dim):
            index = [0] * self.dim
            index[k] = slice(None)
            out.append(pts[tuple(index) + (k,)])
        return out

    # ------------------------------------------------------------------ points
    def contains(self, x: Tensor, tol: float = 0.0) -> Tensor:
        return ((x >= self.lo - tol) & (x <= self.hi + tol)).all(-1)

    def clamp(self, x: Tensor) -> Tensor:
        if self.periodic:
            return self.lo + torch.remainder(x - self.lo, self.size)
        return torch.maximum(torch.minimum(x, self.hi), self.lo)

    def sample_uniform(self, count: int, generator: torch.Generator | None = None) -> Tensor:
        u = torch.rand(count, self.dim, dtype=self.lo.dtype, device=self.lo.device, generator=generator)
        return self.lo + u * self.size

    def __repr__(self) -> str:
        return f"Box(lo={self.lo.tolist()}, hi={self.hi.tolist()}{', periodic=True' if self.periodic else ''})"


def Plane(lo=(0.0, 0.0), hi=(1.0, 1.0), periodic: bool = False) -> Box:
    """Convenience constructor for a 2-D box."""
    box = Box(lo, hi, periodic=periodic)
    if box.dim != 2:
        raise ValueError("Plane must be two-dimensional")
    return box
