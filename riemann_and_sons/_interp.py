"""Grid interpolation primitives (n-dimensional, differentiable, ``torch.func``-safe).

Everything that lives on a regular grid in Riemann & Sons -- learnable metric
parameters, displacement fields, images -- is evaluated off-grid through this
module.  Two schemes are provided:

* ``"linear"``  -- multilinear interpolation (C^0). Cheap, but derivatives
  jump at cell boundaries, so it is unsuitable when smooth curvature is needed.
* ``"cubic"``   -- uniform cubic B-spline (C^2).  Slightly smoother than the
  data it is fitted to, but twice continuously differentiable, which is what
  Christoffel symbols and curvature require.

Implementation notes
--------------------
* Indices are gathered with ``index_select`` on a flattened grid, which is
  compatible with ``torch.func.vmap`` / ``jacfwd`` (unlike ``grid_sample``,
  whose double-backward support has historically been patchy).
* Coordinate ``k`` of a point corresponds to array axis ``k`` of the grid.
* Out-of-domain indices repeat the nearest boundary coefficient unless
  ``periodic=True``, in which case indices wrap (torus). A cubic spline
  stays smooth across the boundary and becomes constant one grid cell beyond it;
  it does not clamp queries to the boundary value.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor

__all__ = ["interpolate", "bspline_prefilter", "grid_spacing", "grid_points"]


def grid_spacing(lo: Tensor, hi: Tensor, shape: tuple[int, ...], periodic: bool = False) -> Tensor:
    n_nodes = torch.as_tensor(shape, dtype=lo.dtype, device=lo.device)
    if periodic:
        return (hi - lo) / n_nodes
    return (hi - lo) / (n_nodes - 1).clamp_min(1)


def grid_points(lo: Tensor, hi: Tensor, shape: tuple[int, ...], periodic: bool = False) -> Tensor:
    """Node coordinates of a regular grid, shape ``(*shape, n)``, ``ij`` indexing."""
    axes = []
    for k, m in enumerate(shape):
        if periodic:
            axes.append(lo[k] + (hi[k] - lo[k]) * torch.arange(m, dtype=lo.dtype, device=lo.device) / m)
        else:
            axes.append(torch.linspace(lo[k].item(), hi[k].item(), m, dtype=lo.dtype, device=lo.device))
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _cubic_weights(t: Tensor) -> Tensor:
    """Uniform cubic B-spline basis for the four nodes floor(u)-1 .. floor(u)+2."""
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - t) ** 3 / 6
    w1 = (3 * t3 - 6 * t2 + 4) / 6
    w2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6
    w3 = t3 / 6
    return torch.stack([w0, w1, w2, w3], dim=-1)


def _linear_weights(t: Tensor) -> Tensor:
    return torch.stack([1 - t, t], dim=-1)


def interpolate(
    values: Tensor,
    x: Tensor,
    lo: Tensor,
    hi: Tensor,
    mode: str = "cubic",
    periodic: bool = False,
) -> Tensor:
    """Evaluate a gridded field at arbitrary points.

    Parameters
    ----------
    values : ``(*grid_shape, F)``  -- node values (or B-spline coefficients).
    x      : ``(..., n)``          -- query points in domain coordinates.
    lo, hi : ``(n,)``              -- domain bounds.
    mode   : ``"cubic"`` or ``"linear"``.
    periodic : wrap indices instead of clamping.

    Returns
    -------
    ``(..., F)``
    """
    grid_shape = tuple(values.shape[:-1])
    n = len(grid_shape)
    feat = values.shape[-1]
    if x.shape[-1] != n:
        raise ValueError(f"point dimension {x.shape[-1]} != grid dimension {n}")

    flat = values.reshape(-1, feat)
    batch_shape = x.shape[:-1]
    xf = x.reshape(-1, n)

    sizes = torch.as_tensor(grid_shape, dtype=torch.long, device=x.device)
    strides = torch.as_tensor(
        [math.prod(grid_shape[k + 1 :]) for k in range(n)], dtype=torch.long, device=x.device
    )
    h = grid_spacing(lo, hi, grid_shape, periodic)
    u = (xf - lo) / h  # continuous index coordinates
    base = torch.floor(u)
    t = u - base
    base = base.long()

    if mode == "cubic":
        weights = _cubic_weights(t)  # (B, n, 4)
        base = base - 1
        support = 4
    elif mode == "linear":
        weights = _linear_weights(t)  # (B, n, 2)
        support = 2
    else:
        raise ValueError(f"unknown interpolation mode {mode!r}")

    out = torch.zeros(xf.shape[0], feat, dtype=values.dtype, device=values.device)
    for combo in itertools.product(range(support), repeat=n):
        idx = base + torch.as_tensor(combo, dtype=torch.long, device=x.device)  # (B, n)
        if periodic:
            idx = torch.remainder(idx, sizes)
        else:
            idx = torch.minimum(torch.clamp(idx, min=0), sizes - 1)
        flat_idx = (idx * strides).sum(-1)
        w = weights[:, 0, combo[0]]
        for d in range(1, n):
            w = w * weights[:, d, combo[d]]
        out = out + w.unsqueeze(-1) * flat.index_select(0, flat_idx)
    return out.reshape(*batch_shape, feat)


def _axis_matrix(m: int, periodic: bool, dtype, device) -> Tensor:
    """1-D cubic B-spline collocation matrix at the nodes (same clamping rule as `interpolate`)."""
    M = torch.zeros(m, m, dtype=dtype, device=device)
    w = torch.tensor([1 / 6, 4 / 6, 1 / 6, 0.0], dtype=dtype, device=device)
    for i in range(m):
        for o in range(4):
            j = i - 1 + o
            j = j % m if periodic else min(max(j, 0), m - 1)
            M[i, j] += w[o]
    return M


def bspline_prefilter(node_values: Tensor, periodic: bool = False) -> Tensor:
    """Convert node *values* into B-spline *coefficients* so that the spline interpolates them.

    Solves the separable collocation system axis by axis.  ``node_values`` has
    shape ``(*grid_shape, F)``.
    """
    grid_shape = tuple(node_values.shape[:-1])
    n = len(grid_shape)
    coeffs = node_values
    for axis in range(n):
        m = grid_shape[axis]
        if m < 2:
            continue
        M = _axis_matrix(m, periodic, coeffs.dtype, coeffs.device)
        moved = coeffs.movedim(axis, 0)
        solved = torch.linalg.solve(M, moved.reshape(m, -1))
        coeffs = solved.reshape(moved.shape).movedim(0, axis)
    return coeffs
