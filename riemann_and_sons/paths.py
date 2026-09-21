"""Geodesics, exponential map, parallel transport and geodesic distance fields.

Three complementary tools:

* :func:`geodesic_shoot` / :func:`exp_map` -- integrate the geodesic ODE
  ``ẍ^k + Γ^k_ij ẋ^i ẋ^j = 0`` with RK4 (initial value problem).
* :func:`geodesic` -- boundary value problem, solved variationally by
  minimising the discrete energy ``Σ Δx_k^T g(m_k) Δx_k / Δt`` over the
  interior points of a polyline (initialised from a straight line or from a
  Dijkstra path).  Robust, differentiable, works for any metric.
* :func:`distance_field` -- geodesic distance from source points on a grid via
  Dijkstra on a 16-connected (2-D) / 3^n-connected graph with Riemannian edge
  lengths.  Not differentiable, but fast and global; its predecessor tree
  yields discrete shortest paths that make excellent initialisations.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field

import numpy as np
import torch
from torch import Tensor

from .curvature import christoffel
from .fields import Field

__all__ = [
    "Path",
    "path_length",
    "path_energy",
    "geodesic_shoot",
    "exp_map",
    "geodesic",
    "parallel_transport",
    "DistanceField",
    "distance_field",
    "resample_path",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def path_length(geometry, points: Tensor) -> Tensor:
    """Riemannian length of a polyline ``(..., K, n)`` (midpoint rule)."""
    d = points[..., 1:, :] - points[..., :-1, :]
    mid = 0.5 * (points[..., 1:, :] + points[..., :-1, :])
    return geometry.norm(mid, d).sum(-1)


def path_energy(geometry, points: Tensor) -> Tensor:
    """Discrete energy ``Σ_k ‖Δx_k‖²_g / Δt`` with ``Δt = 1/(K-1)``; minimisers are constant-speed geodesics."""
    K = points.shape[-2]
    d = points[..., 1:, :] - points[..., :-1, :]
    mid = 0.5 * (points[..., 1:, :] + points[..., :-1, :])
    return geometry.inner(mid, d, d).sum(-1) * (K - 1)


def resample_path(points: Tensor, n_points: int) -> Tensor:
    """Resample a polyline ``(K, n)`` to ``n_points`` equally spaced (in Euclidean arclength) points."""
    seg = (points[1:] - points[:-1]).norm(dim=-1)
    s = torch.cat([seg.new_zeros(1), torch.cumsum(seg, 0)])
    total = s[-1]
    if total <= 0:
        return points[:1].expand(n_points, -1).clone()
    targets = torch.linspace(0, total.item(), n_points, dtype=points.dtype, device=points.device)
    idx = torch.searchsorted(s, targets).clamp(1, len(s) - 1)
    s0, s1 = s[idx - 1], s[idx]
    w = ((targets - s0) / (s1 - s0).clamp_min(1e-12)).unsqueeze(-1)
    return points[idx - 1] * (1 - w) + points[idx] * w


@dataclass
class Path:
    """A polyline in the domain, usually an (approximate) geodesic."""

    points: Tensor  # (K, n)
    info: dict = dc_field(default_factory=dict)

    def length(self, geometry) -> Tensor:
        return path_length(geometry, self.points)

    def energy(self, geometry) -> Tensor:
        return path_energy(geometry, self.points)

    def euclidean_length(self) -> Tensor:
        return (self.points[1:] - self.points[:-1]).norm(dim=-1).sum()

    def resample(self, n_points: int) -> "Path":
        return Path(resample_path(self.points, n_points), dict(self.info))

    @property
    def start(self) -> Tensor:
        return self.points[0]

    @property
    def end(self) -> Tensor:
        return self.points[-1]

    def tangents(self) -> Tensor:
        return self.points[1:] - self.points[:-1]

    def __len__(self) -> int:
        return self.points.shape[0]


# ---------------------------------------------------------------------------
# Initial value problem
# ---------------------------------------------------------------------------
def _geodesic_rhs(geometry, x: Tensor, v: Tensor) -> Tensor:
    G = christoffel(geometry.metric, x)  # (..., n, n, n)
    return -torch.einsum("...kij,...i,...j->...k", G, v, v)


def geodesic_shoot(geometry, x0: Tensor, v0: Tensor, t: float = 1.0, steps: int = 100) -> tuple[Tensor, Tensor]:
    """Integrate the geodesic ODE from ``(x0, v0)`` with RK4.

    Returns ``(positions, velocities)`` of shape ``(..., steps + 1, n)``.
    """
    dt = t / steps
    xs = [x0]
    vs = [v0]
    x, v = x0, v0
    for _ in range(steps):
        k1x, k1v = v, _geodesic_rhs(geometry, x, v)
        k2x, k2v = v + 0.5 * dt * k1v, _geodesic_rhs(geometry, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = v + 0.5 * dt * k2v, _geodesic_rhs(geometry, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = v + dt * k3v, _geodesic_rhs(geometry, x + dt * k3x, v + dt * k3v)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(x)
        vs.append(v)
    return torch.stack(xs, dim=-2), torch.stack(vs, dim=-2)


def exp_map(geometry, x0: Tensor, v0: Tensor, steps: int = 100) -> Tensor:
    """``exp_{x0}(v0)`` -- endpoint of the unit-time geodesic with initial velocity ``v0``."""
    xs, _ = geodesic_shoot(geometry, x0, v0, t=1.0, steps=steps)
    return xs[..., -1, :]


# ---------------------------------------------------------------------------
# Boundary value problem (variational)
# ---------------------------------------------------------------------------
def geodesic(
    geometry,
    x0: Tensor,
    x1: Tensor,
    n_points: int = 64,
    init: str | Tensor = "line",
    iterations: int = 200,
    method: str = "lbfgs",
    lr: float = 2e-3,
    graph_resolution: int = 96,
    clamp_to_domain: bool = True,
    tol: float = 1e-10,
) -> Path:
    """Shortest path between ``x0`` and ``x1`` by minimising the discrete path energy.

    ``init`` may be ``"line"`` (straight segment), ``"graph"`` (Dijkstra path on a
    grid -- recommended when the metric has strong obstacles, since the
    variational problem is non-convex) or a tensor ``(K, n)`` of initial points.

    The returned points are detached.  ``Path.length(geometry)`` re-evaluates
    the metric along the path, so gradients with respect to metric parameters
    are available (exact to first order by the envelope theorem).
    """
    x0 = torch.as_tensor(x0)
    x1 = torch.as_tensor(x1)
    if isinstance(init, str):
        if init == "line":
            s = torch.linspace(0, 1, n_points, dtype=x0.dtype, device=x0.device)[:, None]
            pts = x0 * (1 - s) + x1 * s
        elif init == "graph":
            df = distance_field(geometry, x1[None], graph_resolution)
            pts = resample_path(df.path_to(x0).points, n_points)
            pts[0], pts[-1] = x0, x1
        else:
            raise ValueError("init must be 'line', 'graph' or a tensor")
    else:
        pts = resample_path(torch.as_tensor(init), n_points)
    interior = pts[1:-1].clone().detach().requires_grad_(True)
    info: dict = {"init": init if isinstance(init, str) else "tensor"}

    def assemble(p: Tensor) -> Tensor:
        return torch.cat([x0[None], p, x1[None]], dim=0)

    def energy_fn(p: Tensor) -> Tensor:
        q = geometry.domain.clamp(p) if clamp_to_domain else p
        return path_energy(geometry, assemble(q))

    history: list[float] = []
    with torch.enable_grad():
        if method == "lbfgs":
            opt = torch.optim.LBFGS([interior], lr=1.0, max_iter=iterations, tolerance_grad=tol, tolerance_change=tol, history_size=50, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                e = energy_fn(interior)
                e.backward()
                history.append(e.item())
                return e

            opt.step(closure)
        elif method == "adam":
            opt = torch.optim.Adam([interior], lr=lr)
            for _ in range(iterations):
                opt.zero_grad()
                e = energy_fn(interior)
                e.backward()
                opt.step()
                history.append(e.item())
        else:
            raise ValueError("method must be 'lbfgs' or 'adam'")
    final = assemble(geometry.domain.clamp(interior.detach()) if clamp_to_domain else interior.detach())
    info["energy_history"] = history
    info["iterations"] = len(history)
    return Path(final, info)


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------
def parallel_transport(geometry, points: Tensor, v0: Tensor, substeps: int = 4) -> Tensor:
    """Parallel-transport ``v0`` along the polyline ``points`` ``(K, n)``.

    Integrates ``v̇^k = −Γ^k_ij ẋ^i v^j`` with RK4 on each linear segment.
    Returns the transported vectors at every vertex, shape ``(K, n)``.
    """
    out = [v0]
    v = v0
    for k in range(points.shape[0] - 1):
        a, b = points[k], points[k + 1]
        dx = b - a
        ds = 1.0 / substeps

        def rhs(s, vv):
            x = a + s * dx
            G = christoffel(geometry.metric, x[None])[0]
            return -torch.einsum("kij,i,j->k", G, dx, vv)

        s = 0.0
        for _ in range(substeps):
            k1 = rhs(s, v)
            k2 = rhs(s + 0.5 * ds, v + 0.5 * ds * k1)
            k3 = rhs(s + 0.5 * ds, v + 0.5 * ds * k2)
            k4 = rhs(s + ds, v + ds * k3)
            v = v + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s += ds
        out.append(v)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Geodesic distance fields (Dijkstra)
# ---------------------------------------------------------------------------
def _offsets(n: int, knight: bool) -> list[tuple[int, ...]]:
    offs = [o for o in itertools.product((-1, 0, 1), repeat=n) if any(o)]
    if knight and n == 2:
        offs += [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    return offs


@dataclass
class DistanceField:
    """Geodesic distance to a set of sources, sampled on a grid, plus the Dijkstra predecessor tree."""

    field: Field
    sources: Tensor  # (S, n) original source points
    source_nodes: Tensor  # (S,) flat node indices
    predecessors: Tensor  # (N,) flat predecessor per node (-1 for sources / unreachable)
    geometry: object

    @property
    def values(self) -> Tensor:
        return self.field.values

    def nearest_node(self, x: Tensor) -> int:
        dom = self.field.domain
        res = self.field.resolution
        h = dom.spacing(res)
        idx = torch.round((torch.as_tensor(x) - dom.lo) / h).long()
        sizes = torch.as_tensor(res, device=idx.device)
        idx = torch.remainder(idx, sizes) if dom.periodic else torch.minimum(idx.clamp_min(0), sizes - 1)
        flat = 0
        for k in range(len(res)):
            flat = flat * res[k] + idx[k].item()
        return int(flat)

    def sample(self, x: Tensor) -> Tensor:
        return self.field.sample(x, mode="linear")

    def path_to(self, x: Tensor) -> Path:
        """Discrete shortest path from ``x`` to the nearest source, following the predecessor tree."""
        pts_flat = self.field.points.reshape(-1, self.field.dim)
        node = self.nearest_node(x)
        chain = [node]
        pred = self.predecessors
        while pred[chain[-1]] >= 0:
            chain.append(int(pred[chain[-1]]))
            if len(chain) > pred.numel():
                raise RuntimeError("predecessor chain does not terminate")
        pts = pts_flat[torch.as_tensor(chain)]
        x = torch.as_tensor(x, dtype=pts.dtype, device=pts.device)
        # snap the two ends to the exact query / source coordinates
        src_idx = torch.argmin((self.sources - pts[-1]).norm(dim=-1))
        pts = torch.cat([x[None], pts[1:-1], self.sources[src_idx][None]], dim=0) if len(pts) > 1 else torch.stack([x, self.sources[src_idx]])
        return Path(pts, {"discrete": True})


def distance_field(geometry, sources: Tensor, resolution=128, knight_moves: bool = True) -> DistanceField:
    """Geodesic distance from ``sources`` ``(S, n)`` on a grid, via Dijkstra with Riemannian edge weights.

    Edge weight between neighbouring nodes ``p, q`` with displacement ``d``:
    ``½(‖d‖_{g(p)} + ‖d‖_{g(q)})`` (trapezoidal rule).  In 2-D the graph is
    16-connected (8 neighbours + knight moves) which keeps the metrication
    error at roughly 1-2 %.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    dom = geometry.domain
    n = dom.dim
    res = dom._shape(resolution)
    pts = dom.grid(res)
    h = dom.spacing(res)
    with torch.no_grad():
        g = geometry.g(pts).reshape(-1, n, n)
    N = g.shape[0]
    sizes = np.array(res)
    multi = np.stack(np.meshgrid(*[np.arange(m) for m in res], indexing="ij"), -1).reshape(-1, n)
    strides = np.array([math.prod(res[k + 1 :]) for k in range(n)])
    rows, cols, weights = [], [], []
    for o in _offsets(n, knight_moves):
        o_arr = np.array(o)
        nb = multi + o_arr
        if dom.periodic:
            nb = np.mod(nb, sizes)
            valid = np.ones(N, dtype=bool)
        else:
            valid = np.all((nb >= 0) & (nb < sizes), axis=1)
        src = np.nonzero(valid)[0]
        dst = (nb[valid] * strides).sum(1)
        d = torch.as_tensor(o_arr, dtype=h.dtype, device=h.device) * h
        gs = g[torch.as_tensor(src)]
        gd = g[torch.as_tensor(dst)]
        w = 0.5 * (torch.sqrt(torch.einsum("i,bij,j->b", d, gs, d)) + torch.sqrt(torch.einsum("i,bij,j->b", d, gd, d)))
        rows.append(src)
        cols.append(dst)
        weights.append(w.cpu().numpy())
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()

    sources = torch.as_tensor(sources, dtype=pts.dtype, device=pts.device)
    if sources.ndim == 1:
        sources = sources[None]
    src_nodes = []
    for s in sources:
        idx = torch.round((s - dom.lo) / h).long()
        idx = torch.minimum(idx.clamp_min(0), torch.as_tensor(res) - 1)
        src_nodes.append(int((idx.cpu().numpy() * strides).sum()))
    dist, pred, _ = dijkstra(graph, directed=True, indices=src_nodes, min_only=True, return_predecessors=True)
    dist_t = torch.as_tensor(dist, dtype=pts.dtype, device=pts.device).reshape(res)
    pred_t = torch.as_tensor(pred.astype(np.int64), device=pts.device)
    pred_t[pred_t < 0] = -1
    return DistanceField(Field(dist_t, dom), sources, torch.as_tensor(src_nodes), pred_t, geometry)
