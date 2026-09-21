"""Synthetic scenes, metrics and point worlds for the 2-D laboratory."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .domains import Box
from .fields import Field, Image
from .metrics import ConformalMetric, FunctionMetric, GridMetric

__all__ = [
    "shapes_scene",
    "gaussian_blob",
    "checkerboard",
    "bump_conformal_metric",
    "wall_metric",
    "anisotropic_flow_metric",
    "two_moons",
    "clustered_world",
    "triplet_sampler",
    "landmark_pairs",
]


# ---------------------------------------------------------------- images
def shapes_scene(domain: Box | None = None, resolution: int = 128, soft: float = 0.0) -> Image:
    """Circles, a rectangle and a bar: a scene with clear boundaries (Demos A/B)."""
    dom = domain or Box([0.0, 0.0], [1.0, 1.0])
    pts = dom.grid(resolution)
    x, y = pts[..., 0], pts[..., 1]

    def disk(cx, cy, r):
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        return torch.sigmoid(-d / soft) if soft > 0 else (d < 0).to(x.dtype)

    def rect(x0, x1, y0, y1):
        inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        return inside.to(x.dtype)

    img = torch.zeros_like(x)
    img = torch.maximum(img, 0.9 * disk(0.3, 0.7, 0.15))
    img = torch.maximum(img, 0.6 * disk(0.72, 0.3, 0.12))
    img = torch.maximum(img, 0.75 * rect(0.55, 0.9, 0.6, 0.85))
    img = torch.maximum(img, 0.45 * rect(0.1, 0.45, 0.12, 0.22))
    return Image(img[None], dom)


def gaussian_blob(domain: Box, resolution: int, center, sigma: float, amplitude: float = 1.0) -> Field:
    pts = domain.grid(resolution)
    c = torch.as_tensor(center, dtype=pts.dtype)
    return Field(amplitude * torch.exp(-((pts - c) ** 2).sum(-1) / (2 * sigma**2)), domain)


def checkerboard(domain: Box, resolution: int, tiles: int = 8) -> Image:
    pts = domain.grid(resolution)
    u = ((pts - domain.lo) / domain.size * tiles).floor()
    vals = ((u.sum(-1)) % 2).to(pts.dtype)
    return Image(vals[None], domain)


# ---------------------------------------------------------------- metrics
def bump_conformal_metric(centers, radii, heights, dim: int = 2) -> ConformalMetric:
    """``g = λ(x) I`` with ``λ = 1 + Σ h_k exp(−|x−c_k|²/(2 r_k²))``: smooth expensive blobs."""
    centers = torch.as_tensor(centers, dtype=torch.get_default_dtype())
    radii = torch.as_tensor(radii, dtype=centers.dtype)
    heights = torch.as_tensor(heights, dtype=centers.dtype)

    def lam(x: Tensor) -> Tensor:
        d2 = ((x[..., None, :] - centers) ** 2).sum(-1)  # (..., K)
        return 1 + (heights * torch.exp(-d2 / (2 * radii**2))).sum(-1)

    return ConformalMetric(lam, dim=dim)


def wall_metric(x_wall: float = 0.5, gap=(0.45, 0.55), thickness: float = 0.03, cost: float = 40.0, softness: float = 0.01) -> ConformalMetric:
    """A vertical wall at ``x = x_wall`` with a gap in ``y ∈ gap``; crossing the wall costs ``√cost`` per unit length."""

    def lam(x: Tensor) -> Tensor:
        in_wall = torch.sigmoid((thickness - (x[..., 0] - x_wall).abs()) / softness)
        in_gap = torch.sigmoid((x[..., 1] - gap[0]) / softness) * torch.sigmoid((gap[1] - x[..., 1]) / softness)
        return 1 + cost * in_wall * (1 - in_gap)

    return ConformalMetric(lam, dim=2)


def anisotropic_flow_metric(ratio: float = 16.0, center=(0.5, 0.5), base: float = 1.0) -> FunctionMetric:
    """``g = base (I + (ratio−1) t tᵀ)`` with ``t`` the unit *radial* direction: moving radially is expensive, circling is cheap."""
    c = torch.as_tensor(center, dtype=torch.get_default_dtype())

    def g(x: Tensor) -> Tensor:
        d = x - c
        t = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eye = torch.eye(2, dtype=x.dtype, device=x.device)
        return base * (eye + (ratio - 1) * t[..., :, None] * t[..., None, :])

    return FunctionMetric(g, dim=2)


# ---------------------------------------------------------------- point worlds
def two_moons(n: int = 300, noise: float = 0.05, seed: int = 0, domain: Box | None = None) -> tuple[Tensor, Tensor]:
    """Two interleaving arcs in the unit square; labels 0/1."""
    gen = torch.Generator().manual_seed(seed)
    m = n // 2
    t = torch.rand(m, generator=gen) * math.pi
    a = torch.stack([torch.cos(t), torch.sin(t)], -1)
    b = torch.stack([1 - torch.cos(t), 1 - torch.sin(t) - 0.5], -1)
    pts = torch.cat([a, b]) + noise * torch.randn(2 * m, 2, generator=gen)
    labels = torch.cat([torch.zeros(m, dtype=torch.long), torch.ones(m, dtype=torch.long)])
    # map into [0.1, 0.9]^2
    lo, hi = pts.min(0).values, pts.max(0).values
    pts = 0.1 + 0.8 * (pts - lo) / (hi - lo)
    if domain is not None:
        pts = domain.lo + pts * domain.size
    return pts, labels


def clustered_world(n_per: int = 60, seed: int = 0, spread: float = 0.06) -> tuple[Tensor, Tensor]:
    """Four Gaussian clusters whose *labels* pair clusters diagonally, so that Euclidean neighbours are often wrong.

    Cluster centres (0.25,0.25) & (0.75,0.75) share label 0, (0.25,0.75) & (0.75,0.25) share label 1.
    A metric that makes horizontal/vertical travel expensive relative to the
    diagonals makes same-label clusters geodesically close.
    """
    gen = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]])
    labels_c = torch.tensor([0, 0, 1, 1])
    pts, labels = [], []
    for c, l in zip(centers, labels_c):
        pts.append(c + spread * torch.randn(n_per, 2, generator=gen))
        labels.append(torch.full((n_per,), int(l), dtype=torch.long))
    return torch.cat(pts).clamp(0.02, 0.98), torch.cat(labels)


def triplet_sampler(points: Tensor, labels: Tensor, batch: int = 64, seed: int = 0, local_k: int | None = None):
    """Return a callable producing random ``(anchor, positive, negative)`` batches from labelled points.

    With ``local_k`` set, positives and negatives are drawn only among the
    ``local_k`` *Euclidean-nearest* same-class / other-class points of the anchor
    (hard triplets, the regime that matters for nearest-neighbour retrieval).
    """
    gen = torch.Generator().manual_seed(seed)
    N = points.shape[0]
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    other = ~same & ~torch.eye(N, dtype=torch.bool)
    pos_w = same.double()
    neg_w = other.double()
    if local_k is not None:
        D = torch.cdist(points, points)
        big = torch.full_like(D, float("inf"))
        pos_rank = torch.where(same, D, big).argsort(dim=1)[:, :local_k]
        neg_rank = torch.where(other, D, big).argsort(dim=1)[:, :local_k]
        pos_w = torch.zeros(N, N, dtype=torch.double).scatter_(1, pos_rank, 1.0) * same
        neg_w = torch.zeros(N, N, dtype=torch.double).scatter_(1, neg_rank, 1.0) * other

    def sample():
        a = torch.randint(0, N, (batch,), generator=gen)
        pos_idx = torch.multinomial(pos_w[a], 1, generator=gen)[:, 0]
        neg_idx = torch.multinomial(neg_w[a], 1, generator=gen)[:, 0]
        return points[a], points[pos_idx], points[neg_idx]

    return sample


def landmark_pairs(kind: str = "pinch") -> tuple[Tensor, Tensor]:
    """Control constraints for the deformation demo: sources and targets in the unit square."""
    if kind == "pinch":
        src = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.5, 0.5], [0.1, 0.5], [0.9, 0.5], [0.5, 0.1], [0.5, 0.9]])
        dst = torch.tensor([[0.25, 0.25], [0.75, 0.3], [0.65, 0.75], [0.35, 0.65], [0.58, 0.45], [0.1, 0.5], [0.9, 0.5], [0.5, 0.1], [0.5, 0.9]])
    elif kind == "twist":
        ang = torch.linspace(0, 2 * math.pi, 9)[:-1]
        src = 0.5 + 0.3 * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        rot = ang + 0.5
        dst = 0.5 + 0.3 * torch.stack([torch.cos(rot), torch.sin(rot)], -1)
        border = torch.tensor([[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]])
        src, dst = torch.cat([src, border]), torch.cat([dst, border])
    else:
        raise ValueError("kind must be 'pinch' or 'twist'")
    return src, dst
