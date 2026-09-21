r"""Metric learning: differentiable distances, losses and regularisers.

Because every quantity in the library is obtained from ``g_θ`` by autograd,
learning a geometry is just gradient descent on a loss that depends on
geodesic distances.  Two distance approximations are provided:

* :func:`straight_line_distance` -- length of the *straight* segment under
  ``g`` (an upper bound of the geodesic distance, cheap, fully batched).
* :func:`geodesic_distance` -- batched variational geodesics (a few L-BFGS
  iterations) followed by a length evaluation on the *frozen* path.  Its
  gradient w.r.t. ``θ`` is the **frozen-path approximation** of the true
  distance gradient: it is exact when the inner problem is solved to
  stationarity and the polyline is constant-speed (then ``L² = E`` and the
  envelope theorem applies to ``E``); with few inner iterations it is only
  approximate.  Measured on a 6×6 grid metric: 1.7e-4 relative error against
  finite differences of a re-solved geodesic at convergence, but ~65 % error
  after 10 L-BFGS iterations.  Use ``iterations`` generously or treat the
  few-iteration mode as a stochastic descent direction, not a gradient.

Regularisers keep a learned metric from degenerating into a look-up table:
smoothness ``‖∇g‖²``, anisotropy ``λ_max/λ_min``, Euclidean prior ``‖g − I‖²``,
volume prior ``(log det g)²`` (a triplet margin can be met trivially by
inflating all distances!), and a curvature penalty ``R²``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .curvature import metric_derivative, scalar_curvature
from .metrics import Metric
from .paths import path_energy, path_length

__all__ = [
    "straight_line_distance",
    "geodesic_distance",
    "metric_smoothness",
    "anisotropy",
    "euclidean_prior",
    "volume_prior",
    "curvature_penalty",
    "triplet_loss",
    "fit_metric",
]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def straight_line_distance(metric: Metric, x: Tensor, y: Tensor, samples: int = 8) -> Tensor:
    """``∫₀¹ ‖y − x‖_{g(x + s(y−x))} ds`` by the midpoint rule; shapes ``(..., n) -> (...)``."""
    s = (torch.arange(samples, dtype=x.dtype, device=x.device) + 0.5) / samples
    d = y - x
    pts = x.unsqueeze(-2) + s.reshape(*([1] * (x.ndim - 1)), samples, 1) * d.unsqueeze(-2)  # (..., S, n)
    speed = torch.sqrt(metric.quadratic_form(pts, d.unsqueeze(-2).expand_as(pts)).clamp_min(1e-30))
    return speed.mean(-1)


def geodesic_distance(
    geometry,
    x: Tensor,
    y: Tensor,
    n_points: int = 16,
    iterations: int = 20,
    init: Tensor | None = None,
) -> Tensor:
    """Batched variational geodesic distance ``(B, n), (B, n) -> (B,)``.

    Differentiable w.r.t. the metric through the *frozen-path* gradient: the
    optimised polyline is detached and only the length evaluation carries
    autograd history (see the module docstring for when this is exact).
    """
    B, n = x.shape
    s = torch.linspace(0, 1, n_points, dtype=x.dtype, device=x.device)[None, :, None]
    pts = x[:, None] * (1 - s) + y[:, None] * s if init is None else init
    interior = pts[:, 1:-1].detach().clone().requires_grad_(True)

    def assemble(p):
        return torch.cat([x.detach()[:, None], p, y.detach()[:, None]], dim=1)

    if iterations > 0:
        opt = torch.optim.LBFGS([interior], lr=1.0, max_iter=iterations, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            with torch.no_grad():
                pass
            e = path_energy(geometry, assemble(geometry.domain.clamp(interior))).sum()
            e.backward(inputs=[interior])
            return e

        opt.step(closure)
    final = assemble(geometry.domain.clamp(interior.detach()))
    return path_length(geometry, final)


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------
def metric_smoothness(metric: Metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Mean ``‖∂g/∂x‖_F²`` over the points ``x``."""
    dg = metric_derivative(metric, x, chunk=chunk)
    return (dg**2).sum(dim=(-1, -2, -3)).mean()


def anisotropy(metric: Metric, x: Tensor) -> Tensor:
    """Mean ``λ_max/λ_min − 1``."""
    ev = torch.linalg.eigvalsh(metric(x))
    return (ev[..., -1] / ev[..., 0] - 1).mean()


def euclidean_prior(metric: Metric, x: Tensor) -> Tensor:
    """Mean ``‖g(x) − I‖_F²``."""
    g = metric(x)
    eye = torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
    return ((g - eye) ** 2).sum(dim=(-1, -2)).mean()


def volume_prior(metric: Metric, x: Tensor) -> Tensor:
    """Mean ``(log det g)²`` -- discourages uniform inflation or collapse."""
    return torch.logdet(metric(x)).pow(2).mean()


def curvature_penalty(metric: Metric, x: Tensor, chunk: int | None = None) -> Tensor:
    """Mean squared scalar curvature."""
    return scalar_curvature(metric, x, chunk=chunk).pow(2).mean()


REGULARIZERS: dict[str, Callable] = {
    "smoothness": metric_smoothness,
    "anisotropy": anisotropy,
    "euclidean": euclidean_prior,
    "volume": volume_prior,
    "curvature": curvature_penalty,
}


# ---------------------------------------------------------------------------
# Losses & training loop
# ---------------------------------------------------------------------------
def triplet_loss(d_pos: Tensor, d_neg: Tensor, margin: float = 1.0) -> Tensor:
    """``mean max(0, m + d(x, x⁺) − d(x, x⁻))``."""
    return torch.relu(margin + d_pos - d_neg).mean()


def fit_metric(
    geometry,
    sample_triplets: Callable[[], tuple[Tensor, Tensor, Tensor]],
    epochs: int = 200,
    lr: float = 1e-2,
    margin: float = 1.0,
    distance: str = "straight",
    distance_kw: dict | None = None,
    regularizers: dict[str, float] | None = None,
    reg_points: Callable[[], Tensor] | int = 256,
    callback: Callable[[int, dict], None] | None = None,
    optimizer: str = "adam",
) -> list[dict]:
    """Train ``geometry.metric`` with a triplet loss on (approximate) geodesic distances.

    ``sample_triplets()`` returns ``(anchor, positive, negative)`` batches of
    shape ``(B, n)``.  ``regularizers`` maps names from :data:`REGULARIZERS`
    (or callables) to weights.  Returns the per-epoch history of loss terms.
    """
    metric = geometry.metric
    params = [p for p in metric.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr) if optimizer == "adam" else torch.optim.SGD(params, lr=lr)
    distance_kw = distance_kw or {}
    regularizers = regularizers or {}
    history: list[dict] = []

    def dist(a, b):
        if distance == "straight":
            return straight_line_distance(metric, a, b, **distance_kw)
        if distance == "geodesic":
            return geodesic_distance(geometry, a, b, **distance_kw)
        raise ValueError("distance must be 'straight' or 'geodesic'")

    for epoch in range(epochs):
        a, p, nq = sample_triplets()
        opt.zero_grad()
        terms = {"triplet": triplet_loss(dist(a, p), dist(a, nq), margin)}
        if regularizers:
            pts = reg_points() if callable(reg_points) else geometry.domain.sample_uniform(int(reg_points))
            for name, weight in regularizers.items():
                fn = REGULARIZERS[name] if isinstance(name, str) else name
                terms[name if isinstance(name, str) else fn.__name__] = weight * fn(metric, pts)
        loss = sum(terms.values())
        loss.backward()
        opt.step()
        rec = {k: float(v.detach()) for k, v in terms.items()}
        rec["total"] = float(loss.detach())
        history.append(rec)
        if callback is not None:
            callback(epoch, rec)
    return history
