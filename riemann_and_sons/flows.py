r"""Flows: fields evolving on a fixed geometry, and geometries evolving in time.

Field flows
-----------
* :func:`diffuse` -- Riemannian heat flow ``∂_t f = Δ_g f`` (explicit Euler
  with an automatic CFL step).

Metric flows
------------
:class:`MetricFlow` is the generic abstraction: ``flow.step(metric, dt)``
updates a metric in place, ``flow.integrate(metric, t_end, dt)`` records a
:class:`MetricTrajectory` that can be inspected and animated.

* :class:`RicciFlow`      -- ``∂_t g = −2 Ric(g)`` (optionally volume-normalised)
  for :class:`GridMetric`.  Ricci is evaluated at the nodes by nested autograd
  through the B-spline interpolant and the node metrics are updated
  (collocation).  Positive-definiteness is enforced by eigenvalue clamping.
  In two dimensions ``Ric = (R/2) g``, so the flow is a *conformal rescaling*
  ``∂_t g = −R g``: it never creates anisotropy, it only redistributes
  curvature.  Coordinates are held fixed (no DeTurck gauge); this is fine for
  the compact/periodic 2-D experiments the flow is intended for.
* :class:`GradientFlow`   -- task-driven flow in *parameter space*,
  ``dθ/dt = −∇_θ E(g_θ)`` -- the honest, explicit version of
  ``∂_t g = −grad_g E``.
* :class:`HybridFlow`     -- Lie splitting of several flows, e.g.
  ``−2λ Ric(g) + F_task(g)``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field as dc_field
from typing import Callable, Iterable

import torch
from torch import Tensor

from .fields import Field, _quadrature_weights
from .metrics import GridMetric, Metric
from .operators import LaplaceBeltrami

__all__ = [
    "stable_diffusion_dt",
    "diffuse",
    "MetricFlow",
    "MetricTrajectory",
    "RicciFlow",
    "GradientFlow",
    "HybridFlow",
]


# ---------------------------------------------------------------------------
# Diffusion of fields
# ---------------------------------------------------------------------------
def stable_diffusion_dt(field: Field, geometry, safety: float = 0.4) -> float:
    """Conservative explicit-Euler step for the assembled diffusion operator."""
    return LaplaceBeltrami(geometry, field.resolution, field.domain, detach=True).stable_dt(safety)


def _validate_integration(t: float, dt: float | None, record_every: int | None) -> None:
    if not math.isfinite(t):
        raise ValueError("integration time must be finite")
    if dt is not None and (not math.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be positive and finite")
    if record_every is not None and (not isinstance(record_every, int) or record_every <= 0):
        raise ValueError("record_every must be a positive integer")


def diffuse(
    field: Field,
    geometry,
    t: float,
    dt: float | None = None,
    record_every: int | None = None,
    callback: Callable[[float, Field], None] | None = None,
    detach_metric: bool = False,
):
    """Integrate ``∂_t f = Δ_g f`` up to time ``t`` (explicit Euler).

    Returns the final field, or ``(times, fields)`` when ``record_every`` is set.
    Zero-flux boundaries (or periodic wrap) are inherited from the domain, so
    the Riemannian mass ``∫ f √|g| dx`` is conserved.

    The operator coefficients are built once from the metric and reused across
    all time steps.  They keep their autograd history, so a loss on the result
    differentiates with respect to the metric parameters (``θ → g_θ → Δ_g → f_T``);
    pass ``detach_metric=True`` for cheaper frozen-geometry inference.
    The default step is bounded using the assembled operator. An explicit
    ``dt`` overrides that bound, so the caller is responsible for stability.
    """
    _validate_integration(t, dt, record_every)
    if t <= 0:
        return ([0.0], [field]) if record_every is not None else field
    op = LaplaceBeltrami(geometry, field.resolution, field.domain, detach=detach_metric)
    if dt is None:
        dt = op.stable_dt()
    steps = max(1, math.ceil(t / dt))
    dt = t / steps
    f = field
    times, snaps = [0.0], [f]
    for k in range(1, steps + 1):
        f = f + dt * op(f)
        if callback is not None:
            callback(k * dt, f)
        if record_every is not None and (k % record_every == 0 or k == steps):
            times.append(k * dt)
            snaps.append(f)
    if record_every is not None:
        return times, snaps
    return f


# ---------------------------------------------------------------------------
# Metric flows
# ---------------------------------------------------------------------------
@dataclass
class MetricTrajectory:
    """Snapshots ``g(t_k)`` of an evolving metric plus scalar diagnostics."""

    metric: Metric
    times: list[float] = dc_field(default_factory=list)
    states: list[dict] = dc_field(default_factory=list)
    diagnostics: dict[str, list[float]] = dc_field(default_factory=dict)

    def record(self, t: float, diag: dict[str, float] | None = None) -> None:
        self.times.append(t)
        self.states.append({k: v.detach().clone() for k, v in self.metric.state_dict().items()})
        for k, v in (diag or {}).items():
            self.diagnostics.setdefault(k, []).append(float(v))

    def at(self, index: int) -> Metric:
        """A copy of the metric loaded with snapshot ``index``."""
        m = copy.deepcopy(self.metric)
        m.load_state_dict(self.states[index])
        return m

    def __len__(self) -> int:
        return len(self.times)

    def __iter__(self) -> Iterable[tuple[float, Metric]]:
        for i in range(len(self)):
            yield self.times[i], self.at(i)


class MetricFlow:
    """Abstract metric evolution law.  Subclasses implement :meth:`step`.

    Metric flows are **forward simulation tools**: each step updates the metric's
    parameters in place under ``no_grad`` and trajectory snapshots are detached.
    Learning the parameters of a metric (``GradientFlow``, :func:`learning.fit_metric`)
    is supported; differentiating *through* an entire metric evolution is not.
    """

    def step(self, metric: Metric, dt: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def diagnostics(self, metric: Metric) -> dict[str, float]:
        return {}

    def integrate(
        self,
        metric: Metric,
        t_end: float,
        dt: float,
        record_every: int = 1,
        callback: Callable[[float, Metric], None] | None = None,
    ) -> MetricTrajectory:
        """Step from ``t = 0`` to exactly ``t_end`` (the last step is shortened if needed); ``t_end ≤ 0`` records the initial state only."""
        _validate_integration(t_end, dt, record_every)
        traj = MetricTrajectory(metric)
        traj.record(0.0, self.diagnostics(metric))
        if t_end <= 0:
            return traj
        steps = max(1, math.ceil(t_end / dt))
        t = 0.0
        for k in range(1, steps + 1):
            next_t = min(k * dt, t_end)
            step_dt = next_t - t
            self.step(metric, step_dt)
            t = next_t
            if callback is not None:
                callback(t, metric)
            if k % record_every == 0 or t == t_end:
                traj.record(t, self.diagnostics(metric))
            if t == t_end:
                break
        return traj


class RicciFlow(MetricFlow):
    r"""``∂_t g = −2 Ric(g)``  (normalised: ``+ (2 r / n) g`` with ``r`` the mean scalar curvature)."""

    def __init__(self, normalized: bool = False, min_eigenvalue: float = 1e-3, chunk: int = 1024, adaptive: bool = False, max_substeps: int = 64):
        self.normalized = normalized
        self.min_eigenvalue = min_eigenvalue
        self.chunk = chunk
        self.adaptive = adaptive  # use up to max_substeps based on the heuristic stable_dt
        self.max_substeps = max_substeps

    def stable_dt(self, metric: GridMetric, safety: float = 0.4) -> float:
        """Heuristic explicit step: in 2-D the flow is ``∂_t u = e^{-2u} Δu`` for ``g = e^{2u} I``."""
        with torch.no_grad():
            lam_min = torch.linalg.eigvalsh(metric.node_metric())[..., 0].min()
        h = metric.domain.spacing(metric.resolution).min()
        return float(safety * h**2 * lam_min / (2 * metric.dim))

    def _node_quantities(self, metric: GridMetric):
        nodes = metric.nodes
        with torch.no_grad():
            g = metric.node_metric()
            ric = metric.ricci(nodes, chunk=self.chunk)
            R = torch.einsum("...ij,...ij->...", torch.linalg.inv(g), ric)
            vol = torch.sqrt(torch.linalg.det(g))
        return g, ric, R, vol

    def step(self, metric: Metric, dt: float) -> None:
        if not isinstance(metric, GridMetric):
            raise TypeError("RicciFlow requires a GridMetric (node metrics must be updatable)")
        if self.adaptive:
            n_sub = min(self.max_substeps, max(1, math.ceil(dt / self.stable_dt(metric))))
            for _ in range(n_sub):
                self._single_step(metric, dt / n_sub)
            return
        self._single_step(metric, dt)

    def _single_step(self, metric: GridMetric, dt: float) -> None:
        g, ric, R, vol = self._node_quantities(metric)
        g_new = g - 2 * dt * ric
        if self.normalized:
            weights = vol * _quadrature_weights(metric.domain, metric.resolution)
            r = (R * weights).sum() / weights.sum()
            g_new = g_new + dt * (2 * r / metric.dim) * g
        evals, evecs = torch.linalg.eigh(0.5 * (g_new + g_new.transpose(-1, -2)))
        evals = evals.clamp_min(self.min_eigenvalue)
        g_new = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        metric.set_node_metric(g_new)

    def diagnostics(self, metric: Metric) -> dict[str, float]:
        g, ric, R, vol = self._node_quantities(metric)
        weights = vol * _quadrature_weights(metric.domain, metric.resolution)
        return {
            "volume": float(weights.sum()),
            "total_curvature": float((R * weights).sum()),
            "max_abs_R": float(R.abs().max()),
            "mean_R": float((R * weights).sum() / weights.sum()),
        }


class GradientFlow(MetricFlow):
    r"""Parameter-space gradient flow ``dθ/dt = −∇_θ E(g_θ)`` for any energy ``E(metric)``."""

    def __init__(self, energy: Callable[[Metric], Tensor], parameters: Iterable[Tensor] | None = None, clip: float | None = None):
        self.energy = energy
        self.parameters = list(parameters) if parameters is not None else None
        self.clip = clip
        self.last_energy: float | None = None

    def _params(self, metric: Metric) -> list[Tensor]:
        return self.parameters if self.parameters is not None else [p for p in metric.parameters() if p.requires_grad]

    def step(self, metric: Metric, dt: float) -> None:
        params = self._params(metric)
        with torch.enable_grad():
            e = self.energy(metric)
            grads = torch.autograd.grad(e, params, allow_unused=True)
        self.last_energy = float(e.detach())
        with torch.no_grad():
            for p, gr in zip(params, grads):
                if gr is None:
                    continue
                if self.clip is not None:
                    gr = gr.clamp(-self.clip, self.clip)
                p.sub_(dt * gr)

    def diagnostics(self, metric: Metric) -> dict[str, float]:
        with torch.no_grad():
            return {"energy": float(self.energy(metric))}


class HybridFlow(MetricFlow):
    """Lie splitting of several flows: each ``(weight, flow)`` is stepped with ``weight * dt`` in sequence."""

    def __init__(self, components: list[tuple[float, MetricFlow]]):
        self.components = components

    def step(self, metric: Metric, dt: float) -> None:
        for w, flow in self.components:
            if w != 0:
                flow.step(metric, w * dt)

    def diagnostics(self, metric: Metric) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, (_, flow) in enumerate(self.components):
            for k, v in flow.diagnostics(metric).items():
                out[f"{type(flow).__name__.lower()}_{k}"] = v
        return out
