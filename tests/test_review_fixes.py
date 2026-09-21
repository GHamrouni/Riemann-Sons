"""Regression tests for the correctness contracts flagged in the review of commit 5a8d8ca."""

import math

import torch

import riemann_and_sons as rn
from riemann_and_sons import learning
from riemann_and_sons.operators import LaplaceBeltrami


def _anisotropic_metric():
    def g(x):
        a = 1 + 0.5 * torch.sin(2 * math.pi * x[..., 0])
        b = 1 + 0.5 * torch.cos(2 * math.pi * x[..., 1])
        c = 0.3 * torch.sin(2 * math.pi * (x[..., 0] + x[..., 1]))
        return torch.stack([torch.stack([a, c], -1), torch.stack([c, b], -1)], -2)

    return rn.FunctionMetric(g, dim=2)


# ------------------------------------------------------------------ Laplace–Beltrami structure
def test_laplacian_is_weighted_self_adjoint_and_negative_semidefinite():
    for periodic in (True, False):
        dom = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=periodic)
        op = LaplaceBeltrami(rn.Geometry(dom, _anisotropic_metric()), 8, dom)
        L = op.matrix()
        M = torch.diag(op.mass.reshape(-1))
        ML = M @ L
        assert (ML - ML.T).norm() / ML.norm() < 1e-12  # M L symmetric
        evals = torch.linalg.eigvalsh(0.5 * (ML + ML.T))
        assert evals.max() < 1e-10  # −K is negative semi-definite
        ones = torch.ones(L.shape[0])
        assert (L @ ones).abs().max() < 1e-10  # constants annihilated
        assert (ones @ ML).abs().max() < 1e-10  # weighted mass conservation
        f = torch.randn(L.shape[0])
        assert f @ (ML @ f) <= 1e-12  # energy dissipation


def test_laplacian_has_no_checkerboard_null_mode():
    dom = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    op = LaplaceBeltrami(rn.Geometry(dom), 8, dom)
    i, j = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    checker = ((-1.0) ** (i + j)).to(torch.float64)
    assert op.apply(checker).abs().max() > 1.0


def test_diffusion_gradient_wrt_metric_matches_finite_differences():
    tor = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    f0 = rn.Field.from_function(tor, 16, lambda p: torch.sin(2 * math.pi * p[..., 0]) + 0.3 * torch.cos(4 * math.pi * p[..., 1]))

    def loss_fn(theta):
        metric = rn.FunctionMetric(lambda x: torch.exp(theta) * torch.diag_embed(torch.stack([1 + 0.5 * x[..., 0], torch.ones_like(x[..., 0])], -1)), dim=2)
        out = rn.diffuse(f0, rn.Geometry(tor, metric), t=0.002, dt=0.0005)
        return (out.values**2).sum()

    theta = torch.tensor(0.2, requires_grad=True)
    (grad,) = torch.autograd.grad(loss_fn(theta), theta)
    eps = 1e-5
    fd = (loss_fn(torch.tensor(0.2 + eps)) - loss_fn(torch.tensor(0.2 - eps))) / (2 * eps)
    assert torch.isfinite(grad) and abs(grad.item() - fd.item()) < 1e-6 * max(1.0, abs(fd.item()))


def test_diffusion_detach_option_blocks_metric_gradient():
    tor = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    theta = torch.tensor(0.1, requires_grad=True)
    metric = rn.ConformalMetric(lambda x: torch.exp(theta) * torch.ones_like(x[..., 0]), dim=2)
    f0 = rn.Field.from_function(tor, 8, lambda p: torch.sin(2 * math.pi * p[..., 0]))
    out = rn.diffuse(f0, rn.Geometry(tor, metric), t=0.001, dt=0.0005, detach_metric=True)
    assert not out.values.requires_grad


def test_grid_metric_learns_from_diffusion_observations():
    """One gradient step on a GridMetric through the diffusion operator decreases a diffusion-matching loss."""
    dom = rn.Plane()
    target = rn.Geometry(dom, rn.ConstantMetric(torch.tensor([[3.0, 0.0], [0.0, 1.0]])))
    f0 = rn.Field.from_function(dom, 16, lambda p: torch.exp(-((p - 0.5) ** 2).sum(-1) / 0.005))
    obs = rn.diffuse(f0, target, t=0.001).values.detach()
    gm = rn.GridMetric(dom, 4)
    geo = rn.Geometry(dom, gm)
    opt = torch.optim.Adam(gm.parameters(), lr=0.05)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = ((rn.diffuse(f0, geo, t=0.001).values - obs) ** 2).sum()
        loss.backward()
        assert gm.raw.grad is not None and gm.raw.grad.abs().sum() > 0
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ------------------------------------------------------------------ graph retrieval
def test_graph_retriever_does_not_double_reciprocal_edges():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    r = rn.GeodesicRetriever(pts, rn.Geometry(rn.Box([-1, -1], [4, 1])), method="graph", k_graph=1)
    d = r.query(pts[:1], k=3).distances[0]
    assert torch.allclose(d, torch.tensor([0.0, 1.0, 3.0]), atol=1e-12)


def test_graph_retriever_refresh_tracks_metric_changes():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    scale = torch.tensor(1.0)
    metric = rn.ConformalMetric(lambda x: scale * torch.ones_like(x[..., 0]), dim=2)
    r = rn.GeodesicRetriever(pts, rn.Geometry(rn.Box([-1, -1], [4, 1]), metric), method="graph", k_graph=1)
    scale.fill_(4.0)
    stale = r.query(pts[:1], k=3).distances[0]
    r.refresh()
    fresh = r.query(pts[:1], k=3).distances[0]
    assert torch.allclose(fresh, torch.tensor([0.0, 2.0, 6.0]), atol=1e-12)
    assert not torch.allclose(stale, fresh)


# ------------------------------------------------------------------ frozen-path gradient
def test_frozen_path_gradient_matches_finite_differences_of_resolved_geodesic():
    dom = rn.Plane()
    base = rn.GridMetric.from_function(dom, 6, lambda x: (1 + 3 * torch.exp(-((x - 0.5) ** 2).sum(-1) / 0.02))[..., None, None] * torch.eye(2))
    direction = torch.randn_like(base.raw)
    a, b = torch.tensor([0.1, 0.2]), torch.tensor([0.9, 0.7])

    def solve(eps, iters=500):
        m = rn.GridMetric(dom, 6)
        with torch.no_grad():
            m.raw.copy_(base.raw + eps * direction)
        geo = rn.Geometry(dom, m)
        path = rn.geodesic(geo, a, b, n_points=48, iterations=iters, tol=1e-14)
        return geo, path

    geo, path = solve(0.0)
    assert path.info["grad_norm"] < 1e-4  # inner problem converged
    length = path.length(geo)
    (grad,) = torch.autograd.grad(length, geo.metric.raw)
    frozen = (grad * direction).sum().item()
    eps = 1e-4
    fd = (solve(eps)[1].length(solve(eps)[0]).item() - solve(-eps)[1].length(solve(-eps)[0]).item()) / (2 * eps)
    assert abs(frozen - fd) < 1e-3 * abs(fd)  # measured: ~1.7e-4 relative at convergence
    # with a crude inner solve the frozen-path gradient is only a rough approximation (measured ~65 % off)
    geo10, path10 = solve(0.0, iters=10)
    assert path10.info["grad_norm"] > 1e-2


# ------------------------------------------------------------------ flows
def test_metric_flow_integrate_hits_t_end_exactly_and_zero_is_noop():
    class Count(rn.MetricFlow):
        def __init__(self):
            self.calls = []

        def step(self, m, dt):
            self.calls.append(dt)

    gm = rn.GridMetric(rn.Plane(), 3)
    flow = Count()
    traj = flow.integrate(gm, 0.25, 0.1)
    assert abs(traj.times[-1] - 0.25) < 1e-12 and len(flow.calls) == 3 and abs(flow.calls[-1] - 0.05) < 1e-12
    flow2 = Count()
    traj0 = flow2.integrate(gm, 0.0, 0.1)
    assert traj0.times == [0.0] and flow2.calls == []


# ------------------------------------------------------------------ structured metrics
def test_diagonal_low_rank_quadratic_form_matches_dense():
    D = lambda x: 1 + x**2
    U = lambda x: torch.stack([x, torch.sin(x)], -1)
    m = rn.DiagonalLowRankMetric(D, U)
    x, v = torch.randn(7, 5), torch.randn(7, 5)
    assert torch.allclose(m.quadratic_form(x, v), torch.einsum("bi,bij,bj->b", v, m(x), v), atol=1e-12)
    assert torch.allclose(learning.straight_line_distance(m, x, x + v), learning.straight_line_distance(rn.FunctionMetric(m.forward), x, x + v))
