import math

import pytest
import torch

import riemann_and_sons as rn
from riemann_and_sons import curvature


def test_euclidean_is_flat():
    m = rn.EuclideanMetric()
    x = torch.rand(5, 3)
    assert torch.allclose(m(x), torch.eye(3).expand(5, 3, 3))
    assert torch.allclose(m.christoffel(x), torch.zeros(5, 3, 3, 3))
    assert torch.allclose(m.scalar_curvature(x), torch.zeros(5))


def test_sphere_christoffel_and_curvature(sphere_geometry):
    x = torch.tensor([[0.7, 0.3], [1.2, 2.0], [2.0, -1.0]])
    G = sphere_geometry.christoffel(x)
    th = x[:, 0]
    assert torch.allclose(G[:, 0, 1, 1], -torch.sin(th) * torch.cos(th), atol=1e-10)  # Γ^θ_φφ
    assert torch.allclose(G[:, 1, 0, 1], torch.cos(th) / torch.sin(th), atol=1e-10)  # Γ^φ_θφ
    assert torch.allclose(G[:, 1, 1, 0], G[:, 1, 0, 1])  # symmetry in lower indices
    assert torch.allclose(sphere_geometry.scalar_curvature(x), torch.full((3,), 2.0), atol=1e-9)
    assert torch.allclose(sphere_geometry.gauss_curvature(x), torch.ones(3), atol=1e-9)


def test_hyperbolic_curvature_and_ricci_identity(hyperbolic_geometry):
    x = torch.tensor([[0.3, 0.5], [-1.0, 2.0], [2.0, 0.1]])
    R = hyperbolic_geometry.scalar_curvature(x)
    assert torch.allclose(R, torch.full((3,), -2.0), atol=1e-9)
    Ric = hyperbolic_geometry.ricci(x)
    g = hyperbolic_geometry.g(x)
    assert torch.allclose(Ric, 0.5 * R[:, None, None] * g, atol=1e-9)  # 2-D: Ric = (R/2) g


def test_stereographic_sphere_curvature(stereographic_sphere):
    x = torch.tensor([[0.3, 0.5], [-1.0, 2.0], [0.0, 0.0]])
    assert torch.allclose(stereographic_sphere.scalar_curvature(x), torch.full((3,), 2.0), atol=1e-9)


def test_riemann_symmetries(hyperbolic_geometry):
    x = torch.tensor([[0.3, 0.5]])
    R = hyperbolic_geometry.riemann(x)[0]  # R^ρ_σμν
    assert torch.allclose(R, -R.permute(0, 1, 3, 2), atol=1e-10)  # antisymmetric in μν
    g = hyperbolic_geometry.g(x)[0]
    Rl = torch.einsum("ar,rsmn->asmn", g, R)  # R_ρσμν
    assert torch.allclose(Rl, -Rl.permute(1, 0, 2, 3), atol=1e-10)
    assert torch.allclose(Rl, Rl.permute(2, 3, 0, 1), atol=1e-10)


@pytest.mark.parametrize("param", [rn.CholeskyParameterization(), rn.LogEuclideanParameterization(), rn.ConformalParameterization()])
def test_grid_metric_reproduces_analytic_metric_and_curvature(param, stereographic_sphere):
    dom = rn.Box([-0.8, -0.8], [0.8, 0.8])
    gm = rn.GridMetric.from_function(dom, 48, stereographic_sphere, parameterization=param)
    assert torch.allclose(gm.node_metric(), stereographic_sphere(gm.nodes), atol=1e-10)
    xq = torch.tensor([[0.1, 0.2], [-0.4, 0.3], [0.5, -0.5]])
    assert torch.allclose(gm.scalar_curvature(xq), torch.full((3,), 2.0), atol=2e-3)


def test_grid_metric_is_learnable_through_curvature():
    dom = rn.Plane()
    gm = rn.GridMetric(dom, 6)
    with torch.no_grad():
        gm.raw.add_(0.2 * torch.randn_like(gm.raw))
    R = gm.scalar_curvature(dom.sample_uniform(6))
    R.pow(2).sum().backward()
    assert gm.raw.grad is not None and torch.isfinite(gm.raw.grad).all() and gm.raw.grad.abs().sum() > 0


@pytest.mark.parametrize("param", [rn.CholeskyParameterization(), rn.LogEuclideanParameterization(), rn.DiagonalLowRankParameterization(1)])
def test_spd_parameterization_roundtrip(param):
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    assert torch.allclose(param(param.inverse(A), 2), A, atol=1e-8)
    raw = torch.randn(7, param.n_params(2))
    g = param(raw, 2)
    assert (torch.linalg.eigvalsh(g) > 0).all()


def test_pullback_metric_paraboloid():
    F = lambda p: torch.stack([p[..., 0], p[..., 1], p[..., 0] ** 2 + p[..., 1] ** 2], -1)
    pb = rn.PullbackMetric(F, eps=0.0)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]])
    assert torch.allclose(pb(x)[0], torch.eye(2))
    K = pb.gauss_curvature(x)
    assert torch.allclose(K[0], torch.tensor(4.0), atol=1e-8)  # K = 4/(1+4r²)² at r=0
    assert torch.allclose(K[1], torch.tensor(4.0 / (1 + 4 * 0.25) ** 2), atol=1e-8)


def test_constant_metric_and_conformal_metric_shapes():
    A = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    cm = rn.ConstantMetric(A)
    x = torch.rand(4, 2)
    assert torch.allclose(cm(x), A.expand(4, 2, 2), atol=1e-10)
    assert torch.allclose(cm.christoffel(x), torch.zeros(4, 2, 2, 2), atol=1e-12)
    conf = rn.ConformalMetric(lambda x: 1 + x[..., 0] ** 2)
    assert conf(x).shape == (4, 2, 2)
    assert torch.allclose(conf.volume_element(x), 1 + x[:, 0] ** 2)
    assert torch.allclose(conf.condition_number(x), torch.ones(4))


def test_image_induced_metric_is_identity_on_flat_image():
    dom = rn.Plane()
    img = rn.Image(torch.full((1, 16, 16), 0.5), dom)
    gm = rn.image_induced_metric(img, lam=10.0, sigma=0.0)
    assert torch.allclose(gm.node_metric(), torch.eye(2).expand(16, 16, 2, 2), atol=1e-10)


def test_image_induced_metric_penalises_crossing_edges():
    dom = rn.Plane()
    pts = dom.grid(32)
    img = rn.Image((pts[..., 0] > 0.5).to(torch.float64)[None], dom)  # vertical edge
    gm = rn.image_induced_metric(img, lam=10.0, sigma=1.0)
    g = gm(torch.tensor([[0.5, 0.5]]))[0]
    assert g[0, 0] > 5 * g[1, 1]  # x-direction (across the edge) is expensive
    assert torch.allclose(g[1, 1], torch.tensor(1.0), atol=1e-6)
