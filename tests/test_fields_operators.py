import math

import torch

import riemann_and_sons as rn
from riemann_and_sons.fields import gaussian_blur


def test_image_array_roundtrip_and_domain_aspect():
    arr = torch.rand(20, 30, 3)
    img = rn.Image.from_array(arr)
    assert img.values.shape == (3, 30, 20)
    assert torch.allclose(img.to_array(), arr)
    assert math.isclose(img.domain.hi[1].item(), 20 / 30)


def test_field_sampling_matches_nodes():
    dom = rn.Plane()
    f = rn.Field.from_function(dom, 9, lambda p: torch.sin(p[..., 0]) + p[..., 1])
    assert torch.allclose(f.sample(f.points), f.values, atol=1e-12)


def test_gaussian_blur_preserves_constants():
    dom = rn.Plane()
    f = rn.Field(torch.full((12, 12), 3.0), dom)
    assert torch.allclose(gaussian_blur(f, 1.5).values, f.values)


def test_laplace_beltrami_constant_metric_exact_on_quadratics():
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    dom = rn.Plane()
    geo = rn.Geometry(dom, rn.ConstantMetric(A))
    f = rn.Field.from_function(dom, 41, lambda p: p[..., 0] ** 2 + 3 * p[..., 0] * p[..., 1] - 2 * p[..., 1] ** 2)
    lap = rn.laplace_beltrami(f, geo).values
    H = torch.tensor([[2.0, 3.0], [3.0, -4.0]])
    expected = (torch.linalg.inv(A) * H).sum()
    assert torch.allclose(lap[2:-2, 2:-2], expected.expand(37, 37), atol=1e-9)


def test_laplace_beltrami_conformal_factor(hyperbolic_geometry):
    dom = rn.Box([0.0, 0.5], [1.0, 1.5])
    geo = rn.Geometry(dom, hyperbolic_geometry.metric)
    f = rn.Field.from_function(dom, 81, lambda p: torch.sin(3 * p[..., 0]) * torch.exp(p[..., 1]))
    lap = rn.laplace_beltrami(f, geo).values
    pts = f.points
    true = pts[..., 1] ** 2 * (-8) * torch.sin(3 * pts[..., 0]) * torch.exp(pts[..., 1])  # Δ_g = y² Δ
    rel = (lap - true)[3:-3, 3:-3].abs().max() / true.abs().max()
    assert rel < 5e-4


def test_laplace_beltrami_periodic_eigenfunction():
    tor = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    geo = rn.Geometry(tor)
    f = rn.Field.from_function(tor, 64, lambda p: torch.sin(2 * math.pi * p[..., 0]) * torch.cos(2 * math.pi * p[..., 1]))
    lap = rn.laplace_beltrami(f, geo).values
    rel = (lap + 8 * math.pi**2 * f.values).abs().max() / (8 * math.pi**2)
    assert rel < 2e-3


def test_riemannian_gradient_uses_inverse_metric():
    A = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    dom = rn.Plane()
    geo = rn.Geometry(dom, rn.ConstantMetric(A))
    f = rn.Field.from_function(dom, 21, lambda p: p[..., 0] + p[..., 1])
    rg = rn.riemannian_gradient(f, geo)
    assert torch.allclose(rg[5:-5, 5:-5], torch.tensor([0.25, 1.0]).expand(11, 11, 2), atol=1e-10)


def test_divergence_of_constant_field_is_zero_euclidean():
    dom = rn.Plane()
    geo = rn.Geometry(dom)
    V = torch.ones(16, 16, 2)
    assert torch.allclose(rn.divergence(V, geo).values, torch.zeros(16, 16), atol=1e-12)


def test_multichannel_operators_broadcast():
    dom = rn.Plane()
    img = rn.Image(torch.rand(3, 16, 16), dom)
    assert rn.gradient(img).shape == (3, 16, 16, 2)
    assert rn.laplace_beltrami(img, rn.Geometry(dom)).values.shape == (3, 16, 16)
