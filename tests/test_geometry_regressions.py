"""Numerical and shape contracts for domains, metrics, and curvature."""

import pytest
import torch

import riemann_and_sons as rn
from riemann_and_sons import curvature


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("resolution", [1, (3, 5)])
def test_total_volume_integrates_constant_metric_exactly(periodic, resolution):
    domain = rn.Box([0.0, -1.0], [2.0, 2.0], periodic=periodic)
    metric = rn.ConstantMetric(torch.diag(torch.tensor([4.0, 9.0])))
    assert torch.allclose(rn.Geometry(domain, metric).total_volume(resolution), torch.tensor(36.0))


def test_total_volume_uses_midpoints_and_retains_metric_gradients():
    scale = torch.tensor(2.0, requires_grad=True)
    metric = rn.ConformalMetric(lambda x: scale * (1 + x[..., 0]), dim=2)
    volume = rn.Geometry(rn.Plane(), metric).total_volume((3, 5))
    assert torch.allclose(volume, torch.tensor(3.0))
    assert torch.allclose(torch.autograd.grad(volume, scale)[0], torch.tensor(1.5))


def test_zero_vector_norm_has_exact_zero_value_and_finite_gradient():
    metric = rn.EuclideanMetric()
    vectors = torch.tensor([[0.0, 0.0], [3.0, 4.0]], requires_grad=True)
    norms = metric.norm(torch.zeros_like(vectors), vectors)
    assert torch.equal(norms, torch.tensor([0.0, 5.0]))
    gradient = torch.autograd.grad(norms.sum(), vectors)[0]
    assert torch.allclose(gradient, torch.tensor([[0.0, 0.0], [0.6, 0.8]]))


def test_metric_norm_does_not_hide_nan_values():
    metric = rn.ConformalMetric(lambda x: x[..., 0] * float("nan"), dim=2)
    assert torch.isnan(metric.norm(torch.ones(2), torch.ones(2)))


def test_constant_metric_preserves_small_spd_matrix():
    matrix = torch.tensor([[2e-8, 1e-8], [1e-8, 3e-8]])
    metric = rn.ConstantMetric(matrix)
    assert torch.allclose(metric.matrix, matrix, rtol=1e-12, atol=1e-20)


def test_constant_metric_rejects_asymmetric_input():
    with pytest.raises(ValueError, match="symmetric"):
        rn.ConstantMetric(torch.tensor([[2.0, 0.7], [0.1, 2.0]]))


def test_explicit_float64_inputs_survive_float32_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        lo = torch.tensor([0.0, 0.0], dtype=torch.float64)
        domain = rn.Box(lo, lo + 1)
        matrix = torch.eye(2, dtype=torch.float64)
        assert domain.grid(3).dtype == torch.float64
        assert rn.ConstantMetric(matrix).matrix.dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("rank", [0, 2])
def test_low_rank_parameterization_handles_zero_and_full_rank(rank):
    parameterization = rn.DiagonalLowRankParameterization(rank)
    matrix = torch.diag(torch.tensor([2.0, 3.0]))
    if rank:
        matrix[0, 1] = matrix[1, 0] = 0.5
    raw = parameterization.inverse(matrix)
    assert raw.shape == (parameterization.n_params(2),)
    assert torch.isfinite(raw).all()
    assert torch.allclose(parameterization(raw, 2), matrix, atol=1e-12)
    grid = rn.GridMetric(rn.Plane(), 3, parameterization=parameterization)
    assert torch.allclose(grid.node_metric(), torch.eye(2).expand(3, 3, 2, 2))


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("method, trailing", [
    ("metric_gradient", (3, 3, 3)),
    ("christoffel", (3, 3, 3)),
    ("riemann", (3, 3, 3, 3)),
    ("ricci", (3, 3)),
    ("scalar_curvature", ()),
])
def test_curvature_preserves_empty_batch_shape_and_dtype(method, trailing, chunk):
    x = torch.empty(2, 0, 3, dtype=torch.float64)
    result = getattr(rn.EuclideanMetric(), method)(x, chunk=chunk)
    assert result.shape == (2, 0, *trailing)
    assert result.dtype == x.dtype and result.device == x.device


def test_gaussian_curvature_rejects_higher_dimensions():
    with pytest.raises(ValueError, match="two-dimensional"):
        rn.EuclideanMetric().gauss_curvature(torch.zeros(3))
    with pytest.raises(ValueError, match="two-dimensional"):
        curvature.gauss_curvature(rn.EuclideanMetric(), torch.zeros(4))


@pytest.mark.parametrize("resolution", [0, -2, (3, 0), (3, 2.5), (3,)])
def test_box_rejects_invalid_resolutions(resolution):
    with pytest.raises(ValueError, match="resolution"):
        rn.Plane().grid(resolution)


@pytest.mark.parametrize("lo, hi", [([], []), ([[0.0]], [[1.0]]), ([0.0], [float("inf")])])
def test_box_rejects_invalid_bounds(lo, hi):
    with pytest.raises(ValueError):
        rn.Box(lo, hi)


def test_grid_metric_conversion_moves_domain_dtype():
    metric = rn.GridMetric(rn.Plane(), 3).float()
    assert metric.raw.dtype == metric.nodes.dtype == torch.float32
    assert metric.node_metric().dtype == torch.float32
    metric.double()
    assert metric.raw.dtype == metric.nodes.dtype == torch.float64
    assert metric.node_metric().dtype == torch.float64
