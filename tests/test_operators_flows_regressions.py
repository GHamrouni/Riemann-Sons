"""Numerical contracts for field integration, diffusion and transformations."""

import math

import pytest
import torch

import riemann_and_sons as rn
from riemann_and_sons.fields import gaussian_blur
from riemann_and_sons.flows import stable_diffusion_dt


def test_integer_field_conversion_preserves_fractional_domain():
    field = rn.Field(torch.ones(3), rn.Box([0.0], [0.5])).to(dtype=torch.int64)
    assert field.values.dtype == torch.int64
    assert field.domain.hi.item() == 0.5
    assert torch.allclose(field.integral(), torch.tensor(0.5))


@pytest.mark.parametrize("periodic", [False, True])
def test_constant_integrals_equal_physical_volume(periodic):
    domain = rn.Box([0, 0], [2, 3], periodic=periodic)
    field = rn.Field(torch.ones(2, 3, 4), domain)
    geometry = rn.Geometry(domain, rn.ConstantMetric(torch.diag(torch.tensor([4.0, 9.0]))))
    assert torch.allclose(field.integral(), torch.full((2,), 6.0))
    assert torch.allclose(field.riemannian_integral(geometry), torch.full((2,), 36.0))


def test_neumann_laplacian_matches_cosine_at_boundary_nodes():
    domain = rn.Box([0], [1])
    field = rn.Field.from_function(domain, 17, lambda x: torch.cos(math.pi * x[..., 0]))
    op = rn.LaplaceBeltrami(rn.Geometry(domain), field.resolution)
    eigenvalue = 2 * (math.cos(math.pi / 16) - 1) * 16**2
    assert torch.allclose(op(field).values, eigenvalue * field.values, atol=1e-10)
    assert torch.allclose(op.mass.sum(), torch.tensor(1.0))


def test_diffusion_step_bound_uses_assembled_coefficients():
    # Node-only metric sampling misses the small values at cell centres.
    domain = rn.Box([0], [1], periodic=True)
    metric = rn.FunctionMetric(lambda x: torch.exp(4 * torch.cos(16 * math.pi * x[..., 0]))[..., None, None], dim=1)
    geometry = rn.Geometry(domain, metric)
    field = rn.Field(torch.tensor([1.0, -1.0]).repeat(4), domain)
    op = rn.LaplaceBeltrami(geometry, 8)
    dt = stable_diffusion_dt(field, geometry)
    root_mass = op.mass.sqrt().reshape(-1)
    symmetric = root_mass[:, None] * op.matrix() / root_mass[None, :]
    largest_rate = -torch.linalg.eigvalsh(symmetric).min().item()
    assert dt <= 2 / largest_rate
    out = rn.diffuse(field, geometry, 0.02)
    assert op.energy(out.values) < op.energy(field.values)
    assert torch.allclose(out.riemannian_integral(geometry), field.riemannian_integral(geometry), atol=1e-10)


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_flows_reject_invalid_steps(dt):
    domain = rn.Plane()
    field = rn.Field(torch.ones(3, 3), domain)
    with pytest.raises(ValueError, match="dt"):
        rn.diffuse(field, rn.Geometry(domain), 0.1, dt=dt)
    with pytest.raises(ValueError, match="dt"):
        rn.MetricFlow().integrate(rn.GridMetric(domain, 3), 0.1, dt=dt)


@pytest.mark.parametrize("record_every", [0, -1, 1.5])
def test_flows_reject_invalid_record_intervals(record_every):
    domain = rn.Plane()
    with pytest.raises(ValueError, match="record_every"):
        rn.diffuse(rn.Field(torch.ones(3, 3), domain), rn.Geometry(domain), 0.1, record_every=record_every)
    with pytest.raises(ValueError, match="record_every"):
        rn.MetricFlow().integrate(rn.GridMetric(domain, 3), 0.1, 0.01, record_every=record_every)


def test_scalar_sampling_and_field_dtype_conversion():
    domain = rn.Plane()
    field = rn.Field.from_function(domain, 3, lambda x: x.sum(-1))
    assert field.sample(torch.tensor([0.25, 0.5])).shape == ()
    assert field.sample(torch.tensor([0.25, 0.5])).item() == pytest.approx(0.75)
    for converted in [field.to(torch.float32), field.to(dtype=torch.float32)]:
        assert converted.dtype == converted.domain.lo.dtype == torch.float32
        assert converted.sample(torch.tensor([0.25, 0.5], dtype=torch.float32)).item() == pytest.approx(0.75)


def test_field_function_supports_multiple_channel_axes():
    domain = rn.Plane()
    field = rn.Field.from_function(domain, (3, 4), lambda x: x[..., :, None] * torch.tensor([1.0, 2.0, 3.0]))
    assert field.values.shape == (2, 3, 3, 4)
    assert field.sample(field.points).shape == (3, 4, 2, 3)
    assert torch.allclose(field.sample(field.points), field.points[..., :, None] * torch.tensor([1.0, 2.0, 3.0]))


def test_wide_periodic_blur_commutes_with_translation_and_preserves_mass():
    domain = rn.Box([0], [1], periodic=True)
    values = torch.tensor([1.0, 2.0, 4.0])
    blurred = gaussian_blur(rn.Field(values, domain), sigma=3.0)
    shifted = gaussian_blur(rn.Field(values.roll(1), domain), sigma=3.0)
    assert torch.allclose(shifted.values, blurred.values.roll(1), atol=1e-12)
    assert torch.allclose(blurred.integral(), rn.Field(values, domain).integral())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tps_is_finite_at_landmarks_and_preserves_dtype(dtype):
    sources = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]], dtype=dtype)
    targets = sources.clone()
    targets[-1] += 0.1
    transform = rn.ThinPlateSpline(sources, targets)
    assert transform.sources.dtype == dtype
    assert torch.allclose(transform(sources), targets, atol=1e-6)
    assert torch.isfinite(transform.jacobian(sources)).all()
    affine = rn.Affine(torch.eye(2, dtype=dtype))
    assert affine.A.dtype == dtype
    assert torch.allclose(affine.inverse(sources), sources)


def test_periodic_warp_wraps_even_when_fill_is_set():
    domain = rn.Box([0], [1], periodic=True)
    field = rn.Field(torch.arange(4.0), domain)
    shift = rn.Affine(torch.ones(1, 1), torch.tensor([0.25]))
    warped = rn.warp(field, shift, fill=-100)
    assert torch.allclose(warped.values, field.values.roll(1))


def test_singular_jacobian_has_infinite_condition_number():
    transform = rn.Affine(torch.zeros(2, 2))
    stats = rn.jacobian_stats(transform, torch.tensor([[0.5, 0.5]]))
    assert torch.isinf(stats.condition_number).all()
    assert stats.has_folds
