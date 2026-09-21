import math

import pytest
import torch

import riemann_and_sons as rn


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def sphere_geometry():
    """Unit sphere in polar coordinates: g = diag(1, sin²θ)."""
    metric = rn.FunctionMetric(
        lambda x: torch.diag_embed(torch.stack([torch.ones_like(x[..., 0]), torch.sin(x[..., 0]) ** 2], -1)), dim=2
    )
    return rn.Geometry(rn.Box([0.1, -10.0], [math.pi - 0.1, 10.0]), metric)


@pytest.fixture
def hyperbolic_geometry():
    """Poincaré half-plane: g = I / y²  (K = −1, R = −2)."""
    return rn.Geometry(rn.Box([-3.0, 0.05], [3.0, 4.0]), rn.ConformalMetric(lambda x: 1.0 / x[..., 1] ** 2, dim=2))


@pytest.fixture
def stereographic_sphere():
    """Unit sphere in stereographic coordinates: g = 4 I / (1 + r²)²  (K = +1, R = +2)."""
    return rn.ConformalMetric(lambda x: 4.0 / (1 + (x**2).sum(-1)) ** 2, dim=2)
