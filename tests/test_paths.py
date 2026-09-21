import math

import torch

import riemann_and_sons as rn


def test_euclidean_geodesic_is_straight():
    geo = rn.Geometry(rn.Plane())
    p = rn.geodesic(geo, torch.tensor([0.1, 0.1]), torch.tensor([0.9, 0.7]), n_points=20)
    assert torch.allclose(p.length(geo), torch.tensor(1.0), atol=1e-8)
    s = torch.linspace(0, 1, 20)[:, None]
    assert torch.allclose(p.points, torch.tensor([0.1, 0.1]) + s * torch.tensor([0.8, 0.6]), atol=1e-6)


def test_sphere_shooting_equator_and_meridian(sphere_geometry):
    xs, vs = rn.geodesic_shoot(sphere_geometry, torch.tensor([math.pi / 2, 0.0]), torch.tensor([0.0, 1.0]), t=2.0, steps=200)
    assert torch.allclose(xs[:, 0], torch.full((201,), math.pi / 2), atol=1e-10)
    assert torch.allclose(xs[-1, 1], torch.tensor(2.0), atol=1e-10)
    xs, _ = rn.geodesic_shoot(sphere_geometry, torch.tensor([math.pi / 2, 0.0]), torch.tensor([1.0, 0.0]), t=1.0, steps=200)
    assert torch.allclose(xs[-1], torch.tensor([math.pi / 2 + 1, 0.0]), atol=1e-10)


def test_shooting_conserves_speed(sphere_geometry):
    x0 = torch.tensor([1.0, 0.3])
    v0 = torch.tensor([0.4, 0.7])
    xs, vs = rn.geodesic_shoot(sphere_geometry, x0, v0, t=1.5, steps=300)
    speeds = sphere_geometry.norm(xs, vs)
    assert torch.allclose(speeds, speeds[0].expand_as(speeds), atol=1e-8)


def test_exp_map_batched(sphere_geometry):
    x0 = torch.tensor([[math.pi / 2, 0.0], [1.0, 0.0]])
    v0 = torch.tensor([[0.0, 0.5], [0.3, 0.2]])
    out = rn.exp_map(sphere_geometry, x0, v0, steps=50)
    assert out.shape == (2, 2)
    assert torch.allclose(out[0], torch.tensor([math.pi / 2, 0.5]), atol=1e-9)


def test_hyperbolic_geodesic_distance_and_shape(hyperbolic_geometry):
    p, q = torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 1.0])
    path = rn.geodesic(hyperbolic_geometry, p, q, n_points=64)
    d_true = math.acosh(1 + 4 / 2)  # arccosh(1 + |p−q|² / (2 y_p y_q))
    assert abs(path.length(hyperbolic_geometry).item() - d_true) < 2e-4
    # the geodesic is the semicircle centred at (0,0) of radius √2: apex at y = √2
    assert abs(path.points[:, 1].max().item() - math.sqrt(2)) < 2e-3


def test_distance_field_matches_hyperbolic_formula(hyperbolic_geometry):
    p, q = torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 1.0])
    df = rn.distance_field(hyperbolic_geometry, p[None], resolution=(121, 80))
    d_true = math.acosh(3.0)
    assert abs(df.sample(q[None]).item() - d_true) / d_true < 0.02
    path = df.path_to(q)
    assert torch.allclose(path.start, q) and torch.allclose(path.end, p)


def test_graph_initialised_geodesic_matches_line_initialised(hyperbolic_geometry):
    p, q = torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 1.0])
    a = rn.geodesic(hyperbolic_geometry, p, q, n_points=48, init="line")
    b = rn.geodesic(hyperbolic_geometry, q, p, n_points=48, init="graph", graph_resolution=48)
    assert abs(a.length(hyperbolic_geometry).item() - b.length(hyperbolic_geometry).item()) < 1e-5


def test_geodesic_avoids_expensive_wall():
    from riemann_and_sons import synthetic

    geo = rn.Geometry(rn.Plane(), synthetic.wall_metric(cost=400.0, gap=(0.7, 0.8)))
    path = rn.geodesic(geo, torch.tensor([0.2, 0.2]), torch.tensor([0.8, 0.2]), n_points=64, init="graph", graph_resolution=64)
    crossing = path.points[(path.points[:, 0] - 0.5).abs() < 0.03]
    assert crossing.shape[0] > 0 and crossing[:, 1].min() > 0.6  # it crosses through the gap


def test_parallel_transport_holonomy_on_sphere(sphere_geometry):
    th0 = 1.0
    phi = torch.linspace(0, 2 * math.pi, 400)
    loop = torch.stack([torch.full_like(phi, th0), phi], -1)
    v0 = torch.tensor([1.0, 0.0])
    vT = rn.parallel_transport(sphere_geometry, loop, v0, substeps=2)[-1]
    assert torch.allclose(sphere_geometry.norm(loop[-1], vT), torch.tensor(1.0), atol=1e-8)
    angle = math.atan2(math.sin(th0) * vT[1].item(), vT[0].item())  # in the orthonormal frame
    expected = 2 * math.pi * math.cos(th0)  # holonomy of a latitude circle, up to orientation and mod 2π
    assert abs(math.cos(angle) - math.cos(expected)) < 1e-6 and abs(abs(math.sin(angle)) - abs(math.sin(expected))) < 1e-6


def test_path_length_is_differentiable_wrt_metric():
    dom = rn.Plane()
    gm = rn.GridMetric(dom, 6)
    geo = rn.Geometry(dom, gm)
    path = rn.geodesic(geo, torch.tensor([0.1, 0.1]), torch.tensor([0.9, 0.9]), n_points=16)
    path.length(geo).backward()
    assert gm.raw.grad is not None and torch.isfinite(gm.raw.grad).all()
