"""Regression coverage for path optimization and retrieval correctness."""

import pytest
import torch

import riemann_and_sons as rn
from riemann_and_sons import learning
from riemann_and_sons.paths import resample_path


@pytest.mark.parametrize("method", ["lbfgs", "adam"])
def test_geodesic_inner_solve_preserves_metric_gradients_and_detaches_endpoints(method):
    domain = rn.Plane()
    metric = rn.GridMetric(domain, 4)
    metric.raw.grad = torch.ones_like(metric.raw)
    before = metric.raw.grad.clone()
    x = torch.tensor([0.1, 0.2], requires_grad=True)
    y = torch.tensor([0.9, 0.7], requires_grad=True)

    path = rn.geodesic(rn.Geometry(domain, metric), x, y, n_points=8, iterations=3, method=method)

    assert torch.equal(metric.raw.grad, before)
    assert x.grad is None and y.grad is None
    assert not path.points.requires_grad
    path.length(rn.Geometry(domain, metric)).backward()
    assert torch.isfinite(metric.raw.grad).all()
    assert not torch.equal(metric.raw.grad, before)


def test_geodesic_without_interior_points_or_iterations():
    geometry = rn.Geometry(rn.Plane())
    x, y = torch.tensor([0.1, 0.2]), torch.tensor([0.9, 0.7])
    for kwargs in ({"n_points": 2}, {"iterations": 0}):
        path = rn.geodesic(geometry, x, y, **kwargs)
        assert path.info["energy_history"] == []
        assert torch.allclose(path.length(geometry), (y - x).norm())
    assert torch.allclose(learning.geodesic_distance(geometry, x[None], y[None], n_points=2), (y - x).norm()[None])


def test_resample_preserves_endpoints_at_small_coordinate_scales():
    points = torch.tensor([[0.0, 0.0], [1e-14, 0.0]])
    sampled = resample_path(points, 3)
    assert torch.equal(sampled[0], points[0])
    assert torch.equal(sampled[-1], points[-1])
    assert torch.allclose(sampled[1], points[-1] / 2, atol=0, rtol=1e-12)


def test_periodic_distance_field_wraps_sources_and_does_not_add_duplicate_edges():
    geometry = rn.Geometry(rn.Box([0.0], [1.0], periodic=True))
    for source in (0.0, 1.0, 2.0):
        field = rn.distance_field(geometry, torch.tensor([source]), resolution=2)
        assert torch.allclose(field.values, torch.tensor([0.0, 0.5]))
        assert field.nearest_node(torch.tensor([source])) == 0


def test_distance_field_path_returns_the_source_selected_on_the_grid():
    geometry = rn.Geometry(rn.Plane())
    sources = torch.tensor([[0.0, 0.49], [0.1, 0.0]])
    field = rn.distance_field(geometry, sources, resolution=(11, 2))
    path = field.path_to(torch.tensor([0.0, 0.0]))
    assert torch.equal(path.end, sources[0])


@pytest.mark.parametrize("method", ["grid", "graph", "straight"])
def test_retrieval_paths_always_run_from_query_to_hit(method):
    points = torch.tensor([[0.5, 0.5], [0.75, 0.5]])
    query = torch.tensor([0.25, 0.5])
    retriever = rn.GeodesicRetriever(points, rn.Geometry(rn.Plane()), method=method, resolution=5, k_graph=1)
    result = retriever.query(query, k=2, with_paths=True)
    for index, path in zip(result.indices[0], result.paths[0]):
        assert torch.equal(path.start, query)
        assert torch.equal(path.end, points[index])


def test_disconnected_graph_does_not_fabricate_paths():
    points = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [1.0, 0.0]])
    retriever = rn.GeodesicRetriever(points, rn.Geometry(rn.Plane()), method="graph", k_graph=1)
    result = retriever.query(points[0], k=4)
    assert torch.isinf(result.distances[0, 2:]).all()
    with pytest.raises(ValueError, match="unreachable"):
        retriever.query(points[0], k=4, with_paths=True)


def test_straight_distance_is_zero_for_coincident_points_and_has_finite_gradient():
    domain = rn.Plane()
    metric = rn.GridMetric(domain, 4)
    point = torch.tensor([[0.3, 0.4]])
    distance = learning.straight_line_distance(metric, point, point)
    assert torch.equal(distance, torch.zeros_like(distance))
    distance.sum().backward()
    assert torch.isfinite(metric.raw.grad).all()
    assert torch.equal(metric.raw.grad, torch.zeros_like(metric.raw.grad))


def test_metric_learning_rejects_unknown_optimizer():
    geometry = rn.Geometry(rn.Plane(), rn.GridMetric(rn.Plane(), 4))
    with pytest.raises(ValueError, match="optimizer"):
        learning.fit_metric(geometry, lambda: None, epochs=0, optimizer="adma")
