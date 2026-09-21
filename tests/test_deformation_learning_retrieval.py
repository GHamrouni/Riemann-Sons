import torch

import riemann_and_sons as rn
from riemann_and_sons import deformation, learning, retrieval, synthetic


def test_affine_jacobian_inverse_and_newton():
    A = torch.tensor([[1.2, 0.3], [-0.1, 0.8]])
    aff = rn.Affine(A, torch.tensor([0.05, -0.02]))
    x = torch.rand(10, 2)
    assert torch.allclose(aff.jacobian(x), A.expand(10, 2, 2))
    assert torch.allclose(aff.inverse(aff(x)), x, atol=1e-12)
    assert torch.allclose(deformation.Transformation.inverse(aff, aff(x)), x, atol=1e-10)


def test_thin_plate_spline_interpolates_landmarks_and_has_no_folds():
    src, dst = synthetic.landmark_pairs("pinch")
    tps = rn.ThinPlateSpline(src, dst)
    assert torch.allclose(tps(src), dst, atol=1e-10)
    st = rn.jacobian_stats(tps, rn.Plane().grid(48))
    assert not st.has_folds and st.det.min() > 0.3


def test_jacobian_stats_detect_orientation_reversal():
    refl = rn.Affine(torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
    st = rn.jacobian_stats(refl, rn.Plane().grid(6))
    assert st.has_folds and st.fold_fraction == 1.0
    assert torch.allclose(st.det, torch.full((6, 6), -1.0))
    assert torch.allclose(st.condition_number, torch.ones(6, 6))


def test_displacement_fit_and_inverse():
    dom = rn.Plane()
    src, dst = synthetic.landmark_pairs("pinch")
    disp = rn.Displacement(dom, 10)
    hist = deformation.fit_transformation(disp, src, dst, bending=1e-4, fold=1.0, iterations=80, lr=0.03, sample_points=dom.grid(16).reshape(-1, 2))
    assert hist[-1]["landmarks"] < 0.2 * hist[0]["landmarks"]
    x = torch.rand(20, 2) * 0.8 + 0.1
    assert torch.allclose(disp(disp.inverse(x)), x, atol=1e-8)


def test_warp_identity_and_translation():
    dom = rn.Plane()
    img = synthetic.checkerboard(dom, 64, 8)
    assert torch.allclose(rn.warp(img, rn.Affine(torch.eye(2))).values, img.values, atol=1e-12)
    shift = rn.Affine(torch.eye(2), torch.tensor([16 / 63, 0.0]))  # Φ(x) = x + 16 cells along x (spacing is 1/63)
    w = rn.warp(img, shift, fill=0.0)
    assert torch.allclose(w.values[0, 16:, :], img.values[0, :-16, :], atol=1e-12)
    assert torch.all(w.values[0, :16, :] == 0)


def test_straight_line_distance_reduces_to_euclidean_for_identity():
    a, b = torch.rand(8, 2), torch.rand(8, 2)
    assert torch.allclose(learning.straight_line_distance(rn.EuclideanMetric(), a, b), (a - b).norm(dim=-1))


def test_geodesic_distance_upper_bounded_by_straight_line_and_differentiable():
    dom = rn.Plane()
    gm = rn.GridMetric(dom, 8)
    with torch.no_grad():
        gm.raw.add_(0.4 * torch.randn_like(gm.raw))
    geo = rn.Geometry(dom, gm)
    a, b = dom.sample_uniform(6), dom.sample_uniform(6)
    gd = learning.geodesic_distance(geo, a, b, n_points=12, iterations=15)
    sd = learning.straight_line_distance(gm, a, b, samples=32)
    assert (gd <= sd + 1e-6).all()
    gd.sum().backward()
    assert gm.raw.grad is not None and torch.isfinite(gm.raw.grad).all()


def test_regularizers_on_known_metrics():
    x = torch.rand(10, 2)
    A = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    cm = rn.ConstantMetric(A)
    assert torch.allclose(learning.anisotropy(cm, x), torch.tensor(3.0))
    assert torch.allclose(learning.euclidean_prior(cm, x), torch.tensor(9.0))
    assert torch.allclose(learning.volume_prior(cm, x), torch.tensor(float(torch.log(torch.tensor(4.0)) ** 2)))
    assert torch.allclose(learning.metric_smoothness(cm, x), torch.tensor(0.0), atol=1e-12)
    assert torch.allclose(learning.curvature_penalty(cm, x), torch.tensor(0.0), atol=1e-12)


def test_fit_metric_reduces_triplet_loss():
    pts, labels = synthetic.clustered_world(30)
    dom = rn.Plane()
    geo = rn.Geometry(dom, rn.GridMetric(dom, 8))
    sampler = synthetic.triplet_sampler(pts, labels, batch=64)
    hist = learning.fit_metric(geo, sampler, epochs=40, lr=0.05, margin=0.3, regularizers={"euclidean": 0.05}, reg_points=64)
    assert hist[-1]["triplet"] < 0.5 * hist[0]["triplet"]


def test_retrievers_agree_on_identity_geometry_and_geodesic_changes_neighbours():
    pts, labels = synthetic.clustered_world(30)
    q = torch.tensor([[0.42, 0.15]])  # just left of the wall at x = 0.5
    eu = rn.EuclideanRetriever(pts, labels).query(q, k=5)
    ma = rn.MahalanobisRetriever(pts, torch.eye(2)).query(q, k=5)
    st = rn.GeodesicRetriever(pts, rn.Geometry(rn.Plane()), method="straight").query(q, k=5)
    assert torch.equal(eu.indices, ma.indices) and torch.equal(eu.indices, st.indices)
    geo = rn.Geometry(rn.Plane(), synthetic.wall_metric(cost=400.0, gap=(0.7, 0.8)))
    grid = rn.GeodesicRetriever(pts, geo, method="grid", resolution=64).query(q, k=5, with_paths=True)
    graph = rn.GeodesicRetriever(pts, geo, method="graph", k_graph=8).query(q, k=5, with_paths=True)
    assert not torch.equal(grid.indices, eu.indices)
    assert (pts[grid.indices[0], 0] < 0.5).all()  # everything on the query's side of the wall
    assert (pts[graph.indices[0], 0] < 0.5).all()
    assert len(grid.paths[0]) == 5 and len(graph.paths[0]) == 5
    co = rn.CosineRetriever(pts).query(q, k=3)
    assert co.indices.shape == (1, 3)
    assert 0.0 <= retrieval.precision_at_k(eu, labels[eu.indices[0, :1]], labels) <= 1.0
