import math

import torch

import riemann_and_sons as rn
from riemann_and_sons import learning, synthetic


def test_euclidean_diffusion_matches_heat_kernel():
    dom = rn.Plane()
    geo = rn.Geometry(dom)
    s0, t = 0.05, 0.002
    blob = synthetic.gaussian_blob(dom, 64, (0.5, 0.5), s0)
    out = rn.diffuse(blob, geo, t)
    s2 = s0**2 + 2 * t
    truth = synthetic.gaussian_blob(dom, 64, (0.5, 0.5), math.sqrt(s2), amplitude=s0**2 / s2)
    assert (out.values - truth.values).abs().max() < 1e-3
    assert torch.allclose(out.integral(), blob.integral(), atol=1e-12)


def test_riemannian_diffusion_conserves_riemannian_mass():
    dom = rn.Plane()
    geo = rn.Geometry(dom, synthetic.wall_metric())
    blob = synthetic.gaussian_blob(dom, 48, (0.5, 0.5), 0.05)
    out = rn.diffuse(blob, geo, 0.003)
    assert torch.allclose(out.riemannian_integral(geo), blob.riemannian_integral(geo), atol=1e-10)


def test_diffusion_recording():
    dom = rn.Plane()
    blob = synthetic.gaussian_blob(dom, 32, (0.5, 0.5), 0.1)
    times, snaps = rn.diffuse(blob, rn.Geometry(dom), 0.001, record_every=2)
    assert len(times) == len(snaps) >= 2 and times[0] == 0.0


def test_ricci_flow_on_torus_gauss_bonnet_and_smoothing():
    tor = rn.Box([0, 0], [1, 1], periodic=True)
    gm = rn.GridMetric(tor, 24, parameterization=rn.ConformalParameterization())
    pts = gm.nodes
    u = 0.3 * torch.sin(2 * math.pi * pts[..., 0]) * torch.cos(2 * math.pi * pts[..., 1])
    gm.set_node_metric(torch.exp(2 * u)[..., None, None] * torch.eye(2))
    flow = rn.RicciFlow()
    dt = flow.stable_dt(gm)
    traj = flow.integrate(gm, t_end=30 * dt, dt=dt, record_every=10)
    d = traj.diagnostics
    assert all(abs(c) < 1e-8 for c in d["total_curvature"])  # ∫R dA = 0 on the torus
    assert abs(d["volume"][-1] - d["volume"][0]) < 1e-6 * d["volume"][0]  # dV/dt = −∫R = 0
    assert d["max_abs_R"][-1] < 0.8 * d["max_abs_R"][0]
    assert len(traj) == 4 and isinstance(traj.at(0), rn.GridMetric)


def test_ricci_flow_in_2d_is_conformal_for_any_parameterization():
    tor = rn.Box([0, 0], [1, 1], periodic=True)
    pts = tor.grid(16)
    u = 0.3 * torch.sin(2 * math.pi * pts[..., 0]) * torch.cos(2 * math.pi * pts[..., 1])
    g0 = torch.exp(2 * u)[..., None, None] * torch.eye(2)
    a = rn.GridMetric.from_tensor_field(tor, g0, parameterization=rn.ConformalParameterization())
    b = rn.GridMetric.from_tensor_field(tor, g0)  # Cholesky
    dt = rn.RicciFlow().stable_dt(a)
    rn.RicciFlow().integrate(a, 5 * dt, dt)
    rn.RicciFlow().integrate(b, 5 * dt, dt)
    assert torch.allclose(a.node_metric(), b.node_metric(), atol=1e-6)


def test_gradient_flow_decreases_energy_and_hybrid_runs():
    dom = rn.Plane()
    gm = rn.GridMetric(dom, 5)
    with torch.no_grad():
        gm.raw.add_(0.3 * torch.randn_like(gm.raw))
    pts = dom.grid(10)
    energy = lambda m: learning.euclidean_prior(m, pts)
    flow = rn.GradientFlow(energy)
    traj = flow.integrate(gm, 1.0, 0.1, record_every=2)
    e = traj.diagnostics["energy"]
    assert all(b < a for a, b in zip(e[:-1], e[1:]))
    hy = rn.HybridFlow([(1.0, flow), (0.0, rn.RicciFlow())])
    hy.step(gm, 0.1)
    assert learning.euclidean_prior(gm, pts) < e[-1]
