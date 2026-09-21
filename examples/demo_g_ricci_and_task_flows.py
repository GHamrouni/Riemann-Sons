"""Demo G -- metric flows: Ricci flow on a torus, then task-driven and hybrid flows.

Part 1.  Ricci flow  ∂_t g = −2 Ric(g)  on a flat torus with a conformal bump
g = e^{2u} I.  Known behaviour (used as a test):
  * Gauss–Bonnet: ∫ R dA = 0 for all t,
  * volume is conserved (dV/dt = −∫R dA = 0),
  * curvature diffuses: max|R| → 0, the metric flattens.
In 2-D Ric = (R/2) g, so the flow is a pointwise conformal rescaling -- it can
smooth curvature but never creates anisotropy.  Coordinates are held fixed.

Part 2.  Task-driven flow  dθ/dt = −∇_θ E(g_θ)  with E = triplet loss + prior
(the explicit parameter-space form of ∂_t g = −grad_g E), and a hybrid flow
   ∂_t g = −2λ Ric(g) + F_task(g)
realised by Lie splitting.  Whether the Ricci term helps is an empirical
question; here it is just made observable.
"""

import math

import torch

from _common import OUT, plt, rn, save, timer
from riemann_and_sons import learning, plot, synthetic

torch.manual_seed(0)

# --------------------------------------------------------------------------- Part 1
tor = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
gm = rn.GridMetric(tor, 40, parameterization=rn.ConformalParameterization())
nodes = gm.nodes


def periodic_bump(center, sigma, height):
    d = torch.remainder(nodes - torch.tensor(center) + 0.5, 1.0) - 0.5  # wrap-around displacement on the torus
    return height * torch.exp(-(d**2).sum(-1) / (2 * sigma**2))


u = periodic_bump([0.5, 0.5], 0.12, 0.5) + periodic_bump([0.15, 0.85], 0.1, -0.35)
gm.set_node_metric(torch.exp(2 * u)[..., None, None] * torch.eye(2))
flow = rn.RicciFlow()
dt = flow.stable_dt(gm)
with timer(f"Ricci flow on the torus, dt = {dt:.2e}"):
    traj = flow.integrate(gm, t_end=300 * dt, dt=dt, record_every=30)
for k, v in traj.diagnostics.items():
    print(f"    {k:16s} " + " ".join(f"{x:9.4f}" for x in v[:: max(1, len(v) // 6)]))

fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
for c, i in enumerate([0, 2, 5, len(traj) - 1]):
    g_i = rn.Geometry(tor, traj.at(i))
    plot.metric_scalar(g_i, "volume", ax=axes[0, c], resolution=80, cmap="cividis", title=f"t = {traj.times[i]:.4f}   √det g")
    plot.curvature(g_i, ax=axes[1, c], resolution=80, vmax=traj.diagnostics["max_abs_R"][0], robust=None, title="scalar curvature R")
save(fig, "demo_g_ricci_flow.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
t = traj.times
axes[0].plot(t, traj.diagnostics["max_abs_R"], marker="o"); axes[0].set_title("max |R|  (curvature smooths out)"); axes[0].set_xlabel("t")
axes[1].plot(t, traj.diagnostics["volume"], marker="o"); axes[1].set_title("Riemannian volume (conserved)"); axes[1].set_xlabel("t")
axes[2].plot(t, traj.diagnostics["total_curvature"], marker="o"); axes[2].set_title("∫ R dA  (Gauss–Bonnet: 0 on a torus)"); axes[2].set_xlabel("t")
save(fig, "demo_g_ricci_diagnostics.png")
with timer("Ricci flow animation"):
    rn.Inspector(rn.Geometry(tor, gm), resolution=48, ellipses=(16, 16)).animate(traj, panels=("volume", "curvature"), path=f"{OUT}/demo_g_ricci_flow.gif", fps=3)
    plt.close("all")
    print(f"  saved {OUT}/demo_g_ricci_flow.gif")

# --------------------------------------------------------------------------- Part 2
dom = rn.Plane()
pts, labels = synthetic.clustered_world(60, spread=0.09)
a_fix, p_fix, n_fix = synthetic.triplet_sampler(pts, labels, batch=512, seed=1)()  # one fixed batch -> a deterministic energy
reg_pts = dom.grid(16).reshape(-1, 2)


def energy(metric):
    d = lambda x, y: learning.straight_line_distance(metric, x, y)
    return learning.triplet_loss(d(a_fix, p_fix), d(a_fix, n_fix), margin=0.5) + 0.02 * learning.euclidean_prior(metric, reg_pts)


results = {}
for name, ricci_weight in [("task-driven flow   dθ/dt = −∇θ E", 0.0), ("hybrid   ∂t g = −2λ Ric + F_task", 0.02)]:
    metric = rn.GridMetric(dom, 12)
    task = rn.GradientFlow(energy)
    components = [(1.0, task)]
    if ricci_weight > 0:
        components.append((ricci_weight, rn.RicciFlow(min_eigenvalue=0.05, adaptive=True)))  # adaptive sub-steps keep Ricci stable
    flow = rn.HybridFlow(components)
    with timer(name):
        tr = flow.integrate(metric, t_end=30.0, dt=0.25, record_every=12)
    results[name] = (metric, tr)
    en = tr.diagnostics["gradientflow_energy"]
    print(f"    energy {en[0]:.4f} -> {en[-1]:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for r, (name, (metric, tr)) in enumerate(results.items()):
    g = rn.Geometry(dom, metric)
    plot.metric_field(g, ax=axes[r, 0], resolution=(18, 18), color_by="anisotropy", normalize="local", title=f"{name}\nmetric at t = {tr.times[-1]:.0f}")
    plot.points(pts, ax=axes[r, 0], labels=labels, s=8, cmap="coolwarm")
    plot.curvature(g, ax=axes[r, 1], resolution=64)
    axes[r, 2].plot(tr.times, tr.diagnostics["gradientflow_energy"], marker="o", label="task energy E(g)")
    axes[r, 2].set_xlabel("t"); axes[r, 2].legend(loc="upper right")
    if "ricciflow_max_abs_R" in tr.diagnostics:
        ax2 = axes[r, 2].twinx(); ax2.plot(tr.times, tr.diagnostics["ricciflow_max_abs_R"], color="C3", marker=".", label="max |R|"); ax2.set_ylabel("max |R|", color="C3")
    with torch.no_grad():
        axes[r, 2].set_title(f"E(t);   final max |R| = {g.scalar_curvature(dom.grid(48)).abs().max():.1f}")
save(fig, "demo_g_task_and_hybrid_flows.png")
