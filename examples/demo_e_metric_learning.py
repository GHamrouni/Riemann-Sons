"""Demo E -- watch a 2-D metric being learned.

World: two noisy interleaving moons (labels 0/1).  Supervision: random triplets
(x, x⁺, x⁻).  Model: g_θ(x) on a 16×16 grid (Cholesky parameterisation, cubic
B-spline).  Loss: hinge on Riemannian distances

    L = mean max(0, m + d_g(x, x⁺) − d_g(x, x⁻))  +  regularisers,

first with the cheap straight-line distance (an upper bound of d_g), then a
fine-tuning phase with true variational geodesic distances, which closes the
"loopholes" that straight segments cannot see.

Regularisers matter: without a volume/Euclidean prior the margin can be met by
inflating *all* distances uniformly.

Outputs: metric ellipses at successive epochs, loss curve, geodesics and
nearest neighbours before/after, and an animation of the geometry.
"""

import torch

from _common import OUT, plt, rn, save, timer
from riemann_and_sons import learning, plot, synthetic

torch.manual_seed(0)
dom = rn.Plane()
pts, labels = synthetic.two_moons(300, noise=0.22)
metric = rn.GridMetric(dom, 16)
geo = rn.Geometry(dom, metric)
sampler = synthetic.triplet_sampler(pts, labels, batch=128)
REG = {"euclidean": 0.01, "smoothness": 1e-4, "volume": 0.01}

snap_epochs = [0, 5, 10, 20, 40, 80, 150, 300, 340]
traj = rn.MetricTrajectory(metric)
traj.record(0.0)
history = []


def callback(epoch, rec):
    e = epoch + 1 + (300 if rec.get("phase") == "geodesic" else 0)
    if e in snap_epochs:
        traj.record(float(e))
    if epoch % 50 == 0:
        print(f"    epoch {e:3d}: " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "phase"))


with timer("phase 1: straight-line distances (300 epochs)"):
    history += learning.fit_metric(geo, sampler, epochs=300, lr=0.03, margin=0.4, distance="straight", regularizers=REG, reg_points=256, callback=callback)
with timer("phase 2: geodesic distances (40 epochs)"):
    def cb2(epoch, rec):
        rec["phase"] = "geodesic"
        callback(epoch, rec)
    history += learning.fit_metric(geo, sampler, epochs=40, lr=0.01, margin=0.4, distance="geodesic", distance_kw={"n_points": 8, "iterations": 5}, regularizers=REG, reg_points=256, callback=cb2)

# ---- 1. the geometry being learned
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, (t, m) in zip(axes.flat, list(traj)[:8]):
    plot.metric_field(rn.Geometry(dom, m), ax=ax, resolution=(20, 20), color_by="anisotropy", normalize="local", title=f"epoch {int(t)}  (unit balls, coloured by λmax/λmin)")
    plot.points(pts, ax=ax, labels=labels, s=8, cmap="coolwarm")
save(fig, "demo_e_metric_epochs.png")

# ---- 2. what changed
fig, axes = plt.subplots(1, 4, figsize=(21, 5))
axes[0].plot([h["triplet"] for h in history], label="triplet"); axes[0].plot([h["total"] for h in history], label="total")
axes[0].axvline(300, color="k", ls="--", lw=0.8); axes[0].text(302, max(h["total"] for h in history) * 0.8, "geodesic\nfine-tune", fontsize=8)
axes[0].set_xlabel("epoch"); axes[0].set_yscale("log"); axes[0].legend(); axes[0].set_title("loss")

plot.metric_scalar(geo, "logdet", ax=axes[1], resolution=96, cmap="magma", title="learned  log det g  (bright = expensive)")
plot.points(pts, ax=axes[1], labels=labels, s=8, cmap="coolwarm")

# anchor: a class-0 point whose Euclidean top-10 is contaminated but which is not an isolated outlier
eu = rn.EuclideanRetriever(pts, labels)
out_e = eu.query(pts, k=11)
wrong_e = (labels[out_e.indices[:, 1:]] != labels[:, None]).sum(1)
nn_same = torch.stack([torch.cdist(pts[i : i + 1], pts[(labels == labels[i]) & (torch.arange(len(pts)) != i)]).min() for i in range(len(pts))])
ok = (labels == 0) & (nn_same < nn_same[labels == 0].median())
cand = torch.nonzero(ok)[:, 0]
tip0 = pts[cand[int(torch.argmax(wrong_e[cand]))]]
# targets: one point on each moon at (nearly) the same Euclidean distance ~0.3 from the anchor
D_TARGET = 0.3
other_cands = pts[labels == 1]
tip1 = other_cands[int(torch.argmin(((other_cands - tip0).norm(dim=-1) - D_TARGET).abs()))]
same_cands = pts[labels == 0]
target_same = same_cands[int(torch.argmin(((same_cands - tip0).norm(dim=-1) - D_TARGET).abs()))]
with timer("geodesics under the learned metric"):
    same = rn.geodesic(geo, tip0, target_same, n_points=64, init="graph", graph_resolution=96)
    other = rn.geodesic(geo, tip0, tip1, n_points=64, init="graph", graph_resolution=96)
    print(f"    same moon: d_g {same.length(geo).item():.3f} (Euclid {same.euclidean_length().item():.3f})   other moon: d_g {other.length(geo).item():.3f} (Euclid {other.euclidean_length().item():.3f})")
plot.curvature(geo, ax=axes[2], resolution=64)
plot.points(pts, ax=axes[2], labels=labels, s=6, cmap="coolwarm", alpha=0.5)
plot.geodesics([same], ax=axes[2], color="C2", label=f"same moon   d_g = {same.length(geo).item():.2f}  (Euclid {same.euclidean_length().item():.2f})")
plot.geodesics([other], ax=axes[2], color="k", label=f"other moon  d_g = {other.length(geo).item():.2f}  (Euclid {other.euclidean_length().item():.2f})")
axes[2].legend(loc="lower left", fontsize=8); axes[2].set_title("learned scalar curvature + geodesics")

with timer("nearest neighbours: Euclidean vs learned geodesic"):
    ge = rn.GeodesicRetriever(pts, geo, labels, method="grid", resolution=96)
    res_e, res_g = eu.query(tip0[None], k=10), ge.query(tip0[None], k=10, with_paths=True)
    def loo(retriever, k):
        out = retriever.query(pts, k=k + 1)
        return float((labels[out.indices[:, 1:]] == labels[:, None]).double().mean())
    scores = {"Euclidean": (loo(eu, 10), loo(eu, 50)), "learned": (loo(rn.GeodesicRetriever(pts, geo, labels, method="graph", k_graph=10), 10), loo(rn.GeodesicRetriever(pts, geo, labels, method="graph", k_graph=10), 50))}
    for k, (p10, p50) in scores.items():
        print(f"    {k:10s} P@10 {p10:.3f}   P@50 {p50:.3f}")
ax = axes[3]
plot.points(pts, ax=ax, labels=labels, s=8, cmap="coolwarm", alpha=0.5)
he, hg = pts[res_e.indices[0]], pts[res_g.indices[0]]
ax.scatter(he[:, 0], he[:, 1], marker="s", facecolors="none", edgecolors="C0", s=110, linewidths=1.5, label="Euclidean top-10")
ax.scatter(hg[:, 0], hg[:, 1], facecolors="none", edgecolors="k", s=70, linewidths=1.5, label="learned-geodesic top-10")
plot.geodesics(res_g.paths[0], ax=ax, color="k", linewidth=0.8, endpoints=False)
ax.scatter(tip0[0], tip0[1], marker="*", s=260, c="gold", edgecolors="k", zorder=7)
ax.legend(loc="lower left", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title("neighbours of ★\n" + "   ".join(f"{k}: P@10 {v[0]:.2f}, P@50 {v[1]:.2f}" for k, v in scores.items()), fontsize=9)
save(fig, "demo_e_metric_learning_result.png")

with timer("animation"):
    rn.Inspector(geo, resolution=48, ellipses=(20, 20)).animate(traj, panels=("metric", "det"), path=f"{OUT}/demo_e_metric_learning.gif", fps=2, panel_kw={})
    plt.close("all")
    print(f"  saved {OUT}/demo_e_metric_learning.gif")
