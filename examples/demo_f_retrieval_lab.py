"""Demo F -- the visual retrieval laboratory.

Two noisy interleaving moons.  Euclidean k-NN mixes the classes where the
moons approach each other; a metric learned from triplets raises the cost of
crossing between the moons so that geodesic neighbours stay on the right moon.

For a few queries we show candidates, top-K, geodesic paths, iso-distance
contours and the learned metric field, and we report precision@K for
Euclidean vs learned geometry (straight-line bound and kNN-graph geodesics).
"""

import torch

from _common import plt, rn, save, timer
from riemann_and_sons import learning, plot, synthetic

torch.manual_seed(0)
dom = rn.Plane()
pts, labels = synthetic.two_moons(300, noise=0.22)
metric = rn.GridMetric(dom, 16)
geo = rn.Geometry(dom, metric)
sampler = synthetic.triplet_sampler(pts, labels, batch=128)
REG = {"euclidean": 0.01, "smoothness": 1e-4, "volume": 0.01}
with timer("learning the metric (straight-line, then geodesic fine-tune)"):
    h1 = learning.fit_metric(geo, sampler, epochs=300, lr=0.03, margin=0.4, regularizers=REG, reg_points=256)
    h2 = learning.fit_metric(geo, sampler, epochs=40, lr=0.01, margin=0.4, distance="geodesic", distance_kw={"n_points": 8, "iterations": 5}, regularizers=REG, reg_points=256)
    print(f"    triplet loss {h1[0]['triplet']:.4f} -> {h1[-1]['triplet']:.4f} -> {h2[-1]['triplet']:.4f}")

eu = rn.EuclideanRetriever(pts, labels)
ge_grid = rn.GeodesicRetriever(pts, geo, labels, method="grid", resolution=96)
ge_straight = rn.GeodesicRetriever(pts, geo, labels, method="straight")
ge_graph = rn.GeodesicRetriever(pts, geo, labels, method="graph", k_graph=10)


def loo_precision(retriever, k):
    out = retriever.query(pts, k=k + 1)
    return float((labels[out.indices[:, 1:]] == labels[:, None]).double().mean())


with timer("precision@K (leave-one-out over all 300 points)"):
    scores = {}
    for name, r in [("Euclidean", eu), ("learned, straight-line bound", ge_straight), ("learned, kNN-graph geodesic", ge_graph)]:
        scores[name] = (loo_precision(r, 10), loo_precision(r, 50))
        print(f"    {name:30s} P@10 {scores[name][0]:.3f}   P@50 {scores[name][1]:.3f}")

K = 10
# queries: for each class, the point whose Euclidean top-10 is most contaminated, excluding isolated outliers
with timer("choosing confused queries"):
    out_e = eu.query(pts, k=K + 1)
    wrong_e = (labels[out_e.indices[:, 1:]] != labels[:, None]).sum(1)
    nn_same = torch.stack([torch.cdist(pts[i : i + 1], pts[(labels == labels[i]) & (torch.arange(len(pts)) != i)]).min() for i in range(len(pts))])
    tips = []
    for c in (0, 1):
        ok = (labels == c) & (nn_same < nn_same[labels == c].median())
        cand = torch.nonzero(ok)[:, 0]
        tips.append(pts[cand[int(torch.argmax(wrong_e[cand]))]])
fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
for r, q in enumerate(tips):
    q_label = labels[(pts == q).all(-1)][0]
    with timer(f"query {r}"):
        res_e = eu.query(q[None], k=K)
        res_g = ge_grid.query(q[None], k=K, with_paths=True)
        df = ge_grid._field(q)
    for c, (res, title) in enumerate([(res_e, "Euclidean top-10"), (res_g, "learned-geodesic top-10 (+ geodesics)")]):
        ax = axes[r, c]
        plot.points(pts, ax=ax, labels=labels, s=12, cmap="coolwarm", alpha=0.6)
        hits = pts[res.indices[0]]
        wrong = labels[res.indices[0]] != q_label
        ax.scatter(hits[:, 0], hits[:, 1], facecolors="none", edgecolors="k", s=90, linewidths=1.5, zorder=5)
        ax.scatter(hits[wrong, 0], hits[wrong, 1], marker="x", c="k", s=70, zorder=6)
        ax.scatter(q[0], q[1], marker="*", s=260, c="gold", edgecolors="k", zorder=7)
        if res.paths is not None:
            plot.geodesics(res.paths[0], ax=ax, color="k", linewidth=0.8, endpoints=False)
        ax.set_title(f"{title}: {int(wrong.sum())} wrong")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax = axes[r, 2]
    plot.distance_field(df, ax=ax, contours=20, title="learned geodesic distance from ★  +  metric unit balls")
    plot.metric_field(geo, ax=ax, resolution=(20, 20), color="white", alpha=0.7, normalize="local")
fig.suptitle("   ".join(f"{k}: P@10 {v[0]:.2f}, P@50 {v[1]:.2f}" for k, v in scores.items()), fontsize=11)
save(fig, "demo_f_retrieval_lab.png")
