"""Demo F -- the visual retrieval laboratory, evaluated out of sample.

Two noisy interleaving moons.  For each of several seeds the points are split
into a *training* set (whose labels supervise the metric) and *held-out
queries* (never seen during training).  Retrieval fetches training points for
each held-out query and precision@K is averaged over seeds.

Geometries compared on the same fixed points:
  * Euclidean                     ``‖x − y‖``
  * learned constant Mahalanobis  ``√((x−y)ᵀ A (x−y))``   (the simplest learned geometry)
  * learned grid metric           straight-line bound and kNN-graph geodesics

The figure shows seed 0: two *random* held-out queries (top) and two
*deliberately selected confused* queries (bottom), labelled as such, with
Euclidean vs learned-geodesic top-10, geodesic paths, iso-distance contours and
the learned metric field.
"""

import torch

from _common import plt, rn, save, timer
from riemann_and_sons import learning, plot, synthetic

dom = rn.Plane()
K_EVAL = (10, 50)
REG = {"euclidean": 0.01, "smoothness": 1e-4, "volume": 0.01}
SEEDS = [0, 1, 2]


def train_grid_metric(train_pts, train_labels, seed):
    torch.manual_seed(seed)
    geo = rn.Geometry(dom, rn.GridMetric(dom, 16))
    sampler = synthetic.triplet_sampler(train_pts, train_labels, batch=128, seed=seed)
    learning.fit_metric(geo, sampler, epochs=300, lr=0.03, margin=0.4, regularizers=REG, reg_points=256)
    learning.fit_metric(geo, sampler, epochs=40, lr=0.01, margin=0.4, distance="geodesic", distance_kw={"n_points": 8, "iterations": 20}, regularizers=REG, reg_points=256)
    return geo


def train_mahalanobis(train_pts, train_labels, seed):
    torch.manual_seed(seed)
    geo = rn.Geometry(dom, rn.ConstantMetric(torch.eye(2), learnable=True))
    sampler = synthetic.triplet_sampler(train_pts, train_labels, batch=128, seed=seed)
    learning.fit_metric(geo, sampler, epochs=300, lr=0.03, margin=0.4, regularizers={"volume": 0.01}, reg_points=16)
    return geo


def precision(retriever, queries, q_labels, ref_labels, k):
    out = retriever.query(queries, k=k)
    return float((ref_labels[out.indices] == q_labels[:, None]).double().mean())


scores = {}
kept = None
for seed in SEEDS:
    pts, labels = synthetic.two_moons(300, noise=0.22, seed=seed)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(100 + seed))
    tr, te = perm[:200], perm[200:]
    P, L, Q, QL = pts[tr], labels[tr], pts[te], labels[te]
    with timer(f"seed {seed}: training grid metric and Mahalanobis baseline"):
        geo = train_grid_metric(P, L, seed)
        maha = train_mahalanobis(P, L, seed)
    retrievers = {
        "Euclidean": rn.EuclideanRetriever(P, L),
        "learned Mahalanobis": rn.MahalanobisRetriever(P, maha.metric.matrix.detach(), L),
        "learned grid metric, straight-line": rn.GeodesicRetriever(P, geo, L, method="straight"),
        "learned grid metric, kNN-graph geodesic": rn.GeodesicRetriever(P, geo, L, method="graph", k_graph=10),
    }
    for name, r in retrievers.items():
        for k in K_EVAL:
            scores.setdefault((name, k), []).append(precision(r, Q, QL, L, k))
    if seed == SEEDS[0]:
        kept = (P, L, Q, QL, geo)

print("\nheld-out precision, mean ± std over seeds", SEEDS)
table = {}
for (name, k), v in scores.items():
    v = torch.tensor(v)
    table[(name, k)] = (v.mean().item(), v.std().item())
for name in dict.fromkeys(n for n, _ in scores):
    print(f"  {name:42s} " + "   ".join(f"P@{k} {table[(name, k)][0]:.3f} ± {table[(name, k)][1]:.3f}" for k in K_EVAL))

# ---------------------------------------------------------------- figure (seed 0)
P, L, Q, QL, geo = kept
eu = rn.EuclideanRetriever(P, L)
ge_grid = rn.GeodesicRetriever(P, geo, L, method="grid", resolution=96)
K = 10
out_e = eu.query(Q, k=K)
wrong_e = (L[out_e.indices] != QL[:, None]).sum(1)
gen = torch.Generator().manual_seed(7)
confused_idx = torch.topk(wrong_e, 2).indices
random_idx = [int(i) for i in torch.randperm(len(Q), generator=gen) if int(i) not in confused_idx.tolist()][:2]
rows = [(Q[i], QL[i], "random") for i in random_idx] + [(Q[i], QL[i], "selected confused") for i in confused_idx]

fig, axes = plt.subplots(len(rows), 3, figsize=(16.5, 5.2 * len(rows)))
for r, (q, ql, kind) in enumerate(rows):
    res_e = eu.query(q[None], k=K)
    res_g = ge_grid.query(q[None], k=K, with_paths=True)
    df = ge_grid._field(q)
    for c, (res, title) in enumerate([(res_e, "Euclidean top-10"), (res_g, "learned-geodesic top-10")]):
        ax = axes[r, c]
        plot.points(P, ax=ax, labels=L, s=12, cmap="coolwarm", alpha=0.6)
        hits = P[res.indices[0]]
        wrong = L[res.indices[0]] != ql
        ax.scatter(hits[:, 0], hits[:, 1], facecolors="none", edgecolors="k", s=90, linewidths=1.5, zorder=5)
        ax.scatter(hits[wrong, 0], hits[wrong, 1], marker="x", c="k", s=70, zorder=6)
        ax.scatter(q[0], q[1], marker="*", s=260, c="gold", edgecolors="k", zorder=7)
        if res.paths is not None:
            plot.geodesics(res.paths[0], ax=ax, color="k", linewidth=0.8, endpoints=False)
        ax.set_title(f"[{kind} held-out query]  {title}: {int(wrong.sum())} wrong", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax = axes[r, 2]
    plot.distance_field(df, ax=ax, contours=20, title="learned geodesic distance from ★")
    plot.metric_field(geo, ax=ax, resolution=(20, 20), color="white", alpha=0.7, normalize="local")
names = list(dict.fromkeys(n for n, _ in scores))
fig.suptitle("held-out P@10 / P@50, mean over 3 seeds\n" + "     ".join(f"{n}: {table[(n, 10)][0]:.2f} / {table[(n, 50)][0]:.2f}" for n in names), fontsize=11)
save(fig, "demo_f_retrieval_lab.png", rect=[0, 0, 1, 0.965])
