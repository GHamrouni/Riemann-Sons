"""Retrieval under a programmable geometry.

The representation (a fixed set of points) never changes; the retriever only
changes the *geometry* used to interpret their relationships:

* :class:`EuclideanRetriever`   -- ``‖x − y‖``
* :class:`CosineRetriever`      -- ``1 − cos(x, y)``
* :class:`MahalanobisRetriever` -- ``√((x−y)ᵀ A (x−y))`` (a constant metric)
* :class:`GeodesicRetriever`    -- ``d_g(x, y)`` for an arbitrary :class:`Geometry`,
  approximated by a grid distance field, a kNN graph with Riemannian edge
  lengths (any dimension), or numerical straight-segment lengths (fully batched).

Snapshot semantics.  A :class:`GeodesicRetriever` evaluates the metric when it
builds its graph (``method="graph"``) or a distance field (``method="grid"``,
cached per query point) and keeps those numbers.  If the metric's parameters
change afterwards call :meth:`GeodesicRetriever.refresh`, otherwise cached
weights and fresh query links would mix two geometries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .learning import straight_line_distance
from .paths import Path, distance_field

__all__ = [
    "RetrievalResult",
    "Retriever",
    "EuclideanRetriever",
    "CosineRetriever",
    "MahalanobisRetriever",
    "GeodesicRetriever",
    "precision_at_k",
]


@dataclass
class RetrievalResult:
    indices: Tensor  # (Q, k)
    distances: Tensor  # (Q, k)
    paths: list[list[Path]] | None = None  # per query, per hit

    def __len__(self) -> int:
        return self.indices.shape[0]


class Retriever:
    name = "retriever"

    def __init__(self, points: Tensor, labels: Tensor | None = None):
        self.points = torch.as_tensor(points)
        if self.points.ndim != 2 or self.points.shape[0] == 0:
            raise ValueError("points must have shape (N, n) with N >= 1")
        if not self.points.is_floating_point():
            self.points = self.points.to(torch.get_default_dtype())
        self.labels = None if labels is None else torch.as_tensor(labels)

    def distances(self, queries: Tensor) -> Tensor:  # (Q, N)
        raise NotImplementedError

    def query(self, queries: Tensor, k: int = 5, with_paths: bool = False) -> RetrievalResult:
        if not 1 <= k <= len(self.points):
            raise ValueError(f"k must be between 1 and {len(self.points)}")
        queries = torch.as_tensor(queries, dtype=self.points.dtype, device=self.points.device)
        if queries.ndim == 1:
            queries = queries[None]
        D = self.distances(queries)
        d, idx = torch.topk(D, k, dim=-1, largest=False)
        paths = self._paths(queries, idx) if with_paths else None
        return RetrievalResult(idx, d, paths)

    def _paths(self, queries: Tensor, idx: Tensor) -> list[list[Path]]:
        return [[Path(torch.stack([q, self.points[j]])) for j in row] for q, row in zip(queries, idx)]


class EuclideanRetriever(Retriever):
    name = "euclidean"

    def distances(self, queries: Tensor) -> Tensor:
        return torch.cdist(queries, self.points)


class CosineRetriever(Retriever):
    name = "cosine"

    def distances(self, queries: Tensor) -> Tensor:
        q = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        p = self.points / self.points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return 1 - q @ p.T


class MahalanobisRetriever(Retriever):
    name = "mahalanobis"

    def __init__(self, points: Tensor, A: Tensor, labels: Tensor | None = None):
        super().__init__(points, labels)
        self.A = torch.as_tensor(A, dtype=self.points.dtype, device=self.points.device)

    def distances(self, queries: Tensor) -> Tensor:
        diff = queries[:, None, :] - self.points[None, :, :]
        return torch.sqrt(torch.einsum("qni,ij,qnj->qn", diff, self.A, diff).clamp_min(0))


class GeodesicRetriever(Retriever):
    """Nearest neighbours under ``d_g``.

    ``method``:
      * ``"grid"``     -- Dijkstra distance field from each query on a grid over the
        domain (2-D laboratory; returns discrete geodesic paths).
      * ``"graph"``    -- kNN graph on the points with Riemannian edge lengths
        ``½(‖d‖_{g(p)} + ‖d‖_{g(q)})``; the query is linked to its ``k_graph``
        Euclidean neighbours (any dimension).  Edges are unique and undirected.
      * ``"straight"`` -- numerical straight-segment length (batched, differentiable).

    Graph distances may be infinite when the graph is disconnected. Requesting
    paths for unreachable hits raises ``ValueError``; increase ``k_graph`` to
    connect more points. All returned paths run from the query to the hit.
    """

    name = "geodesic"

    def __init__(self, points: Tensor, geometry, labels: Tensor | None = None, method: str = "grid", resolution: int = 96, k_graph: int = 10, samples: int = 8):
        super().__init__(points, labels)
        if method not in ("grid", "graph", "straight"):
            raise ValueError(f"unknown method {method!r}")
        if k_graph < 1:
            raise ValueError("k_graph must be positive")
        self.geometry = geometry
        self.method = method
        self.resolution = resolution
        self.k_graph = k_graph
        self.samples = samples
        self._fields: dict = {}
        self._graph = None
        self._last_preds: list = []
        if method == "graph":
            self._build_graph()

    def refresh(self) -> None:
        """Drop everything computed from the metric (call after the metric's parameters change)."""
        self._fields.clear()
        self._graph = None
        if self.method == "graph":
            self._build_graph()

    # -- straight -------------------------------------------------------------
    def _straight(self, queries: Tensor) -> Tensor:
        Q, N = queries.shape[0], self.points.shape[0]
        q = queries[:, None, :].expand(Q, N, -1)
        p = self.points[None, :, :].expand(Q, N, -1)
        return straight_line_distance(self.geometry.metric, q, p, self.samples)

    # -- grid ------------------------------------------------------------------
    def _field(self, q: Tensor):
        key = tuple(q.tolist())
        if key not in self._fields:
            self._fields[key] = distance_field(self.geometry, q[None], self.resolution)
        return self._fields[key]

    def _grid(self, queries: Tensor) -> Tensor:
        return torch.stack([self._field(q).sample(self.points) for q in queries])

    # -- graph -----------------------------------------------------------------
    def _edge_length(self, a: Tensor, b: Tensor) -> Tensor:
        d = b - a
        with torch.no_grad():
            return 0.5 * (self.geometry.norm(a, d) + self.geometry.norm(b, d))

    def _build_graph(self) -> None:
        from scipy.sparse import coo_matrix

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph, N - 1)
        distances = torch.cdist(P, P)
        distances.fill_diagonal_(float("inf"))
        nn_idx = torch.topk(distances, k, dim=-1, largest=False).indices
        rows = torch.arange(N, device=P.device).repeat_interleave(k)
        cols = nn_idx.reshape(-1)
        pairs = torch.unique(torch.stack([torch.minimum(rows, cols), torch.maximum(rows, cols)], dim=1), dim=0)  # unique undirected edges
        a, b = pairs[:, 0], pairs[:, 1]
        w = self._edge_length(P[a], P[b]).cpu().numpy()
        a_np, b_np = a.cpu().numpy(), b.cpu().numpy()
        self._graph = coo_matrix((np.concatenate([w, w]), (np.concatenate([a_np, b_np]), np.concatenate([b_np, a_np]))), shape=(N, N)).tocsr()

    def _graph_query(self, q: Tensor):
        from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
        from scipy.sparse.csgraph import dijkstra

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph, N)
        nn = torch.topk((P - q).norm(dim=-1), k, largest=False).indices
        w = self._edge_length(q[None].expand(k, -1), P[nn]).cpu().numpy()
        link = coo_matrix((w, (np.zeros(k, dtype=int), nn.cpu().numpy())), shape=(1, N)).tocsr()
        G = vstack([hstack([self._graph, link.T]), hstack([link, csr_matrix((1, 1))])]).tocsr()
        dist, pred = dijkstra(G, directed=False, indices=N, return_predecessors=True)
        return torch.as_tensor(dist[:N], dtype=P.dtype, device=P.device), pred

    # -- dispatch --------------------------------------------------------------
    def distances(self, queries: Tensor) -> Tensor:
        if self.method == "straight":
            return self._straight(queries)
        if self.method == "grid":
            return self._grid(queries)
        out = []
        self._last_preds = []
        for q in queries:
            d, pred = self._graph_query(q)
            out.append(d)
            self._last_preds.append(pred)
        return torch.stack(out)

    def _paths(self, queries: Tensor, idx: Tensor) -> list[list[Path]]:
        if self.method == "grid":
            return [[Path(self._field(q).path_to(self.points[j]).points.flip(0), {"discrete": True}) for j in row] for q, row in zip(queries, idx)]
        if self.method == "graph":
            N = self.points.shape[0]
            paths = []
            for qi, (q, row) in enumerate(zip(queries, idx)):
                pred = self._last_preds[qi]
                per = []
                for j in row:
                    if pred[int(j)] < 0:
                        raise ValueError("cannot build a path to an unreachable point; increase k_graph or request fewer hits")
                    chain = [int(j)]
                    while pred[chain[-1]] >= 0 and pred[chain[-1]] != N:
                        chain.append(int(pred[chain[-1]]))
                    nodes = self.points[torch.as_tensor(chain[::-1])]  # from the query's neighbour to the hit
                    per.append(Path(torch.cat([q[None], nodes])))
                paths.append(per)
            return paths
        return super()._paths(queries, idx)


def precision_at_k(result: RetrievalResult, query_labels: Tensor, labels: Tensor) -> float:
    """Fraction of retrieved items sharing the query's label."""
    hit = labels[result.indices] == torch.as_tensor(query_labels)[:, None]
    return float(hit.double().mean())
