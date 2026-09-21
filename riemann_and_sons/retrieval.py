"""Retrieval under a programmable geometry.

The representation (a fixed set of points) never changes; the retriever only
changes the *geometry* used to interpret their relationships:

* :class:`EuclideanRetriever`   -- ``‖x − y‖``
* :class:`CosineRetriever`      -- ``1 − cos(x, y)``
* :class:`MahalanobisRetriever` -- ``√((x−y)^T A (x−y))`` (a constant metric)
* :class:`GeodesicRetriever`    -- ``d_g(x, y)`` for an arbitrary :class:`Geometry`,
  by a grid distance field (2-D, exact paths), a kNN graph with Riemannian edge
  lengths (any dimension), or the straight-line upper bound (fully batched).
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
        self.labels = None if labels is None else torch.as_tensor(labels)

    def distances(self, queries: Tensor) -> Tensor:  # (Q, N)
        raise NotImplementedError

    def query(self, queries: Tensor, k: int = 5, with_paths: bool = False) -> RetrievalResult:
        queries = torch.as_tensor(queries)
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
        self.A = torch.as_tensor(A, dtype=self.points.dtype)

    def distances(self, queries: Tensor) -> Tensor:
        diff = queries[:, None, :] - self.points[None, :, :]
        return torch.sqrt(torch.einsum("qni,ij,qnj->qn", diff, self.A, diff).clamp_min(0))


class GeodesicRetriever(Retriever):
    """Nearest neighbours under ``d_g``.

    ``method``:
      * ``"grid"``     -- Dijkstra distance field from each query on a grid over the
        domain (2-D laboratory; returns discrete geodesic paths).
      * ``"graph"``    -- kNN graph on the points with Riemannian edge lengths; the
        query is linked to its ``k_graph`` Euclidean neighbours (any dimension).
      * ``"straight"`` -- straight-line upper bound (batched, differentiable).
    """

    name = "geodesic"

    def __init__(self, points: Tensor, geometry, labels: Tensor | None = None, method: str = "grid", resolution: int = 96, k_graph: int = 10, samples: int = 8):
        super().__init__(points, labels)
        self.geometry = geometry
        self.method = method
        self.resolution = resolution
        self.k_graph = k_graph
        self.samples = samples
        self._fields: dict = {}
        if method == "graph":
            self._build_graph()

    # -- straight ------------------------------------------------------------
    def _straight(self, queries: Tensor) -> Tensor:
        Q, N = queries.shape[0], self.points.shape[0]
        q = queries[:, None, :].expand(Q, N, -1)
        p = self.points[None, :, :].expand(Q, N, -1)
        return straight_line_distance(self.geometry.metric, q, p, self.samples)

    # -- grid -----------------------------------------------------------------
    def _field(self, q: Tensor):
        key = tuple(q.tolist())
        if key not in self._fields:
            self._fields[key] = distance_field(self.geometry, q[None], self.resolution)
        return self._fields[key]

    def _grid(self, queries: Tensor) -> Tensor:
        return torch.stack([self._field(q).sample(self.points) for q in queries])

    # -- graph ----------------------------------------------------------------
    def _build_graph(self) -> None:
        from scipy.sparse import coo_matrix

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph + 1, N)
        D = torch.cdist(P, P)
        nn_idx = torch.topk(D, k, dim=-1, largest=False).indices[:, 1:]
        rows = torch.arange(N)[:, None].expand_as(nn_idx).reshape(-1)
        cols = nn_idx.reshape(-1)
        w = self._edge_length(P[rows], P[cols])
        rows_np, cols_np, w_np = rows.numpy(), cols.numpy(), w.detach().cpu().numpy()
        self._graph = coo_matrix((np.concatenate([w_np, w_np]), (np.concatenate([rows_np, cols_np]), np.concatenate([cols_np, rows_np]))), shape=(N, N)).tocsr()

    def _edge_length(self, a: Tensor, b: Tensor) -> Tensor:
        d = b - a
        return 0.5 * (self.geometry.norm(a, d) + self.geometry.norm(b, d))

    def _graph_query(self, q: Tensor):
        from scipy.sparse import coo_matrix, vstack, hstack, csr_matrix
        from scipy.sparse.csgraph import dijkstra

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph, N)
        nn = torch.topk((P - q).norm(dim=-1), k, largest=False).indices
        w = self._edge_length(q[None].expand(k, -1), P[nn]).detach().cpu().numpy()
        link = coo_matrix((w, (np.zeros(k, dtype=int), nn.numpy())), shape=(1, N)).tocsr()
        G = vstack([hstack([self._graph, link.T]), hstack([link, csr_matrix((1, 1))])]).tocsr()
        dist, pred = dijkstra(G, directed=False, indices=N, return_predecessors=True)
        return torch.as_tensor(dist[:N], dtype=P.dtype), pred

    def distances(self, queries: Tensor) -> Tensor:  # (Q, N)
        raise NotImplementedError

    def query(self, queries: Tensor, k: int = 5, with_paths: bool = False) -> RetrievalResult:
        queries = torch.as_tensor(queries)
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
        self.A = torch.as_tensor(A, dtype=self.points.dtype)

    def distances(self, queries: Tensor) -> Tensor:
        diff = queries[:, None, :] - self.points[None, :, :]
        return torch.sqrt(torch.einsum("qni,ij,qnj->qn", diff, self.A, diff).clamp_min(0))


class GeodesicRetriever(Retriever):
    """Nearest neighbours under ``d_g``.

    ``method``:
      * ``"grid"``     -- Dijkstra distance field from each query on a grid over the
        domain (2-D laboratory; returns discrete geodesic paths).
      * ``"graph"``    -- kNN graph on the points with Riemannian edge lengths; the
        query is linked to its ``k_graph`` Euclidean neighbours (any dimension).
      * ``"straight"`` -- straight-line upper bound (batched, differentiable).
    """

    name = "geodesic"

    def __init__(self, points: Tensor, geometry, labels: Tensor | None = None, method: str = "grid", resolution: int = 96, k_graph: int = 10, samples: int = 8):
        super().__init__(points, labels)
        self.geometry = geometry
        self.method = method
        self.resolution = resolution
        self.k_graph = k_graph
        self.samples = samples
        self._fields: dict = {}
        if method == "graph":
            self._build_graph()

    # -- straight ------------------------------------------------------------
    def _straight(self, queries: Tensor) -> Tensor:
        Q, N = queries.shape[0], self.points.shape[0]
        q = queries[:, None, :].expand(Q, N, -1)
        p = self.points[None, :, :].expand(Q, N, -1)
        return straight_line_distance(self.geometry.metric, q, p, self.samples)

    # -- grid -----------------------------------------------------------------
    def _field(self, q: Tensor):
        key = tuple(q.tolist())
        if key not in self._fields:
            self._fields[key] = distance_field(self.geometry, q[None], self.resolution)
        return self._fields[key]

    def _grid(self, queries: Tensor) -> Tensor:
        return torch.stack([self._field(q).sample(self.points) for q in queries])

    # -- graph ----------------------------------------------------------------
    def _build_graph(self) -> None:
        from scipy.sparse import coo_matrix

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph + 1, N)
        D = torch.cdist(P, P)
        nn_idx = torch.topk(D, k, dim=-1, largest=False).indices[:, 1:]
        rows = torch.arange(N)[:, None].expand_as(nn_idx).reshape(-1)
        cols = nn_idx.reshape(-1)
        w = self._edge_length(P[rows], P[cols])
        rows_np, cols_np, w_np = rows.numpy(), cols.numpy(), w.detach().cpu().numpy()
        self._graph = coo_matrix((np.concatenate([w_np, w_np]), (np.concatenate([rows_np, cols_np]), np.concatenate([cols_np, rows_np]))), shape=(N, N)).tocsr()

    def _edge_length(self, a: Tensor, b: Tensor) -> Tensor:
        d = b - a
        return 0.5 * (self.geometry.norm(a, d) + self.geometry.norm(b, d))

    def _graph_query(self, q: Tensor):
        from scipy.sparse import coo_matrix, vstack, hstack, csr_matrix
        from scipy.sparse.csgraph import dijkstra

        P = self.points
        N = P.shape[0]
        k = min(self.k_graph, N)
        nn = torch.topk((P - q).norm(dim=-1), k, largest=False).indices
        w = self._edge_length(q[None].expand(k, -1), P[nn]).detach().cpu().numpy()
        link = coo_matrix((w, (np.zeros(k, dtype=int), nn.numpy())), shape=(1, N)).tocsr()
        G = vstack([hstack([self._graph, link.T]), hstack([link, csr_matrix((1, 1))])]).tocsr()
        dist, pred = dijkstra(G, directed=False, indices=N, return_predecessors=True)
        return torch.as_tensor(dist[:N], dtype=P.dtype), pred

    def _graph(self_, queries: Tensor) -> Tensor:  # noqa: N805 - shadowed name on purpose
        raise RuntimeError

    def distances(self, queries: Tensor) -> Tensor:
        if self.method == "straight":
            return self._straight(queries)
        if self.method == "grid":
            return self._grid(queries)
        if self.method == "graph":
            out = []
            self._last_preds = []
            for q in queries:
                d, pred = self._graph_query(q)
                out.append(d)
                self._last_preds.append(pred)
            return torch.stack(out)
        raise ValueError(f"unknown method {self.method!r}")

    def _paths(self, queries: Tensor, idx: Tensor) -> list[list[Path]]:
        if self.method == "grid":
            return [[self._field(q).path_to(self.points[j]) for j in row] for q, row in zip(queries, idx)]
        if self.method == "graph":
            paths = []
            N = self.points.shape[0]
            for qi, (q, row) in enumerate(zip(queries, idx)):
                pred = self._last_preds[qi]
                per = []
                for j in row:
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
