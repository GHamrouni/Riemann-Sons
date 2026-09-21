# Roadmap

The progression is **2-D synthetic → image-derived → high-dimensional**. No high-dimensional feature is
added before a useful 2-D analogue exists to make its behaviour inspectable.

## Done in v0.1

- Differentiable metrics, Christoffel symbols, curvature; analytical test-suite.
- Laplace–Beltrami and Riemannian diffusion; geodesics (IVP, BVP, distance fields); parallel transport.
- Metric flows: Ricci, task-driven (parameter-space), hybrid.
- Deformations with first-class Jacobians and fold detection.
- Metric learning with regularisers; the 2-D retrieval laboratory; the inspector.

## Next

1. **Bridge through images.** MNIST / Fashion-MNIST: `I → E(I) ∈ R^d` with a frozen encoder; compare
   pixel-space and embedding-space geometry. Every embedding is an image, so neighbourhood changes in
   high dimension stay visually inspectable.
2. **Learned geometry for frozen embeddings**, in strictly increasing complexity, each level having to
   show measurable value over the previous one:
   cosine → Euclidean → global Mahalanobis (`ConstantMetric`) → local diagonal → low-rank local
   (`DiagonalLowRankParameterization`) → pullback `J_Fᵀ J_F + εI` (`PullbackMetric`) → general learned
   Riemannian metric.
3. **Pullback metrics** as the main high-dimensional direction: they connect neural networks, Jacobians,
   differentiable geometry and metric learning, and are cheap to evaluate along straight segments.
   Document when `J_Fᵀ J_F` is a true metric (full column rank) versus merely PSD.
4. **Scalable geodesics in high dimension**: straight-line bounds, kNN-graph geodesics on the data
   manifold (already in `GeodesicRetriever(method="graph")`), and learned shortcut networks.
5. **RAG experiment.** Frozen embedding model, fixed document embeddings, learned `g_θ` on top;
   baseline cosine / Euclidean retrieval vs geodesic retrieval on a benchmark with relevance labels.
6. **Curvature-based regularisation** of learned metrics, and gauge-aware Ricci flow (DeTurck).

## Non-goals

- Symbolic differential geometry.
- Making cage / deformation the central abstraction.
- Claiming any regulariser or flow is optimal; everything here is an experimental instrument.
