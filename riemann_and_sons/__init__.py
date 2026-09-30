"""Riemann & Sons -- fine deformations of spaces.

A differentiable Riemannian-geometry research toolkit built on PyTorch.

    Representation  |  Geometry  |  Operation

are kept explicitly separate: a field, image or point cloud (representation)
lives on a :class:`Geometry` ``(M, g)`` and operations (geodesics, diffusion,
retrieval, ...) take the geometry as an argument.  Changing the geometry never
requires changing the object living on it.

Use ``torch.set_default_dtype(torch.float64)`` before constructing inputs for
curvature and other precision-sensitive experiments. Importing this package
does not change PyTorch's default dtype or install warning filters.
"""

from . import curvature, deformation, fields, flows, learning, operators, paths, plot, retrieval, synthetic
from .deformation import (
    Affine,
    Displacement,
    JacobianStats,
    ThinPlateSpline,
    Transformation,
    jacobian_stats,
    warp,
)
from .domains import Box, Plane
from .fields import Field, Image, ScalarField
from .flows import GradientFlow, HybridFlow, MetricFlow, MetricTrajectory, RicciFlow, diffuse
from .geometry import Geometry
from .inspector import Inspector
from .metrics import (
    CholeskyParameterization,
    ConformalMetric,
    ConformalParameterization,
    ConstantMetric,
    DiagonalLowRankMetric,
    DiagonalLowRankParameterization,
    DiagonalParameterization,
    EuclideanMetric,
    FunctionMetric,
    GridMetric,
    LogEuclideanParameterization,
    Metric,
    PullbackMetric,
    image_induced_metric,
)
from .operators import LaplaceBeltrami, divergence, gradient, laplace_beltrami, riemannian_gradient
from .paths import DistanceField, Path, distance_field, exp_map, geodesic, geodesic_shoot, parallel_transport, path_length
from .retrieval import CosineRetriever, EuclideanRetriever, GeodesicRetriever, MahalanobisRetriever

__version__ = "0.1.0"
__all__ = [name for name in dir() if not name.startswith("_")]
