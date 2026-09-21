"""Riemann & Sons -- fine deformations of spaces.

A differentiable Riemannian-geometry research toolkit built on PyTorch.

    Representation  |  Geometry  |  Operation

are kept explicitly separate: a field, image or point cloud (representation)
lives on a :class:`Geometry` ``(M, g)`` and operations (geodesics, diffusion,
retrieval, ...) take the geometry as an argument.  Changing the geometry never
requires changing the object living on it.

Importing the package sets PyTorch's default dtype to ``float64``: this is a
toolkit for mathematical experiments where curvature is obtained by nested
automatic differentiation, and single precision is not adequate for that.
Call ``torch.set_default_dtype(torch.float32)`` afterwards if you disagree.
"""

import warnings as _warnings

import torch as _torch

_torch.set_default_dtype(_torch.float64)
# PyTorch emits this from inside torch.func on first use; it is not actionable for users of this library.
_warnings.filterwarnings("ignore", message=".*torch.jit.script.*deprecated.*", category=FutureWarning)

from . import curvature, deformation, fields, flows, learning, operators, paths, plot, retrieval, synthetic  # noqa: E402
from .deformation import (  # noqa: E402
    Affine,
    Displacement,
    JacobianStats,
    ThinPlateSpline,
    Transformation,
    jacobian_stats,
    warp,
)
from .domains import Box, Plane  # noqa: E402
from .fields import Field, Image, ScalarField  # noqa: E402
from .flows import GradientFlow, HybridFlow, MetricFlow, MetricTrajectory, RicciFlow, diffuse  # noqa: E402
from .geometry import Geometry  # noqa: E402
from .inspector import Inspector  # noqa: E402
from .metrics import (  # noqa: E402
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
from .operators import LaplaceBeltrami, divergence, gradient, laplace_beltrami, riemannian_gradient  # noqa: E402
from .paths import DistanceField, Path, distance_field, exp_map, geodesic, geodesic_shoot, parallel_transport, path_length  # noqa: E402
from .retrieval import CosineRetriever, EuclideanRetriever, GeodesicRetriever, MahalanobisRetriever  # noqa: E402

__version__ = "0.1.0"
__all__ = [name for name in dir() if not name.startswith("_")]
