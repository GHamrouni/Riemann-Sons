"""Fields: scalar / vector / image data living on a gridded :class:`Box`.

    *The image is not the object being modeled geometrically.  The image is a
    field living on a space whose geometry can be changed.*

A :class:`Field` stores ``values`` of shape ``(*channels, *resolution)`` --
the trailing ``domain.dim`` axes are spatial and axis ``k`` corresponds to
coordinate ``x_k`` (``ij`` indexing).  :class:`Image` adds converters from
and to the usual row-major ``(H, W[, C])`` array layout.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as Fnn

from ._interp import interpolate
from .domains import Box

__all__ = ["Field", "ScalarField", "Image", "gaussian_blur"]


def _quadrature_weights(domain: Box, resolution) -> Tensor:
    """Tensor-product trapezoidal weights; periodic grids use uniform weights."""
    weights = torch.ones(resolution, dtype=domain.lo.dtype, device=domain.lo.device)
    if not domain.periodic:
        for axis, size in enumerate(resolution):
            if size > 1:
                weights.select(axis, 0).mul_(0.5)
                weights.select(axis, size - 1).mul_(0.5)
    return weights * domain.spacing(resolution).prod()


class Field:
    def __init__(self, values: Tensor, domain: Box):
        values = torch.as_tensor(values)
        if values.ndim < domain.dim:
            raise ValueError("values must have at least `domain.dim` spatial axes")
        domain._shape(values.shape[-domain.dim :])
        self.values = values
        self.domain = domain

    # ----------------------------------------------------------- structure
    @property
    def dim(self) -> int:
        return self.domain.dim

    @property
    def resolution(self) -> tuple[int, ...]:
        return tuple(self.values.shape[-self.dim :])

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(self.values.shape[: -self.dim])

    @property
    def spacing(self) -> Tensor:
        return self.domain.spacing(self.resolution)

    @property
    def points(self) -> Tensor:
        return self.domain.grid(self.resolution)

    @property
    def is_scalar(self) -> bool:
        return self.values.ndim == self.dim

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def like(self, values: Tensor) -> "Field":
        return type(self)(values, self.domain)

    def clone(self) -> "Field":
        return self.like(self.values.clone())

    def detach(self) -> "Field":
        return self.like(self.values.detach())

    def to(self, *args, **kw) -> "Field":
        values = self.values.to(*args, **kw)
        coordinate_dtype = values.dtype if values.is_floating_point() else self.domain.lo.dtype
        return type(self)(values, self.domain.to(device=values.device, dtype=coordinate_dtype))

    # ----------------------------------------------------------- algebra
    def map(self, fn: Callable[[Tensor], Tensor]) -> "Field":
        return self.like(fn(self.values))

    def _binary(self, other, op):
        other_values = other.values if isinstance(other, Field) else other
        return self.like(op(self.values, other_values))

    def __add__(self, other):
        return self._binary(other, torch.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, torch.sub)

    def __mul__(self, other):
        return self._binary(other, torch.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, torch.div)

    def __neg__(self):
        return self.like(-self.values)

    # ----------------------------------------------------------- sampling
    def sample(self, x: Tensor, mode: str = "linear") -> Tensor:
        """Interpolate the field at points ``x`` ``(..., n)`` -> ``(..., *channels)`` (scalar: ``(...)``)."""
        feat = self.values.reshape(-1, *self.resolution).movedim(0, -1)  # (*res, C)
        out = interpolate(feat, x, self.domain.lo, self.domain.hi, mode, self.domain.periodic)
        out = out.reshape((*x.shape[:-1], *self.channels))
        return out

    # ----------------------------------------------------------- integrals
    def integral(self) -> Tensor:
        """Euclidean integral ``∫ f dx`` with trapezoidal weights (uniform on periodic grids)."""
        return (self.values * _quadrature_weights(self.domain, self.resolution)).sum(dim=tuple(range(-self.dim, 0)))

    def riemannian_integral(self, geometry) -> Tensor:
        """``∫ f √|g| dx`` (per channel)."""
        w = geometry.volume_element(self.points)
        return (self.values * w * _quadrature_weights(self.domain, self.resolution)).sum(dim=tuple(range(-self.dim, 0)))

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_function(cls, domain: Box, resolution, fn: Callable[[Tensor], Tensor]) -> "Field":
        pts = domain.grid(resolution)
        vals = fn(pts)
        if vals.ndim > domain.dim:  # (*res, *channels) -> (*channels, *res)
            vals = vals.permute(*range(domain.dim, vals.ndim), *range(domain.dim))
        return cls(vals, domain)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(channels={self.channels}, resolution={self.resolution}, domain={self.domain!r})"


ScalarField = Field


class Image(Field):
    """A multi-channel field ``I : (M, g) -> R^C`` with ``values`` of shape ``(C, N_x, N_y)``.

    Conversion from/to the row-major image convention (``rows = y`` growing
    downwards) is handled by :meth:`from_array` / :meth:`to_array`.  Internally
    axis 0 is ``x`` and axis 1 is ``y`` growing *upwards*, so that the field
    shares the coordinate conventions of every other object in the library.
    """

    def __init__(self, values: Tensor, domain: Box):
        values = torch.as_tensor(values)
        if values.ndim == domain.dim:
            values = values.unsqueeze(0)
        super().__init__(values, domain)

    @property
    def n_channels(self) -> int:
        return self.values.shape[0]

    @classmethod
    def from_array(cls, array, domain: Box | None = None, flip_y: bool = True) -> "Image":
        """From ``(H, W)`` or ``(H, W, C)`` array (numpy or tensor). Pixel values are cast to the default dtype."""
        arr = torch.as_tensor(array).to(torch.get_default_dtype())
        if arr.ndim == 2:
            arr = arr.unsqueeze(-1)
        if arr.ndim != 3:
            raise ValueError("expected (H, W) or (H, W, C)")
        if flip_y:
            arr = arr.flip(0)
        vals = arr.permute(2, 1, 0).contiguous()  # (C, W, H): axis0 = x (columns), axis1 = y (rows, upwards)
        if domain is None:
            H, W = arr.shape[:2]
            aspect = H / W
            domain = Box([0.0, 0.0], [1.0, aspect])
        return cls(vals, domain)

    @classmethod
    def from_file(cls, path: str, domain: Box | None = None, grayscale: bool = False) -> "Image":
        import matplotlib.image as mpimg

        arr = torch.as_tensor(mpimg.imread(path)).to(torch.get_default_dtype())
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = cls.from_array(arr, domain)
        return img.to_gray() if grayscale else img

    def to_array(self, flip_y: bool = True) -> Tensor:
        """Back to ``(H, W, C)`` (squeezed to ``(H, W)`` for single-channel images)."""
        arr = self.values.permute(2, 1, 0)  # (H, W, C)
        if flip_y:
            arr = arr.flip(0)
        return arr.squeeze(-1) if arr.shape[-1] == 1 else arr

    def to_gray(self) -> "Image":
        if self.n_channels == 1:
            return self
        w = torch.tensor([0.299, 0.587, 0.114], dtype=self.dtype, device=self.device)[: self.n_channels]
        w = w / w.sum()
        return Image((self.values * w[:, None, None]).sum(0, keepdim=True), self.domain)

    def scalar(self) -> Field:
        """The single-channel image as a plain scalar field (values ``(N_x, N_y)``)."""
        return Field(self.to_gray().values[0], self.domain)


# ---------------------------------------------------------------------------
def gaussian_blur(field: Field, sigma: float, truncate: float = 3.0) -> Field:
    """Separable Gaussian blur along the spatial axes; ``sigma`` in grid cells.

    Boundary handling follows the domain: reflect padding for boxes, circular
    padding for periodic domains.
    """
    if sigma <= 0:
        return field
    if not math.isfinite(sigma) or not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("sigma must be finite and truncate must be positive and finite")
    radius = max(1, int(math.ceil(truncate * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=field.dtype, device=field.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    vals = field.values
    n = field.dim
    lead = vals.shape[:-n]
    out = vals.reshape(-1, *field.resolution)  # (C, *res)
    for axis in range(n):
        moved = out.movedim(axis + 1, -1)  # (C, ..., N_axis)
        shape = moved.shape
        flat = moved.reshape(-1, 1, shape[-1])
        size = shape[-1]
        index = torch.arange(-radius, size + radius, device=field.device)
        if field.domain.periodic:
            index = index.remainder(size)
        elif size == 1:
            index = torch.zeros_like(index)
        else:
            index = index.remainder(2 * (size - 1))
            index = torch.minimum(index, 2 * (size - 1) - index)
        padded = flat.index_select(-1, index)
        conv = Fnn.conv1d(padded, kernel.reshape(1, 1, -1))
        out = conv.reshape(shape).movedim(-1, axis + 1)
    return field.like(out.reshape(*lead, *field.resolution))
