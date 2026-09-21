import torch
from torch.func import jacfwd, vmap

import riemann_and_sons as rn
from riemann_and_sons._interp import bspline_prefilter, interpolate


def test_box_grid_and_spacing():
    box = rn.Box([0.0, -1.0], [2.0, 1.0])
    pts = box.grid((5, 3))
    assert pts.shape == (5, 3, 2)
    assert torch.allclose(pts[0, 0], torch.tensor([0.0, -1.0]))
    assert torch.allclose(pts[-1, -1], torch.tensor([2.0, 1.0]))
    assert torch.allclose(box.spacing((5, 3)), torch.tensor([0.5, 1.0]))
    assert box.contains(torch.tensor([1.0, 0.0]))
    assert not box.contains(torch.tensor([3.0, 0.0]))
    assert torch.allclose(box.clamp(torch.tensor([3.0, -5.0])), torch.tensor([2.0, -1.0]))


def test_periodic_box_wraps():
    box = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    assert torch.allclose(box.clamp(torch.tensor([1.25, -0.25])), torch.tensor([0.25, 0.75]))
    assert box.grid(4).shape == (4, 4, 2)
    assert torch.allclose(box.spacing(4), torch.tensor([0.25, 0.25]))


def test_linear_interpolation_reproduces_linear_functions():
    box = rn.Box([0.0, 0.0], [1.0, 1.0])
    pts = box.grid((9, 7))
    vals = (2 * pts[..., :1] - 3 * pts[..., 1:]).clone()
    q = torch.rand(40, 2)
    out = interpolate(vals, q, box.lo, box.hi, "linear")
    assert torch.allclose(out, 2 * q[:, :1] - 3 * q[:, 1:], atol=1e-12)


def test_cubic_interpolation_linear_precision_in_interior():
    box = rn.Box([0.0, 0.0], [1.0, 1.0])
    pts = box.grid((17, 17))
    vals = (2 * pts[..., :1] - 3 * pts[..., 1:]).clone()
    q = 0.2 + 0.6 * torch.rand(40, 2)  # away from the clamped boundary
    out = interpolate(vals, q, box.lo, box.hi, "cubic")
    assert torch.allclose(out, 2 * q[:, :1] - 3 * q[:, 1:], atol=1e-12)


def test_prefilter_makes_spline_interpolate_nodes():
    box = rn.Box([0.0, 0.0], [1.0, 1.0])
    pts = box.grid((12, 9))
    f = torch.sin(6 * pts[..., :1]) * torch.cos(4 * pts[..., 1:])
    coef = bspline_prefilter(f)
    assert torch.allclose(interpolate(coef, pts, box.lo, box.hi, "cubic"), f, atol=1e-10)


def test_periodic_prefilter_and_wrap():
    box = rn.Box([0.0, 0.0], [1.0, 1.0], periodic=True)
    pts = box.grid(16)
    f = torch.sin(2 * torch.pi * pts[..., :1]) * torch.cos(2 * torch.pi * pts[..., 1:])
    coef = bspline_prefilter(f, periodic=True)
    assert torch.allclose(interpolate(coef, pts, box.lo, box.hi, "cubic", periodic=True), f, atol=1e-10)
    q = torch.tensor([[0.3, 0.4]])
    shifted = q + torch.tensor([[1.0, -2.0]])
    assert torch.allclose(
        interpolate(coef, q, box.lo, box.hi, "cubic", periodic=True),
        interpolate(coef, shifted, box.lo, box.hi, "cubic", periodic=True),
    )


def test_interpolation_is_vmap_and_jacfwd_compatible():
    box = rn.Box([0.0, 0.0], [1.0, 1.0])
    pts = box.grid((9, 9))
    vals = (2 * pts[..., :1] - 3 * pts[..., 1:]).clone()
    q = 0.2 + 0.6 * torch.rand(10, 2)
    J = vmap(jacfwd(lambda p: interpolate(vals, p, box.lo, box.hi)))(q)
    assert torch.allclose(J, torch.tensor([[[2.0, -3.0]]]).expand(10, 1, 2), atol=1e-10)
