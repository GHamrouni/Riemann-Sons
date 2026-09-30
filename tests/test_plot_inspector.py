import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import riemann_and_sons as rn
from riemann_and_sons import plot, synthetic


def test_ellipse_parameters():
    g = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])
    a, b, ang = plot.ellipse_parameters(g)
    assert torch.allclose(a, torch.tensor([1.0])) and torch.allclose(b, torch.tensor([0.5]))
    assert torch.allclose(torch.cos(ang * torch.pi / 180).abs(), torch.tensor([0.0]), atol=1e-12)  # long axis along y


def test_plot_functions_render():
    dom = rn.Plane()
    geo = rn.Geometry(dom, synthetic.wall_metric())
    fig, axes = plt.subplots(2, 3)
    plot.metric_field(geo, ax=axes[0, 0])
    plot.metric_field(rn.Geometry(dom, synthetic.anisotropic_flow_metric()), ax=axes[0, 1], color_by="anisotropy", normalize="local")
    df = rn.distance_field(geo, torch.tensor([[0.2, 0.2]]), 32)
    plot.distance_field(df, ax=axes[0, 2])
    plot.geodesics([df.path_to(torch.tensor([0.8, 0.2]))], ax=axes[0, 2])
    plot.curvature(rn.Geometry(dom, synthetic.bump_conformal_metric([[0.5, 0.5]], [0.1], [5.0])), ax=axes[1, 0], resolution=16)
    src, dst = synthetic.landmark_pairs("twist")
    tps = rn.ThinPlateSpline(src, dst)
    plot.jacobian_map(tps, dom, ax=axes[1, 1], resolution=24)
    plot.deformation_grid(tps, dom, ax=axes[1, 1])
    plot.image(synthetic.shapes_scene(dom, 32), ax=axes[1, 2])
    plot.metric_scalar(geo, "condition", ax=axes[1, 2], resolution=16)
    plt.close(fig)


def test_inspector_show_and_animate(tmp_path):
    dom = rn.Plane()
    gm = rn.GridMetric.from_function(dom, 12, synthetic.bump_conformal_metric([[0.5, 0.5]], [0.15], [3.0]))
    geo = rn.Geometry(dom, gm)
    insp = rn.Inspector(geo, field=synthetic.shapes_scene(dom, 32), resolution=24, ellipses=(8, 8))
    fig = insp.show(metric=True, det=True, condition=True, curvature=True, distance=torch.tensor([[0.2, 0.2]]), geodesics=[((0.2, 0.2), (0.8, 0.8))], diffusion=0.001, geodesic_kw={"n_points": 16, "iterations": 20})
    assert len(fig.axes) >= 6
    plt.close(fig)
    traj = rn.GradientFlow(lambda m: rn.learning.euclidean_prior(m, dom.grid(8))).integrate(gm, 0.2, 0.1)
    anim = insp.animate(traj, panels=("metric", "det"), path=str(tmp_path / "flow.gif"), fps=2)
    assert (tmp_path / "flow.gif").exists()
    assert len(plt.gcf().axes) == 3  # two panels and one colorbar, independent of frame count
    plt.close("all")
