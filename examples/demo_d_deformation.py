"""Demo D -- smooth transformations from control constraints (the original jmorphism problem).

Given landmark pairs (p_k -> q_k) build a smooth Φ : M → M and warp the image
by inverse sampling I'(x) = I(Φ⁻¹(x)).  Two constructions:

* ThinPlateSpline  -- closed form, exact interpolation, minimal bending energy
* Displacement     -- B-spline displacement field fitted by gradient descent on
                      landmark error + bending energy + fold penalty

Jacobians are first class: det J_Φ is shown as a heatmap and regions with
det J ≤ 0 (folds) are hatched.  The third row deliberately over-constrains the
TPS to produce a fold.
"""

import torch

from _common import constraint_arrows, plt, rn, save, timer
from riemann_and_sons import deformation, plot, synthetic

dom = rn.Plane()
image = synthetic.shapes_scene(dom, 160) * 0.6 + synthetic.checkerboard(dom, 160, 10) * 0.4
src, dst = synthetic.landmark_pairs("pinch")

rows = []
with timer("thin-plate spline"):
    rows.append(("Thin-plate spline (exact interpolation)", rn.ThinPlateSpline(src, dst)))
with timer("fitted B-spline displacement"):
    disp = rn.Displacement(dom, 16)
    hist = deformation.fit_transformation(disp, src, dst, bending=1e-4, fold=5.0, iterations=400, lr=0.02, sample_points=dom.grid(32).reshape(-1, 2))
    print(f"    landmark MSE {hist[0]['landmarks']:.2e} -> {hist[-1]['landmarks']:.2e}, bending {hist[-1]['bending']:.2e}")
    rows.append(("B-spline displacement fitted with bending + fold penalties", disp))
with timer("over-constrained TPS (folds)"):
    bad_dst = dst.clone()
    bad_dst[4] = torch.tensor([0.78, 0.32])  # drag the centre landmark across its neighbour
    rows.append(("Over-constrained TPS: orientation reversal (det J ≤ 0 hatched)", rn.ThinPlateSpline(src, bad_dst)))

fig, axes = plt.subplots(len(rows), 4, figsize=(19, 4.8 * len(rows)))
for r, (name, phi) in enumerate(rows):
    targets = bad_dst if r == 2 else dst
    plot.image(image, ax=axes[r, 0])
    plot.deformation_grid(rn.Affine(torch.eye(2)), dom, ax=axes[r, 0], color="C0", alpha=0.6)
    constraint_arrows(axes[r, 0], src, targets)
    axes[r, 0].set_title(f"{name}\noriginal image, grid, constraints")
    plot.deformation_grid(phi, dom, ax=axes[r, 1], color="C0", title="transformed grid  Φ(grid)")
    axes[r, 1].scatter(targets[:, 0], targets[:, 1], c="C3", s=25, zorder=6)
    with timer(f"warp row {r}"):
        warped = rn.warp(image, phi, fill=0.0)
    plot.image(warped, ax=axes[r, 2], title="warped image  I(Φ⁻¹(x))")
    plot.jacobian_map(phi, dom, ax=axes[r, 3], resolution=160)
    st = rn.jacobian_stats(phi, dom.grid(160))
    axes[r, 3].set_title(f"det J_Φ   min {st.det.min():.2f}  max {st.det.max():.2f}  folds {100 * st.fold_fraction:.1f}%")
    print(f"    {name}: {st.summary()}")
save(fig, "demo_d_deformation.png")
