"""Demo H -- learn a geometry from observations of how signals spread across it.

This tests the central proposition directly: with the *representation* (the
input patterns) fixed, can useful behaviour be recovered by changing the
*geometry*?

Setup
-----
* A reference anisotropic metric ``g*`` (smoothly rotating preferred direction).
* Observations: several input patterns diffused under ``g*`` with an
  independent, higher-resolution solver (2× finer grid), then sampled onto the
  training grid, at several times.
* Model: an 8×8 ``GridMetric`` (Cholesky parameterisation) initialised at ``g = I``.
* Loss: ``Σ_{k,t} ‖Diffuse(I_k, g_θ, t) − I_k,t^obs‖² + λ ‖∇g‖²``, minimised by Adam.
  Gradients flow through the finite-element Laplace–Beltrami operator.
* Evaluation on *held-out* patterns and a *held-out* (longer) time.

What is identifiable
--------------------
In 2-D, ``Δ_{λg} = Δ_g / λ``: a conformal rescaling of the metric only rescales
time locally.  Diffusion observations therefore constrain the *shape* of the
metric (``g / √det g``: orientation and anisotropy) much more strongly than its
scale, and a single pattern at a single time does not identify the metric at
all.  We report the error of the shape and of ``det g`` separately.
"""

import math

import torch

from _common import plt, rn, save, timer
from riemann_and_sons import learning, plot, synthetic

torch.manual_seed(0)
dom = rn.Plane()
N_OBS, N_TRAIN = 64, 32
TIMES_TRAIN = [0.001, 0.002, 0.004]
TIME_HELDOUT = 0.006


def reference_metric_fn(x):
    phi = 0.9 * torch.sin(2 * math.pi * x[..., 0]) + 0.6 * torch.cos(2 * math.pi * x[..., 1])
    t = torch.stack([torch.cos(phi), torch.sin(phi)], -1)
    eye = torch.eye(2, dtype=x.dtype)
    return eye + 5.0 * t[..., :, None] * t[..., None, :]  # expensive across t, cheap along it


ref = rn.Geometry(dom, rn.FunctionMetric(reference_metric_fn, dim=2))

# ---------------------------------------------------------------- patterns
def impulse(c, s=0.04):
    return lambda p: torch.exp(-((p - torch.tensor(c)) ** 2).sum(-1) / (2 * s**2))


def smooth_noise(seed, sigma=2.0):
    """A random smooth field: excites all directions everywhere (coverage for identifiability)."""
    gen = torch.Generator().manual_seed(seed)

    def fn(p):
        n = p.shape[0]
        base = rn.Field(torch.rand(n, n, generator=gen), dom)
        return rn.fields.gaussian_blur(base, sigma * n / 32).values

    return fn


patterns = {
    "impulse A": impulse([0.3, 0.35]),
    "impulse B": impulse([0.7, 0.65]),
    "stripes": lambda p: 0.5 + 0.5 * torch.sin(8 * math.pi * p[..., 0]),
    "checkerboard": lambda p: ((p[..., 0] * 6).floor() + (p[..., 1] * 6).floor()) % 2,
    "disk": lambda p: (((p - 0.5) ** 2).sum(-1) < 0.06).to(p.dtype),
    "smooth noise 1": smooth_noise(1),
    "smooth noise 2": smooth_noise(2),
    "impulse C (held out)": impulse([0.5, 0.75]),
    "bar (held out)": lambda p: ((p[..., 0] > 0.4) & (p[..., 0] < 0.6)).to(p.dtype),
    "smooth noise 3 (held out)": smooth_noise(3),
}
train_names = [n for n in patterns if "held out" not in n]
test_names = [n for n in patterns if "held out" in n]

with timer("generating observations with the 2x-resolution reference solver"):
    obs = {}
    for name, fn in patterns.items():
        f_hi = rn.Field.from_function(dom, N_OBS, fn)
        times, snaps = [], []
        f = f_hi
        t_prev = 0.0
        for t in TIMES_TRAIN + [TIME_HELDOUT]:
            f = rn.diffuse(f, ref, t - t_prev, detach_metric=True)
            t_prev = t
            snaps.append(f.sample(dom.grid(N_TRAIN)).detach())  # observation on the training grid
        obs[name] = snaps
inputs = {name: rn.Field.from_function(dom, N_TRAIN, fn) for name, fn in patterns.items()}


def predict(geometry, field, times):
    """Diffuse once, collecting snapshots at ``times`` (single differentiable pass)."""
    out, f, t_prev = [], field, 0.0
    for t in times:
        f = rn.diffuse(f, geometry, t - t_prev)
        t_prev = t
        out.append(f.values)
    return out


def metric_errors(geometry):
    pts = dom.grid(48)
    with torch.no_grad():
        g, g_ref = geometry.g(pts), ref.g(pts)
        shape = lambda m: m / torch.sqrt(torch.linalg.det(m))[..., None, None]
        e_shape = ((shape(g) - shape(g_ref)).norm(dim=(-1, -2)) / shape(g_ref).norm(dim=(-1, -2))).mean().item()
        e_det = (torch.log(torch.linalg.det(g)) - torch.log(torch.linalg.det(g_ref))).abs().mean().item()
    return e_shape, e_det


# ---------------------------------------------------------------- training
gm = rn.GridMetric(dom, 8)
geo = rn.Geometry(dom, gm)
euclid = rn.Geometry(dom)
opt = torch.optim.Adam(gm.parameters(), lr=0.03)
reg_pts = dom.grid(12).reshape(-1, 2)
SMOOTH = 1e-7  # ‖∇g‖² is O(10³) for a metric with anisotropy 6 rotating once across the domain; keep the prior far below the data term
ITERS = 120
history = {"train": [], "heldout": [], "shape": [], "det": []}


def heldout_loss(geometry):
    with torch.no_grad():
        tot = 0.0
        for name in test_names:
            preds = predict(geometry, inputs[name], TIMES_TRAIN + [TIME_HELDOUT])
            tot += sum(((p - o) ** 2).mean().item() for p, o in zip(preds, obs[name]))
        return tot / len(test_names)


baseline = heldout_loss(euclid)
print(f"    held-out loss with g = I: {baseline:.3e}")
with timer("training the metric through the diffusion operator"):
    for it in range(ITERS):
        opt.zero_grad()
        loss = 0.0
        for name in train_names:
            preds = predict(geo, inputs[name], TIMES_TRAIN)
            loss = loss + sum(((p - o) ** 2).mean() for p, o in zip(preds, obs[name]))
        loss = loss / len(train_names) + SMOOTH * learning.metric_smoothness(gm, reg_pts)
        loss.backward()
        opt.step()
        if it % 10 == 0 or it == ITERS - 1:
            e_shape, e_det = metric_errors(geo)
            ho = heldout_loss(geo)
            history["train"].append(loss.item()); history["heldout"].append(ho); history["shape"].append(e_shape); history["det"].append(e_det)
            print(f"    it {it:3d}  train {loss.item():.3e}  held-out {ho:.3e}  metric shape err {e_shape:.3f}  |Δ log det| {e_det:.3f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
plot.metric_field(ref, ax=axes[0, 0], resolution=(16, 16), color="C0", normalize="local", title="reference metric g*  (unit balls, anisotropy 6 everywhere)")
plot.metric_field(geo, ax=axes[0, 1], resolution=(16, 16), color_by="anisotropy", normalize="local", title="learned metric g_θ  (8×8 grid, from diffusion only)")
its = list(range(0, ITERS, 10)) + [ITERS - 1]
axes[0, 2].plot(its, history["train"], label="train loss"); axes[0, 2].plot(its, history["heldout"], label="held-out loss (new patterns, t = 0.006)")
axes[0, 2].axhline(baseline, color="k", ls="--", lw=0.8, label="held-out loss with g = I")
axes[0, 2].set_yscale("log"); axes[0, 2].set_xlabel("iteration"); axes[0, 2].legend(fontsize=8); axes[0, 2].set_title("diffusion-matching loss")
axes[0, 3].plot(its, history["shape"], label="metric shape error  ‖ĝ − ĝ*‖/‖ĝ*‖,  ĝ = g/√det g")
axes[0, 3].plot(its, history["det"], label="mean |log det g − log det g*|")
axes[0, 3].set_xlabel("iteration"); axes[0, 3].legend(fontsize=8); axes[0, 3].set_title("recovery of the metric (identifiability)")

name = test_names[0]
t_show = TIME_HELDOUT
pred_learned = predict(geo, inputs[name], [t_show])[0].detach()
pred_euclid = predict(euclid, inputs[name], [t_show])[0]
truth = obs[name][-1]
vmax = float(truth.max())
plot.heatmap(inputs[name].values, dom, ax=axes[1, 0], cmap="gray", title=f"held-out input: {name}", colorbar=False)
plot.heatmap(truth, dom, ax=axes[1, 1], cmap="gray", vmin=0, vmax=vmax, title=f"observed (reference g*, t = {t_show})", colorbar=False)
plot.heatmap(pred_learned, dom, ax=axes[1, 2], cmap="gray", vmin=0, vmax=vmax, title=f"predicted with learned g_θ   MSE {((pred_learned - truth) ** 2).mean():.2e}", colorbar=False)
plot.heatmap(pred_euclid, dom, ax=axes[1, 3], cmap="gray", vmin=0, vmax=vmax, title=f"predicted with g = I   MSE {((pred_euclid - truth) ** 2).mean():.2e}", colorbar=False)
save(fig, "demo_h_learn_geometry_from_diffusion.png")
