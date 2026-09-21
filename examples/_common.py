"""Shared boilerplate for the demo scripts."""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import riemann_and_sons as rn  # noqa: E402,F401

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def save(fig, name: str, dpi: int = 110) -> str:
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {os.path.relpath(path)}")
    return path


class timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.time() - self.t0:.1f}s", flush=True)


def constraint_arrows(ax, src, dst, color="C3"):
    d = (dst - src).numpy()
    s = src.numpy()
    ax.quiver(s[:, 0], s[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, color=color, width=0.008, zorder=6)
    ax.scatter(s[:, 0], s[:, 1], c="white", edgecolors=color, s=40, zorder=7)
