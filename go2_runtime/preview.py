#!/usr/bin/env python3
"""Render the latest real snapshot; RGB and depth are shown separately, not aligned."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
data = np.load(args.run/"latest.npz")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(data["rgb"]); axes[0].set_title("Live RGB (separate stream)")
axes[1].imshow(data["depth"], vmin=0, vmax=4, cmap="turbo"); axes[1].set_title("Depth (0-4 m)")
axes[2].imshow(data["occ"], origin="lower", extent=(-3, 3, 0, 6), cmap="gray_r")
path = data["path"]
axes[2].plot(path[:, 1], path[:, 0], "r-", label="FMM")
if "mppi_plan" in data.files and len(data["mppi_plan"]):
    plan = data["mppi_plan"]
    axes[2].plot(plan[:, 1], plan[:, 0], "c-", label="MPPI candidate")
axes[2].set_title("Real BEV + plans (shadow)"); axes[2].legend()
axes[3].imshow(data["soft_cost"], origin="lower", extent=(-3, 3, 0, 6), cmap="magma")
axes[3].set_title("FMM soft safety cost")
for ax in axes[:2]:
    ax.axis("off")
for ax in axes[2:]:
    ax.set_xlabel("Left (m)"); ax.set_ylabel("Forward (m)")
fig.tight_layout()
fig.savefig(args.run/"preview.png", dpi=140)
print(args.run/"preview.png")
