"""Average k shape per scribe group, as in the heatmaps of Popović, Dhali & Schomaker 2021.

Main version: U-Net masks (data/crops/unet_k/), faint letters removed. Each mask is scaled so its ink is 100 px
tall (aspect kept), placed with its ink centroid at the centre of a 160 x 160 canvas, and averaged per group.
Each pixel of an average = share of letters with ink there. Output: outputs/letter_shapes/average_k.png.
With --all: every k in data/all_k.csv (masks in data/crops/unet_all_k/), output outputs/letter_shapes/average_k_all.png.
"""
import sys

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

from classifier_test import ROOT
from davis_timm_tests import letters

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SIZE, HEIGHT = 160, 100
FRAMES = {"Scribe 1": "#1f77b4", "Scribe 2": "#ff7f0e", "Scribe 3": "#2ca02c"}


def aligned(annotation_id, crops="unet_k"):
    ink = np.asarray(Image.open(ROOT / "data/crops" / crops / f"{annotation_id}.png").convert("L")) < 128
    rows, cols = np.nonzero(ink)
    ink = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)
    scale = HEIGHT / ink.shape[0]
    width = max(1, round(ink.shape[1] * scale))
    ink = np.asarray(Image.fromarray(ink * 255).resize((width, HEIGHT), Image.Resampling.BILINEAR)) / 255
    canvas = np.zeros((SIZE, SIZE), np.float32)
    top, left = (SIZE - HEIGHT) // 2, (SIZE - width) // 2
    canvas[top:top + HEIGHT, left:left + width] = ink
    cy, cx = center_of_mass(canvas)
    return shift(canvas, (SIZE / 2 - cy, SIZE / 2 - cx), order=1)


def main():
    every = "--all" in sys.argv
    crops, suffix = ("unet_all_k", "_all") if every else ("unet_k", "")
    k = [r for r in letters(every) if not r["faint"]]
    groups = {"Scribe 1": ["1"], "Scribe 2": ["2"], "Scribe 3": ["3"], "Scribes 2 and 3": ["2", "3"]}
    means = {name: np.mean([aligned(r["annotation_id"], crops) for r in k if r["scribe"] in s], axis=0)
             for name, s in groups.items()}
    counts = {name: sum(r["scribe"] in s for r in k) for name, s in groups.items()}
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for ax, name in zip(axes, ["Scribe 1", "Scribe 2", "Scribe 3", "Scribes 2 and 3"]):
        ax.imshow(means[name], cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"{name} (n={counts[name]})", fontsize=9)
    diff = means["Scribe 1"] - means["Scribes 2 and 3"]
    limit = np.abs(diff).max()
    blue_orange = LinearSegmentedColormap.from_list("blue_orange", [FRAMES["Scribe 2"], "white", FRAMES["Scribe 1"]])
    axes[4].imshow(diff, cmap=blue_orange, vmin=-limit, vmax=limit)
    axes[4].set_title("Scribe 1 minus Scribes 2 and 3", fontsize=9)
    for ax, name in zip(axes, ["Scribe 1", "Scribe 2", "Scribe 3", "Scribes 2 and 3", ""]):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(name in FRAMES)
            spine.set_color(FRAMES.get(name, "white"))
            spine.set_linewidth(3)
    fig.tight_layout()
    fig.savefig(ROOT / f"outputs/letter_shapes/average_k{suffix}.png", dpi=150)
    np.save(ROOT / f"outputs/letter_shapes/average_k{suffix}.npy", np.stack([means[n] for n in groups]))


if __name__ == "__main__":
    main()
