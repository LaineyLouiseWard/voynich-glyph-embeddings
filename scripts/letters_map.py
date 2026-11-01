"""Real letters placed on the page-level spread, framed by scribe.

PCA (2 components) fitted on the page-hand unit means of outputs/unsupervised_classification/units_dinov3_unet_all.csv,
the same view as outputs/unsupervised_classification/spread_dinov3_unet_all.png. Every k in data/all_k.csv, faint
letters removed, is projected onto it from its L2-normalised DINOv3 embedding (dinov3_unet_all). Every letter the tests
and grouping use (Scribes 1-5): the 0.5th to 99.5th percentile range is cut into a grid, and each square holding at
least 3 letters shows the letter nearest its centre as its cleaned crop (data/crops/unet_all/), framed blue, orange,
green, red or purple for Scribe 1, 2, 3, 4 or 5. Output: outputs/maps/letters_all.png.
"""
import csv

import matplotlib
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

from classifier_test import ROOT, embeddings_for
from davis_timm_tests import letters

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"1": (31, 119, 180), "2": (255, 127, 14), "3": (44, 160, 44), "4": (214, 39, 40), "5": (148, 103, 189)}
GRID = (26, 18)
MIN_LETTERS = 3


def tile(letter_id, scribe):
    img = Image.open(ROOT / "data/crops/unet_all" / f"{letter_id}.png").convert("RGB").resize((40, 40))
    return np.asarray(ImageOps.expand(img, border=3, fill=COLOURS[scribe]))


def atlas(k, z):
    keep = [i for i, r in enumerate(k) if r["scribe"] in COLOURS]
    z = z[keep]
    (x0, x1), (y0, y1) = (np.percentile(z[:, d], (0.5, 99.5)) for d in (0, 1))
    xs, ys = np.linspace(x0, x1, GRID[0] + 1), np.linspace(y0, y1, GRID[1] + 1)
    fig, ax = plt.subplots(figsize=(13, 9))
    for a in range(GRID[0]):
        for b in range(GRID[1]):
            inside = np.flatnonzero((z[:, 0] >= xs[a]) & (z[:, 0] < xs[a + 1]) & (z[:, 1] >= ys[b]) & (z[:, 1] < ys[b + 1]))
            if len(inside) < MIN_LETTERS:
                continue
            centre = ((xs[a] + xs[a + 1]) / 2, (ys[b] + ys[b + 1]) / 2)
            i = inside[np.argmin(np.hypot(z[inside, 0] - centre[0], z[inside, 1] - centre[1]))]
            r = k[keep[i]]
            ax.add_artist(AnnotationBbox(OffsetImage(tile(r["annotation_id"], r["scribe"]), zoom=0.9), centre, frameon=False))
    for s in COLOURS:
        ax.scatter([], [], c=[tuple(c / 255 for c in COLOURS[s])], label=f"Scribe {s}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/maps/letters_all.png", dpi=100)


def main():
    units = list(csv.DictReader(open(ROOT / "outputs/unsupervised_classification/units_dinov3_unet_all.csv")))
    u = np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units])
    pca = PCA(2).fit(u)
    k = [r for r in letters(True) if not r["faint"]]
    x = embeddings_for("dinov3_unet_all", k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    atlas(k, pca.transform(x))


if __name__ == "__main__":
    main()
