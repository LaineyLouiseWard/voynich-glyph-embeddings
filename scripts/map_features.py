"""What the letter map shows: each scribe's core area, and measured letter features across the map (after Denby 2023).

Same view as scripts/letters_map.py: every k of Scribes 1-5 (faint removed) projected onto the PCA of the page-hand
unit means. Panel 1: for each scribe, the line enclosing the densest half of its letters (Gaussian kernel density);
Scribe 4 has too few letters (22) for a line.
Panels 2-3: the map cut into the same 26 x 18 grid as letters_all.png; each square with at least 10 letters is
coloured by the median ink height in scan pixels and the median width / height. Also Spearman correlations of PC 1 and PC 2 with height, width / height and ink share, over all letters and
within each of Scribes 1-3. Outputs: outputs/maps/map_features.png, outputs/maps/map_features.txt.
Usage: python scripts/map_features.py
"""
import csv

import matplotlib
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA

from classifier_test import ROOT, embeddings_for
from davis_timm_tests import letters
from letter_size import ink_box
from letters_map import COLOURS, GRID

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_LETTERS = 10
OUTLINED = ("1", "2", "3", "5")


def main():
    units = list(csv.DictReader(open(ROOT / "outputs/unsupervised_classification/units_dinov3_unet_all.csv")))
    u = np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units])
    pca = PCA(2).fit(u)
    k = [r for r in letters(True) if not r["faint"]]
    x = embeddings_for("dinov3_unet_all", k)
    z = pca.transform(x / np.linalg.norm(x, axis=1, keepdims=True))
    s = np.array([r["scribe"] for r in k])
    box = np.array([ink_box(r["annotation_id"]) for r in k])
    ink = np.array([(np.asarray(Image.open(ROOT / "data/crops/unet_all_k" / f"{r['annotation_id']}.png").convert("L"))
                     < 128).mean() for r in k])
    features = {"height (px)": box[:, 0], "width / height": box[:, 1], "ink share": ink}

    lines = []
    for d, axis in enumerate(("PC 1", "PC 2")):
        for name, v in features.items():
            within = ", ".join(f"Scribe {c} {spearmanr(z[s == c, d], v[s == c])[0]:+.2f}" for c in ("1", "2", "3"))
            lines.append(f"{axis} vs {name}: all {spearmanr(z[:, d], v)[0]:+.2f}; within {within}")

    (x0, x1), (y0, y1) = (np.percentile(z[:, d], (0.5, 99.5)) for d in (0, 1))
    xs, ys = np.linspace(x0, x1, GRID[0] + 1), np.linspace(y0, y1, GRID[1] + 1)
    col, row = np.digitize(z[:, 0], xs) - 1, np.digitize(z[:, 1], ys) - 1
    grids = {name: np.full((GRID[1], GRID[0]), np.nan) for name in ("height (px)", "width / height")}
    for a in range(GRID[0]):
        for b in range(GRID[1]):
            inside = (col == a) & (row == b)
            if inside.sum() >= MIN_LETTERS:
                grids["height (px)"][b, a] = np.median(box[inside, 0])
                grids["width / height"][b, a] = np.median(box[inside, 1])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    for c in OUTLINED:
        kde = gaussian_kde(z[s == c].T)
        level = np.percentile(kde(z[s == c].T), 50)
        axes[0].contour(gx, gy, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape), levels=[level],
                        colors=[tuple(v / 255 for v in COLOURS[c])], linewidths=2)
        axes[0].plot([], [], color=tuple(v / 255 for v in COLOURS[c]), label=f"Scribe {c}")
    axes[0].legend(frameon=False)
    axes[0].set_title("Densest half of each scribe's letters", fontsize=10)
    for ax, (name, g) in zip(axes[1:], grids.items()):
        im = ax.imshow(g, origin="lower", extent=(x0, x1, y0, y1), aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"Median {name}", fontsize=10)
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/maps/map_features.png", dpi=100)
    (ROOT / "outputs/maps/map_features.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
