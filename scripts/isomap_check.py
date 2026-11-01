"""Does a curved map (Isomap, as in Denby 2023) show a split that the flat PCA map hides?

Every k the analysis uses (Scribes 1-5, faint removed), L2-normalised DINOv3 embeddings. Two 2D maps: the PCA of the
page-hand unit means used by letters_map.py, and Isomap (10 neighbours) fitted on the letters. For each map, the line
enclosing the densest half of each scribe's letters (Scribe 4 has too few letters), and how well map position alone
tells Scribe 1 from Scribes 2-3: 25-nearest-neighbour classifier, 5 folds held out by sheet, balanced accuracy.
Outputs: outputs/maps/isomap_check.png, outputs/maps/isomap_check.txt.
Usage: python scripts/isomap_check.py
"""
import csv

import matplotlib
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from classifier_test import ROOT, bifolium, embeddings_for, folio
from davis_timm_tests import letters
from letters_map import COLOURS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    units = list(csv.DictReader(open(ROOT / "outputs/unsupervised_classification/units_dinov3_unet_all.csv")))
    u = np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units])
    k = [r for r in letters(True) if not r["faint"]]
    x = embeddings_for("dinov3_unet_all", k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    s = np.array([r["scribe"] for r in k])
    sheet = np.array([bifolium(folio(r["page"])) for r in k])
    maps = {"PCA of page means": PCA(2).fit(u).transform(x),
            "Isomap, 10 neighbours": Isomap(n_neighbors=10, n_components=2).fit_transform(x)}
    m = np.isin(s, ["1", "2", "3"])
    y = (s[m] == "1").astype(int)
    lines = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, z) in zip(axes, maps.items()):
        pred = cross_val_predict(KNeighborsClassifier(25), z[m], y, groups=sheet[m], cv=GroupKFold(5))
        lines.append(f"{name}: Scribe 1 vs Scribes 2-3 from map position alone, held out by sheet: "
                     f"balanced accuracy {balanced_accuracy_score(y, pred):.2f}")
        (x0, x1), (y0, y1) = (np.percentile(z[:, d], (0.5, 99.5)) for d in (0, 1))
        gx, gy = np.meshgrid(np.linspace(x0, x1, 150), np.linspace(y0, y1, 150))
        ax.scatter(z[:, 0], z[:, 1], s=1, c="0.8", zorder=0)
        for c in ("1", "2", "3", "5"):
            kde = gaussian_kde(z[s == c].T)
            colour = tuple(v / 255 for v in COLOURS[c])
            ax.contour(gx, gy, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape),
                       levels=[np.percentile(kde(z[s == c].T), 50)], colors=[colour], linewidths=2)
            ax.plot([], [], color=colour, label=f"Scribe {c}")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/maps/isomap_check.png", dpi=100)
    (ROOT / "outputs/maps/isomap_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
