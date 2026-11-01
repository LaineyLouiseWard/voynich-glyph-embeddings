"""Pages on the first two principal components, coloured by Davis scribe and by Currier A/B.

One point per page-hand unit (page x Davis scribe; mean of its letters' L2-normalised DINOv3 embeddings, faint letters
removed) from outputs/unsupervised_classification/units_dinov3_unet_all.csv, the units the grouping without labels uses. PCA fitted on
the units, as in scripts/letters_map.py. Left: scribe colours of letters_map.py. Right: Currier A brown, B grey (not
scribe colours). Output: outputs/unsupervised_classification/spread_dinov3_unet_all.png.
Usage: python scripts/page_spread.py
"""
import csv

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

from classifier_test import ROOT
from letters_map import COLOURS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANGUAGE = {"A": "#8c564b", "B": "#7f7f7f"}


def main():
    units = list(csv.DictReader(open(ROOT / "outputs/unsupervised_classification/units_dinov3_unet_all.csv")))
    z = PCA(2).fit_transform(np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for s, colour in COLOURS.items():
        m = np.array([r["scribe"] == s for r in units])
        axes[0].scatter(z[m, 0], z[m, 1], s=22, color=tuple(v / 255 for v in colour), label=f"Scribe {s} ({m.sum()})")
    for language, colour in LANGUAGE.items():
        m = np.array([r["language"] == language for r in units])
        axes[1].scatter(z[m, 0], z[m, 1], s=22, color=colour, label=f"Currier {language} ({m.sum()})")
    for ax, title in zip(axes, ("Pages by Davis scribe", "Pages by Currier A/B")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC 1")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("PC 2")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/unsupervised_classification/spread_dinov3_unet_all.png", dpi=100)


if __name__ == "__main__":
    main()
