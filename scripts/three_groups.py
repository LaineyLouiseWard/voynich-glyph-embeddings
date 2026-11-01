"""Force three groups without labels: do they recover Davis's Scribes 1-3?

Forcing a number of groups always returns that many, so the question is only whether they line up with the scribes.
Letters: every k of Scribes 1-3 (faint removed), L2-normalised DINOv3 embeddings (dinov3_unet_all), split into 3 by
k-means (10 starts, seed 0) and by Ward clustering. Pages: the page-hand unit means of Scribes 1-3
(outputs/unsupervised_classification/units_dinov3_unet_all.csv), 5 PCA dimensions as in scripts/page_heaps.R, split the same two ways.
For each split: the table of groups against scribes, and the adjusted Rand index (1 = the same grouping, 0 = no better
than chance) against Davis's scribe and against Currier A/B.
Output: outputs/unsupervised_classification/three_groups.txt.
Usage: python scripts/three_groups.py
"""
import csv

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from classifier_test import ROOT, embeddings_for
from davis_timm_tests import letters

SCRIBES = ("1", "2", "3")


def report(name, groups, scribe, language):
    lines = [f"{name}: agreement with Davis {adjusted_rand_score(scribe, groups):.3f}, "
             f"with Currier A/B {adjusted_rand_score(language, groups):.3f}"]
    for g in np.unique(groups):
        counts = [int(np.sum((groups == g) & (scribe == s))) for s in SCRIBES]
        total = sum(counts)
        lines.append(f"  group {g} ({total}): " + ", ".join(
            f"Scribe {s} {c} ({100 * c / total:.0f}%)" for s, c in zip(SCRIBES, counts)))
    return lines


def splits(x):
    return {"k-means": KMeans(3, n_init=10, random_state=0).fit_predict(x),
            "Ward": fcluster(linkage(x, method="ward"), 3, criterion="maxclust")}


def main():
    k = [r for r in letters(True) if not r["faint"] and r["scribe"] in SCRIBES]
    x = embeddings_for("dinov3_unet_all", k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    scribe = np.array([r["scribe"] for r in k])
    language = np.array([r["language"] for r in k])
    lines = [f"letters ({len(k)})"]
    for name, groups in splits(x).items():
        lines += report(name, groups, scribe, language)

    units = [r for r in csv.DictReader(open(ROOT / "outputs/unsupervised_classification/units_dinov3_unet_all.csv")) if r["scribe"] in SCRIBES]
    u = np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units])
    z = PCA(5).fit_transform(u - u.mean(axis=0))
    scribe = np.array([r["scribe"] for r in units])
    language = np.array([r["language"] for r in units])
    lines.append(f"page-hand units ({len(units)})")
    for name, groups in splits(z).items():
        lines += report(name, groups, scribe, language)
    (ROOT / "outputs/unsupervised_classification/three_groups.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
