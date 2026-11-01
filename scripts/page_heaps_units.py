"""Page-hand units for scripts/page_heaps.R: one row per page and Davis scribe, the mean of its letters' L2-normalised
embeddings, faint letters removed, with the unit's sheet (Zandbergen $B, as in the classifier tests) and Currier language.

Output: outputs/unsupervised_classification/units_<stem>.csv (page, scribe, language, sheet, letters, e1 ... eD).
Usage: python scripts/page_heaps_units.py [stem]   (default dinov3_unet_all; stems ending _all use data/all_k.csv)
"""
import csv
import sys

import numpy as np

from classifier_test import ROOT, bifolium, embeddings_for, folio
from davis_timm_tests import letters


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "dinov3_unet_all"
    k = [r for r in letters(stem.endswith("_all")) if not r["faint"]]
    x = embeddings_for(stem, k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    units = {}
    for r, v in zip(k, x):
        units.setdefault((r["page"], r["scribe"]), dict(language=r["language"], rows=[]))["rows"].append(v)
    out = ROOT / "outputs/unsupervised_classification"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"units_{stem}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "scribe", "language", "sheet", "letters"] + [f"e{i + 1}" for i in range(x.shape[1])])
        for (page, scribe), u in sorted(units.items()):
            writer.writerow([page, scribe, u["language"], bifolium(folio(page)), len(u["rows"])]
                            + [f"{v:.7g}" for v in np.mean(u["rows"], axis=0)])
    print(f"{stem}: {len(units)} page-hand units from {len(k)} letters")


if __name__ == "__main__":
    main()
