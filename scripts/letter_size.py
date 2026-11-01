"""True size of each k on the page, per scribe: ink height in scan pixels and width / height of the ink.

Every k in data/all_k.csv, faint letters removed, Scribes 1-5, measured on its box-size U-Net mask
(data/crops/unet_all_k/). Also on herbal pages only (folios 1-66, Scribes 1 and 2), and the height of each page's scan
per scribe, to show the scans share one scale. Output: outputs/size_and_spacing/letter_size.txt.
Usage: python scripts/letter_size.py
"""
import re
from collections import defaultdict

import numpy as np
from PIL import Image

from classifier_test import ROOT
from davis_timm_tests import letters

Image.MAX_IMAGE_PIXELS = None


def ink_box(letter_id):
    """Height in scan pixels and width / height of the ink in a box-size U-Net mask."""
    ys, xs = np.nonzero(np.asarray(Image.open(ROOT / "data/crops/unet_all_k" / f"{letter_id}.png").convert("L")) < 128)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return h, w / h


def summary(values):
    a = np.array(values)
    return f"median {np.median(a):.2f} [IQR {np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}], n={len(a)}"


def main():
    height, ratio, herbal, scan = (defaultdict(list) for _ in range(4))
    pages = {}
    for r in letters(True):
        if r["faint"]:
            continue
        h, wh = ink_box(r["annotation_id"])
        height[r["scribe"]].append(h)
        ratio[r["scribe"]].append(wh)
        if r["scribe"] in ("1", "2") and int(re.match(r"f?(\d+)", r["page"]).group(1)) <= 66:
            herbal[r["scribe"]].append(h)
        pages.setdefault((r["page"], r["scribe"]), r["iipimage"])
    for (_, scribe), image in pages.items():
        scan[scribe].append(Image.open(ROOT / "data/raw/digipal_project/images" / image).size[1])
    lines = []
    for s in ("1", "2", "3", "4", "5"):
        lines.append(f"Scribe {s}: ink height px {summary(height[s])}; width/height {summary(ratio[s])}")
    for s in ("1", "2"):
        lines.append(f"herbal pages, Scribe {s}: ink height px {summary(herbal[s])}")
    for s in ("1", "2", "3"):
        lines.append(f"Scribe {s} page scans: height px {summary(scan[s])}, range {min(scan[s])}-{max(scan[s])}")
    (ROOT / "outputs/size_and_spacing/letter_size.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
