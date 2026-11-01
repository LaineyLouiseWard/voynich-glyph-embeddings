"""Remove neighbouring-letter ink from the raw SBB 2022 k masks.

Reads data/crops/sbb2022_raw/, writes data/crops/sbb2022_k/ (0 = ink, 255 = background, box size).
Usage: python scripts/k_clean.py
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/crops/sbb2022_raw", ROOT / "data/crops/sbb2022_k"


LOWER_FROM = 0.35   # legs are measured from this share of the box height down
LEG_RUN = 0.45      # a leg is a column with an unbroken ink run at least this share of box height
LOOP_ABOVE = 0.4    # ink right of the leg reaching above this share of box height is the loop, kept
FOOT = 0.10         # ink kept right of the right leg, as a share of box width (the foot or tick)
SMALL = 0.25        # detached pieces smaller than this share of the main piece are candidates to drop


def longest_run(column):
    best = run = 0
    for value in column:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def right_leg(ink):
    """Right edge (x) of the rightmost leg, or None if no leg is found."""
    height = ink.shape[0]
    lower = ink[int(LOWER_FROM * height):]
    legs = [x for x in range(ink.shape[1]) if longest_run(lower[:, x]) >= LEG_RUN * height]
    return max(legs) if legs else None


def keep_mask(ink):
    """Boolean mask of the ink that belongs to the k."""
    height, width = ink.shape
    kept = ink.copy()
    leg = right_leg(ink)
    if leg is not None:
        start = leg + 1 + round(FOOT * width)
        right = label(ink[:, start:], connectivity=2)
        for r in regionprops(right):
            if r.bbox[0] >= LOOP_ABOVE * height:
                kept[:, start:][right == r.label] = False
    parts = label(kept, connectivity=2)
    regions = regionprops(parts)
    if not regions:
        return kept
    main = max(regions, key=lambda r: r.area)
    top, left, bottom, right = main.bbox
    for r in regions:
        if r is main:
            continue
        r_top, r_left, r_bottom, r_right = r.bbox
        on_edge = r_top == 0 or r_left == 0 or r_bottom == height or r_right == width
        outside = r_bottom <= top + 2 or r_top >= bottom - 2 or r_right <= left + 2 or r_left >= right - 2
        if (on_edge and outside) or (r.area < SMALL * main.area and (on_edge or outside)):
            kept[parts == r.label] = False
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW.glob("*.png"), key=lambda p: int(p.stem)):
        ink = np.asarray(Image.open(path).convert("L")) < 128
        kept = keep_mask(ink)
        Image.fromarray(np.where(kept, 0, 255).astype(np.uint8)).save(OUT / path.name)


if __name__ == "__main__":
    main()
