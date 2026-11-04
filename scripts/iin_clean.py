"""Remove neighbouring-letter ink from the raw SBB 2022 iin masks (the iin sibling of k_clean.py).

An `iin` is two short `i` strokes and an `n`, often three separate pieces, so k_clean.py's rule (drop small detached
pieces) would delete the `i`. Here, with pieces 8-connected: the largest piece (the `n`) and every piece touching no box
edge are kept. A piece cut by a box edge is neighbour ink (the letter before, the lines above and below) and dropped
when it is shorter than EDGE times the height of the letter's own `i` (the median height of the inner pieces at least
I_MIN of the box tall; I_DEFAULT of the box if there are none), when it touches the left edge and two such `i` are
already inside the box, or when it touches the bottom or right edge right of the centre of the `n` (nothing of the
`iin` but the `n` lies there).
Reads data/crops/sbb2022_raw_iin/, writes data/crops/sbb2022_iin/ (0 = ink, 255 = background, box size).
Usage: python scripts/iin_clean.py
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/crops/sbb2022_raw_iin", ROOT / "data/crops/sbb2022_iin"

EDGE = 0.6       # an edge piece shorter than this share of the i height is dropped
I_MIN = 0.2      # inner pieces at least this share of the box height count as i strokes
I_DEFAULT = 0.35  # i height, as a share of the box height, when no inner i is found


def keep_mask(ink):
    """Boolean mask of the ink that belongs to the iin."""
    height, width = ink.shape
    parts = label(ink, connectivity=2)
    regions = regionprops(parts)
    if not regions:
        return ink.copy()
    main = max(regions, key=lambda r: r.area)

    def on_edge(r):
        top, left, bottom, right = r.bbox
        return top == 0 or left == 0 or bottom == height or right == width

    strokes = [r for r in regions if r is not main and not on_edge(r) and r.bbox[2] - r.bbox[0] >= I_MIN * height
               and r.bbox[1] < main.bbox[1]]
    i_height = np.median([r.bbox[2] - r.bbox[0] for r in strokes]) if strokes else I_DEFAULT * height
    kept = ink.copy()
    for r in regions:
        if r is main or not on_edge(r):
            continue
        short = r.bbox[2] - r.bbox[0] < EDGE * i_height
        extra = r.bbox[1] == 0 and len(strokes) >= 2
        right_of_n = (r.bbox[2] == height or r.bbox[3] == width) and r.bbox[1] >= (main.bbox[1] + main.bbox[3]) / 2
        if short or extra or right_of_n:
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
