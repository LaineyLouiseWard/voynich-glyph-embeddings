"""Cut the `n` out of each cleaned `iin`: everything from the start of its last upright stroke.

The upright strokes are found from the ink per column in the lower half of the box (rows from BAND of the height down),
where the two `i` and the `n` stand side by side and the curl of the `n` above them is left out: a stroke is a run of
columns holding ink, and runs are separated by at least one empty column. The cut is the first column of the last run.
The `n` keeps all ink from the cut rightwards, plus the ink above the band of any piece (8-connected) that reaches the
cut, so the curl of the `n` is kept where it bends back over the `i`. A letter whose lower half is one run (strokes
touching) gets no cut and is left out of the `n`. With two runs, the cut is kept only if their widths differ by at most
a factor of RATIO (two joined `i`, then the `n`); a lopsided pair is one `i` and an `i` joined to the `n`, so no cut.
Reads data/all_iin.csv and data/crops/unet_box_iin/; writes data/crops/unet_box_n/ (0 = ink, 255 = background, box
height, from the cut to the right edge) and outputs/cleaning/cut_n.csv.
Usage: python scripts/cut_n.py
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

ROOT = Path(__file__).resolve().parents[1]
MASKS, OUT_MASKS, OUT = ROOT / "data/crops/unet_box_iin", ROOT / "data/crops/unet_box_n", ROOT / "outputs/cleaning"
BAND = 0.5
RATIO = 1.5


def runs(columns):
    """(start, end) of each run of True values."""
    out, start = [], None
    for x, value in enumerate(list(columns) + [False]):
        if value and start is None:
            start = x
        elif not value and start is not None:
            out.append((start, x))
            start = None
    return out


def cut(ink):
    """The `n` mask (box height, from the cut column on) and the cut column, or (None, None) without a cut."""
    height = ink.shape[0]
    top = int(BAND * height)
    strokes = runs(ink[top:].any(axis=0))
    if len(strokes) < 2:
        return None, None
    if len(strokes) == 2:
        (a, b), (c, d) = strokes
        if max(b - a, d - c) > RATIO * min(b - a, d - c):
            return None, None
    x = strokes[-1][0]
    parts = label(ink, connectivity=2)
    reaching = np.isin(parts, np.unique(parts[:, x:][ink[:, x:]]))
    n = np.zeros_like(ink)
    n[:, x:] = ink[:, x:]
    n[:top] |= ink[:top] & reaching[:top]
    columns = np.flatnonzero(n.any(axis=0))
    return n[:, columns.min():], x


def main():
    OUT_MASKS.mkdir(parents=True, exist_ok=True)
    for old in OUT_MASKS.glob("*.png"):
        old.unlink()
    rows = []
    for r in csv.DictReader((ROOT / "data/all_iin.csv").open()):
        ink = np.asarray(Image.open(MASKS / f"{r['letter_id']}.png").convert("L")) < 128
        n, x = cut(ink)
        if n is not None:
            Image.fromarray(np.where(n, 0, 255).astype(np.uint8)).save(OUT_MASKS / f"{r['letter_id']}.png")
        rows.append(dict(letter_id=r["letter_id"], box_width=ink.shape[1], cut=-1 if x is None else x,
                         n_width=0 if n is None else n.shape[1]))
    with (OUT / "cut_n.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['cut'] >= 0 for r in rows)} of {len(rows)} letters cut")


if __name__ == "__main__":
    main()
