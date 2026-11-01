"""Read-only checks of basic descriptive claims (Davis 2020/2022, Timm 2026, Carter 1946).

Rules fixed before looking at any number:
- Letters: every k from letters(True), faint letters removed (also reported with faint kept for Scribes 4 and 5).
- Size: ink_box from letter_size.py (ink height in scan pixels, width / height).
- Loop: holes = filled ink minus ink in the box-size U-Net mask; holes under 20 px ignored; the largest hole is the
  loop. Reported: share of letters with a closed loop, loop area / ink height^2 (relative loop size),
  loop height / ink height, loop roundness = minor / major axis (1 = circle).
- Each statistic: median over letters [IQR], and median of page medians (pages counted equally).
- Spacing: kept pages of data/word_boxes/aligned.csv, Davis scribe 1-3 from config/zl_page_labels.csv, 115r left out.
  Line centre = median word-box centre y of its words; line pitch = median of positive differences between
  consecutive line centres; word gap = median x0(next) - x1(previous) within a line; word-box height median.
  Per scribe: median of page values. Pages with fewer than 4 lines left out of pitch.
Run from scripts/ with PYTHONDONTWRITEBYTECODE=1 so nothing is written inside the repo.
Usage: python scripts/precedent_basic.py > outputs/precedents/precedent_basic.txt
"""
import csv
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from skimage.measure import regionprops

from classifier_test import ROOT, bifolium, folio  # noqa: E402
from davis_timm_tests import letters  # noqa: E402
from letter_size import ink_box  # noqa: E402

MIN_HOLE = 20


def loop(letter_id):
    ink = np.asarray(Image.open(ROOT / "data/crops/unet_all_k" / f"{letter_id}.png").convert("L")) < 128
    ys, xs = np.nonzero(ink)
    h = ys.max() - ys.min() + 1
    holes, n = label(binary_fill_holes(ink) & ~ink)
    props = [p for p in regionprops(holes) if p.area >= MIN_HOLE]
    if not props:
        return None
    p = max(props, key=lambda p: p.area)
    r0, _, r1, _ = p.bbox
    round_ = p.minor_axis_length / p.major_axis_length if p.major_axis_length > 0 else np.nan
    return p.area / h ** 2, (r1 - r0) / h, round_


def summary(values):
    a = np.asarray(values, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n=0"
    return f"{np.median(a):.3f} [IQR {np.percentile(a, 25):.3f}-{np.percentile(a, 75):.3f}], n={len(a)}"


def page_median(values_by_page):
    meds = [np.nanmedian(v) for v in values_by_page.values() if len(v)]
    return f"{np.median(meds):.3f} over {len(meds)} pages"


def herbal(page):
    return folio(page) <= 66


def main():
    rows = letters(True)
    stats = defaultdict(lambda: defaultdict(list))         # key -> stat -> values
    by_page = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # key -> stat -> page -> values
    sheets = defaultdict(set)
    for r in rows:
        s = r["scribe"]
        keys = [f"S{s} all"] if r["faint"] else [f"S{s}", f"S{s} all"]
        if not r["faint"]:
            if s == "1":
                keys.append("S1 herbal" if herbal(r["page"]) else "S1 non-herbal")
            if s == "2":
                f = folio(r["page"])
                keys.append("S2 herbal" if f <= 66 else ("S2 bathing 75-84" if 75 <= f <= 84 else "S2 other"))
            if s == "3":
                keys.append("S3 herbal" if herbal(r["page"]) else "S3 non-herbal")
            if s == "5":
                keys.append(f"S5 sheet {bifolium(folio(r['page']))}")
        try:
            h, wh = ink_box(r["annotation_id"])
        except ValueError:  # mask with no ink
            print("no ink:", r["annotation_id"], r["scribe"], "faint" if r["faint"] else "not faint")
            continue
        lp = loop(r["annotation_id"])
        for key in keys:
            sheets[key].add(bifolium(folio(r["page"])))
            vals = {"height": h, "w/h": wh, "closed": 0.0 if lp is None else 1.0}
            if lp is not None:
                vals.update({"loop area/h^2": lp[0], "loop h/ink h": lp[1], "loop roundness": lp[2]})
            for stat, v in vals.items():
                stats[key][stat].append(v)
                by_page[key][stat][r["page"]].append(v)
    order = ["S1", "S2", "S3", "S4", "S5", "S4 all", "S5 all", "S1 herbal", "S1 non-herbal", "S2 herbal",
             "S2 bathing 75-84", "S2 other", "S3 herbal", "S3 non-herbal"] + sorted(k for k in stats if "sheet" in k)
    for key in order:
        if key not in stats:
            continue
        pages = {p for p in by_page[key]["height"]}
        print(f"== {key}: {len(stats[key]['height'])} letters, {len(pages)} pages, {len(sheets[key])} sheets")
        for stat in ("height", "w/h", "closed", "loop area/h^2", "loop h/ink h", "loop roundness"):
            if stat == "closed":
                a = np.array(stats[key][stat])
                print(f"   closed loop share: {a.mean():.3f} of {len(a)}")
            else:
                print(f"   {stat}: letters {summary(stats[key][stat])}; page medians {page_median(by_page[key][stat])}")
    s1_pages = sorted({r['page'] for r in rows if r['scribe'] == '1' and not r['faint'] and not herbal(r['page'])},
                      key=lambda p: (folio(p), p))
    print("S1 non-herbal pages:", " ".join(s1_pages))
    spacing()


def spacing():
    hand = {r["page"]: r["davis_hand"] for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}
    words = defaultdict(list)
    for r in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open()):
        if r["kept"] == "1":
            words[r["page"]].append(r)
    out = defaultdict(lambda: defaultdict(list))
    for page, ws in words.items():
        s = hand.get(page, "")
        if page == "115r" or s not in ("1", "2", "3"):
            continue
        lines = defaultdict(list)
        for w in ws:
            lines[int(w["line"])].append(w)
        centres, gaps, heights = [], [], []
        for _, lw in sorted(lines.items()):
            lw = sorted(lw, key=lambda w: int(w["index"]))
            centres.append(np.median([(float(w["y0"]) + float(w["y1"])) / 2 for w in lw]))
            heights += [float(w["y1"]) - float(w["y0"]) for w in lw]
            gaps += [float(b["x0"]) - float(a["x1"]) for a, b in zip(lw[:-1], lw[1:])]
        diffs = [d for d in np.diff(centres) if d > 0]
        keys = [f"S{s}"] + ([f"S{s} herbal"] if folio(page) <= 66 and s in ("1", "2") else [])
        for key in keys:
            if len(centres) >= 4 and diffs:
                out[key]["pitch"].append(np.median(diffs))
                out[key]["pitch / word-box height"].append(np.median(diffs) / np.median(heights))
            if gaps:
                out[key]["word gap"].append(np.median(gaps))
                out[key]["word gap / word-box height"].append(np.median(gaps) / np.median(heights))
            out[key]["word-box height"].append(np.median(heights))
    print("== spacing from aligned Voynichese word boxes (median of page values [IQR], pages)")
    for key in ("S1", "S2", "S3", "S1 herbal", "S2 herbal"):
        for stat in ("pitch", "word gap", "word-box height", "pitch / word-box height", "word gap / word-box height"):
            print(f"   {key} {stat}: {summary(out[key][stat])}")


if __name__ == "__main__":
    main()
