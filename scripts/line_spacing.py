"""Does `k` size follow line spacing, as Timm's cramped-writing argument predicts?

Line pitch per page = median gap between consecutive line centres of the aligned word boxes (data/word_boxes/aligned.csv,
kept pages, at least 4 lines); letter size from the box-size U-Net masks (letter_size.ink_box), non-faint k of
Scribes 1-3. Reports: Scribe 1 herbal against pharmaceutical pages; per-scribe median line pitch; per-scribe Spearman
correlation over pages between line pitch and page median k height (pages with 5 or more k, 115r left out); and a
page-level least-squares fit of k height and width / height on line pitch plus a Scribe 1 term, with a 95% interval
from 2,000 resamples of whole sheets (seed 0). Every result is repeated with a second spacing measure taken from the
ink: the bottoms of the k boxes on consecutive transcription lines. Output: outputs/size_and_spacing/line_spacing.txt.
Usage: python scripts/line_spacing.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from classifier_test import ROOT, bifolium, folio
from davis_timm_tests import letters
from letter_size import ink_box

out = []
labels = {r["page"]: r for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}

height, ratio = defaultdict(list), defaultdict(list)
for r in letters(True):
    if r["faint"] or r["scribe"] not in ("1", "2", "3"):
        continue
    h, wh = ink_box(r["annotation_id"])
    height[(r["page"], r["scribe"])].append(h)
    ratio[(r["page"], r["scribe"])].append(wh)

words = defaultdict(lambda: defaultdict(list))
for w in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open()):
    if w["kept"] == "1":
        words[w["page"]][int(w["line"])].append((float(w["y0"]) + float(w["y1"])) / 2)
pitch = {}
for page, lines in words.items():
    centres = [np.median(v) for _, v in sorted(lines.items())]
    diffs = [d for d in np.diff(centres) if d > 0]
    if len(centres) >= 4 and diffs:
        pitch[page] = float(np.median(diffs))

# Second, independent measure from the ink: the bottom of each k box sits on its line. Per page, the median bottom
# of the k on each transcription line; spacing = median over consecutive k-bearing lines of gap / lines apart.
bottoms = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader((ROOT / "data/all_k.csv").open()):
    if r["line"]:
        bottoms[r["page"]][int(r["line"])].append(float(r["y1"]))
k_pitch = {}
for page, lines in bottoms.items():
    ls = sorted(lines)
    ys = [np.median(lines[n]) for n in ls]
    gaps = [(ys[i + 1] - ys[i]) / (ls[i + 1] - ls[i]) for i in range(len(ls) - 1) if ys[i + 1] > ys[i]]
    if len(ls) >= 3 and len(gaps) >= 2:
        k_pitch[page] = float(np.median(gaps))
both = sorted(set(pitch) & set(k_pitch) - {"115r"})
a, b = np.array([pitch[p] for p in both]), np.array([k_pitch[p] for p in both])
out.append(f"Two measures of line spacing, {len(both)} pages with both: Spearman {spearmanr(a, b)[0]:+.2f}, "
           f"median absolute difference {np.median(np.abs(a - b)):.1f} px")


def illus(page):
    return labels.get(page, {}).get("illustration", "")


def analyse(pitch, label):
    out.append(f"== line spacing from {label}")
    for s in ("1", "2", "3"):
        ps = sorted({pitch[p] for (p, sc) in height if sc == s and p in pitch and p != "115r"})
        out.append(f"Scribe {s}: median line spacing {np.median(ps):.1f} px over {len(ps)} pages")
    for code, name in (("H", "herbal"), ("P", "pharma")):
        keys = [k for k in height if k[1] == "1" and illus(k[0]) == code]
        hs = [h for k in keys for h in height[k]]
        rs = [x for k in keys for x in ratio[k]]
        ps = [pitch[k[0]] for k in keys if k[0] in pitch]
        out.append(f"Scribe 1 {name}: {len(keys)} pages, {len(hs)} k; k height median {np.median(hs):.1f}; "
                   f"w/h {np.median(rs):.3f}; line spacing median {np.median(ps):.1f} px over {len(ps)} pages")
    for s in ("1", "2", "3"):
        pts = [(pitch[p], np.median(v)) for (p, sc), v in height.items()
               if sc == s and len(v) >= 5 and p in pitch and p != "115r"]
        x, y = zip(*pts)
        rho, pval = spearmanr(x, y)
        out.append(f"Scribe {s}: Spearman(line spacing, page median k height) = {rho:+.2f}, p = {pval:.3f}, {len(pts)} pages")
    # Does Scribe 1 (A) differ from Scribes 2-3 (B) at equal spacing? Page level, pages with >= 5 non-faint k,
    # 115r out. Least squares: page median ~ 1 + spacing + [Scribe 1]; 95% interval from 2,000 sheet resamples.
    rows = [(p, s, pitch[p], np.median(height[(p, s)]), np.median(ratio[(p, s)]))
            for (p, s) in height if len(height[(p, s)]) >= 5 and p in pitch and p != "115r"]
    sheet = np.array([bifolium(folio(r[0])) for r in rows])
    X = np.column_stack([np.ones(len(rows)), [r[2] for r in rows], [r[1] == "1" for r in rows]]).astype(float)
    for col, name in ((3, "k height px"), (4, "k width/height")):
        y = np.array([r[col] for r in rows])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rng = np.random.default_rng(0)
        sheets = np.unique(sheet)
        boot = []
        for _ in range(2000):
            idx = np.concatenate([np.flatnonzero(sheet == s) for s in rng.choice(sheets, len(sheets))])
            if len(set(X[idx, 2])) == 2:
                boot.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][2])
        out.append(f"{name}: Scribe 1 term at equal spacing {beta[2]:+.3f} [95% {np.quantile(boot, .025):+.3f}, "
                   f"{np.quantile(boot, .975):+.3f}], slope {beta[1]:+.3f} per px; {len(rows)} pages, {len(sheets)} sheets")
    s1 = sorted(r[2] for r in rows if r[1] == "1")
    bp = sorted(r[2] for r in rows if r[1] != "1")
    out.append(f"spacing ranges: Scribe 1 {s1[0]:.0f}-{s1[-1]:.0f} (n={len(s1)}), Scribes 2-3 {bp[0]:.0f}-{bp[-1]:.0f} "
               f"(n={len(bp)}); Scribe 1 pages under 72 px: {sum(p < 72 for p in s1)}; B pages over 72 px: {sum(p > 72 for p in bp)}")


analyse(pitch, "word outlines")
analyse(k_pitch, "k bottoms")
(ROOT / "outputs/size_and_spacing/line_spacing.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
