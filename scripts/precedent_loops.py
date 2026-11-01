"""Sheet-bootstrap intervals for the loop and size differences (read-only).

Rules fixed before looking: page value = median over its non-faint k (loop stats over letters with a closed loop,
holes >= 20 px). Group value = median of page values. Difference = group A - group B. 95% interval from 2,000 draws
of whole sheets with replacement within each group (seed 0). Run from scripts/ with PYTHONDONTWRITEBYTECODE=1.
Usage: python scripts/precedent_loops.py > outputs/precedents/precedent_loops.txt
"""
from collections import defaultdict

import numpy as np

from precedent_basic import loop  # noqa: E402
from classifier_test import bifolium, folio  # noqa: E402
from davis_timm_tests import letters  # noqa: E402
from letter_size import ink_box  # noqa: E402

vals = defaultdict(lambda: defaultdict(list))  # (scribe, page) -> stat -> values
for r in letters(True):
    if r["faint"] or r["page"] == "115r":
        continue
    h, wh = ink_box(r["annotation_id"])
    key = (r["scribe"], r["page"])
    vals[key]["height"].append(h)
    vals[key]["w/h"].append(wh)
    lp = loop(r["annotation_id"])
    if lp is not None:
        vals[key]["loop area/h^2"].append(lp[0])
        vals[key]["loop roundness"].append(lp[2])


def pages(scribe, stat):
    out = defaultdict(list)  # sheet -> page values
    for (s, p), d in vals.items():
        if s == scribe and d[stat]:
            out[bifolium(folio(p))].append(np.median(d[stat]))
    return out


def draw(by_sheet, rng):
    sheets = list(by_sheet)
    pick = rng.choice(len(sheets), len(sheets))
    return np.median(np.concatenate([by_sheet[sheets[i]] for i in pick]))


rng = np.random.default_rng(0)
for stat, pairs in (("loop roundness", [("1", "2"), ("1", "3"), ("3", "2")]),
                    ("loop area/h^2", [("1", "2"), ("1", "3"), ("3", "2"), ("4", "1"), ("4", "2"), ("4", "3")]),
                    ("height", [("5", "1"), ("5", "2"), ("5", "3")]),
                    ("w/h", [("5", "1"), ("5", "2"), ("5", "3")])):
    for a, b in pairs:
        pa, pb = pages(a, stat), pages(b, stat)
        obs = np.median(np.concatenate(list(pa.values()))) - np.median(np.concatenate(list(pb.values())))
        boot = [draw(pa, rng) - draw(pb, rng) for _ in range(2000)]
        print(f"{stat}: Scribe {a} - Scribe {b} = {obs:+.3f} [95% {np.quantile(boot, .025):+.3f}, "
              f"{np.quantile(boot, .975):+.3f}]; sheets {len(pa)} and {len(pb)}")
