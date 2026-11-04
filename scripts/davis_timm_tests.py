"""Tests where Davis and Timm predict different results, on SBB + U-Net crops (readings fixed in notes/results.md).

1. Text type fixed: Scribe 2 vs Scribe 3 on Currier-B pages, held-out classifier (10 folds by sheet of parchment),
   chance from 200 label shuffles between the same units.
2. Crossbar or text type: classifier trained on all Scribe 1 and 2 `k`, applied to Scribes 3, 4 and 5.
Each test runs on all letters and without faint letters (k ink share in the bottom 10% of the U-Net masks).
Stems ending in _all (dinov3_unet_all, hinge_unet_all) use every k in data/all_k.csv, cleaned in data/crops/unet_all_k/.
Stems ending in _all_iin and _all_n use data/all_iin.csv; faint letters are the bottom 10% of the cleaned `iin`
(data/crops/unet_box_iin/) for both; letters whose cleaned `iin` ink touches the box's top or left edge are
dropped (KEEP_EDGE_INK=1 keeps them, for the sensitivity run); _all_n keeps the letters with a cut `n` (data/crops/unet_box_n/).
Usage: python scripts/davis_timm_tests.py [stem]   (default dinov3_unet)
"""
import csv
import os
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from classifier_test import (ROOT, bifolium, embeddings_for, folio, held_out, page_weights, recall, score,
                             sheet_interval, shuffled_scores)

FAINT_SHARE = 10


def stem_letter(stem):
    """(every, letter) for a stem: _all is every k, _all_iin and _all_n every iin."""
    if stem.endswith("_all"):
        return True, "k"
    for letter in ("iin", "n"):
        if stem.endswith(f"_all_{letter}"):
            return True, letter
    return False, "k"


def letters(every=False, letter="k"):
    """Timm's upright k, or with every=True each k in data/all_k.csv (Davis scribe as both scribe and hand)."""
    language = {r["page"]: r["language"] for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}
    if every:
        table = "all_k" if letter == "k" else "all_iin"
        k = [dict(r, annotation_id=r["letter_id"], scribe=r["davis_scribe"], hand=r["davis_scribe"])
             for r in csv.DictReader((ROOT / f"data/{table}.csv").open())]
        crops = "unet_all_k" if letter == "k" else "unet_box_iin"
    else:
        dropped = {r["annotation_id"] for r in csv.DictReader((ROOT / "config/dropped_tilted_k.csv").open())}
        k = [r for r in csv.DictReader((ROOT / "data/glyphs.csv").open())
             if r["allograph"] == "k" and r["annotation_id"] not in dropped]
        crops = "unet_k"
    ink = np.array([(np.asarray(Image.open(ROOT / "data/crops" / crops / f"{r['annotation_id']}.png").convert("L")) < 128).mean()
                    for r in k])
    faint = ink < np.percentile(ink, FAINT_SHARE)
    for r, flag in zip(k, faint):
        r["language"] = language.get(r["page"].lstrip("f"), "")
        r["faint"] = bool(flag)
    if letter in ("iin", "n") and not os.environ.get("KEEP_EDGE_INK"):
        k = [r for r in k if not touches_top_or_left(ROOT / "data/crops" / crops / f"{r['annotation_id']}.png")]
    if letter == "n":
        k = [r for r in k if (ROOT / "data/crops/unet_box_n" / f"{r['annotation_id']}.png").exists()]
    return k


def touches_top_or_left(path):
    """Cleaned ink on the box's top row or left column: ink from the line above or the letter before was left in."""
    ink = np.asarray(Image.open(path).convert("L")) < 128
    return bool(ink[0].any() or ink[:, 0].any())


def text_type_fixed(k, x, label):
    index = [i for i, r in enumerate(k) if r["scribe"] in ("2", "3") and r["language"] == "B"]
    rows, xs = [k[i] for i in index], x[index]
    y = np.array([int(r["scribe"]) for r in rows])
    group = np.array([bifolium(folio(r["page"])) for r in rows])
    unit_key = list(zip(group, [r["hand"] for r in rows]))
    units = {u: i for i, u in enumerate(dict.fromkeys(unit_key))}
    member = np.array([units[u] for u in unit_key])
    unit_labels = np.array([y[member == i][0] for i in range(len(units))])
    folds = list(GroupKFold(n_splits=10).split(xs, y, group))
    weights = page_weights(rows)
    prediction = held_out(xs, y, folds, weights)
    observed = score(y, prediction, weights)
    low, high = sheet_interval(xs, y, weights, member, group, observed)
    null = shuffled_scores(xs, unit_labels, member, folds, weights)
    per = {s: recall(y, prediction, weights, s) for s in (2, 3)}
    print(f"[{label}] text type fixed, Scribe 2 (n={np.sum(y == 2)}) vs Scribe 3 (n={np.sum(y == 3)}), type-B pages: "
          f"balanced accuracy {observed:.3f}, 95% interval [{low:.3f}, {high:.3f}]; shuffled {null.mean():.3f} "
          f"[{np.quantile(null, .025):.3f}, {np.quantile(null, .975):.3f}]; p={(1 + np.sum(null >= observed)) / 201:.3f}; "
          f"Scribe 2 right {per[2]:.2f}, Scribe 3 right {per[3]:.2f}")


def crossbar_or_text(k, x, label):
    train = [i for i, r in enumerate(k) if r["scribe"] in ("1", "2")]
    y = np.array([int(k[i]["scribe"]) for i in train])
    group = np.array([bifolium(folio(k[i]["page"])) for i in train])
    folds = list(GroupKFold(n_splits=10).split(x[train], y, group))
    own = held_out(x[train], y, folds)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(x[train], y)
    parts = [f"held-out Scribe 1 called 1: {(own[y == 1] == 1).mean():.2f} (n={np.sum(y == 1)})",
             f"held-out Scribe 2 called 1: {(own[y == 2] == 1).mean():.2f} (n={np.sum(y == 2)})"]
    for scribe in ("3", "4", "5"):
        index = [i for i, r in enumerate(k) if r["scribe"] == scribe]
        called_one = (model.predict(x[index]) == 1).mean()
        parts.append(f"Scribe {scribe} called 1: {called_one:.2f} (n={len(index)})")
    print(f"[{label}] crossbar or text type, classifier trained on Scribes 1 and 2: " + "; ".join(parts))


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "dinov3_unet"
    k = letters(*stem_letter(stem))
    x = embeddings_for(stem, k)
    usable = np.flatnonzero(np.linalg.norm(x, axis=1) > 0)  # Hinge is all zero when cleaning left no contour
    empty = len(k) - len(usable)
    k, x = [k[i] for i in usable], x[usable]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    keep = [i for i, r in enumerate(k) if not r["faint"]]
    print(f"{stem}: {len(k)} letters ({empty} left out, nothing to measure), {len(k) - len(keep)} flagged faint")
    for label, rows, xs in (("all", k, x), ("without faint", [k[i] for i in keep], x[keep])):
        text_type_fixed(rows, xs, label)
        crossbar_or_text(rows, xs, label)


if __name__ == "__main__":
    main()
