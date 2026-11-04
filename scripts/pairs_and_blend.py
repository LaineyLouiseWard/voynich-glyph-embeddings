"""Pairs of scribes, and gap-or-blend between Scribes 1 and 2 on herbal pages (readings in notes/results.md).

Main version: SBB + U-Net crops, faint letters removed. Output: outputs/classifier_tests/pairs_and_blend.txt.
Gap or blend: each letter's held-out log-odds of Scribe 2, averaged per page. The noise of a page's average is the pooled
spread of single letters within their page divided by the square root of that page's number of letters.
Usage: python scripts/pairs_and_blend.py [stem]   (default dinov3_unet)
"""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

from classifier_test import (ROOT, bifolium, embeddings_for, folio, held_out, page_weights, score, sheet_interval,
                             shuffled_scores)
from davis_timm_tests import letters, stem_letter

SUFFIX = ""


def setup(rows):
    y = np.array([int(r["scribe"]) for r in rows])
    group = np.array([bifolium(folio(r["page"])) for r in rows])
    unit_key = list(zip(group, [r["hand"] for r in rows]))
    units = {u: i for i, u in enumerate(dict.fromkeys(unit_key))}
    member = np.array([units[u] for u in unit_key])
    unit_labels = np.array([y[member == i][0] for i in range(len(units))])
    return y, member, unit_labels, list(GroupKFold(n_splits=10).split(rows, y, group))


def pair(k, x, a, b):
    index = [i for i, r in enumerate(k) if r["scribe"] in (a, b)]
    rows = [k[i] for i in index]
    y, member, unit_labels, folds = setup(rows)
    weights = page_weights(rows)
    prediction = held_out(x[index], y, folds, weights)
    observed = score(y, prediction, weights)
    group = np.array([bifolium(folio(r["page"])) for r in rows])
    low, high = sheet_interval(x[index], y, weights, member, group, observed)
    null = shuffled_scores(x[index], unit_labels, member, folds, weights)
    return (f"Scribe {a} vs {b} (n={np.sum(y == int(a))}, {np.sum(y == int(b))}): {observed:.3f}, "
            f"95% interval [{low:.3f}, {high:.3f}]; shuffled {null.mean():.3f}; p={(1 + np.sum(null >= observed)) / 201:.3f}")


def blend(k, x):
    index = [i for i, r in enumerate(k) if r["scribe"] in ("1", "2") and folio(r["page"]) <= 66]
    rows = [k[i] for i in index]
    y, _, _, folds = setup(rows)
    p = np.clip(held_out(x[index], y, folds, page_weights(rows), proba=True)[:, 1], 1e-6, 1 - 1e-6)
    score = np.log(p / (1 - p))  # log-odds of Scribe 2
    pages = {}
    for r, s in zip(rows, score):
        pages.setdefault((r["page"], r["scribe"]), []).append(s)
    spread = [np.asarray(v) - np.mean(v) for v in pages.values() if len(v) > 1]
    letter_sd = np.sqrt(sum((d ** 2).sum() for d in spread) / sum(len(d) - 1 for d in spread))
    means = {key: np.mean(v) for key, v in pages.items()}
    page_sd = {key: letter_sd / np.sqrt(len(v)) for key, v in pages.items()}
    noise = np.median(list(page_sd.values()))
    one = np.array([m for (page, s), m in means.items() if s == "1"])
    two = np.array([m for (page, s), m in means.items() if s == "2"])
    low, high = np.median(one), np.median(two)
    between = [key for key, m in means.items() if low + 2 * page_sd[key] < m < high - 2 * page_sd[key]]
    lines = [f"herbal pages: Scribe 1 {len(one)}, Scribe 2 {len(two)}; "
             f"letters per page: median {np.median([len(v) for v in pages.values()]):.0f}",
             f"medians: Scribe 1 {low:.2f}, Scribe 2 {high:.2f}; SD of one letter within its page {letter_sd:.2f}, "
             f"of a typical page mean {noise:.2f}",
             f"gap between medians = {(high - low) / noise:.1f} typical page SDs",
             f"pages in between (more than 2 of their own SDs from both medians): {len(between)} "
             f"({', '.join(f'{p} S{s}' for p, s in sorted(between))})"]
    return lines


def main():
    global SUFFIX
    stem = sys.argv[1] if len(sys.argv) > 1 else "dinov3_unet"
    SUFFIX = ("" if stem == "dinov3_unet" else f"_{stem}") + ("_edge_kept" if os.environ.get("KEEP_EDGE_INK") else "")
    k = [r for r in letters(*stem_letter(stem)) if not r["faint"]]
    x = embeddings_for(stem, k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    lines = [f"{stem}, faint letters removed, {len(k)} letters"]
    lines += [pair(k, x, a, b) for a, b in (("1", "2"), ("1", "3"), ("2", "3"))]
    herbal = [i for i, r in enumerate(k) if folio(r["page"]) <= 66]
    lines.append("herbal pages only (folios 1-66): " + pair([k[i] for i in herbal], x[herbal], "1", "2"))
    lines += blend(k, x)
    text = "\n".join(lines)
    (ROOT / f"outputs/classifier_tests/pairs_and_blend{SUFFIX}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
