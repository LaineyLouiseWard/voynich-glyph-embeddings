"""Held-out scribe classifier on frozen embeddings.

Logistic regression, 10 folds grouped by sheet of parchment (bifolium, from Zandbergen's file), trained and scored with every page weighted equally
within its scribe and scribes weighted equally. 95% interval: each scribe's sheets resampled, classifier refitted.
Null: scribe labels shuffled between the same units, 200 times.
Usage: python scripts/classifier_test.py dinov3_unet --scribes 1,2,3 [--no-faint]
--no-faint drops letters whose U-Net k ink share is in the bottom 10% of the 425 masks.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]


def folio(page):
    return int(re.search(r"\d+", page).group())


SHEET = {folio(r["page"]): r["quire"] + r["bifolium"]
         for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open()) if re.search(r"\d", r["page"])}


def bifolium(n):
    """Sheet of parchment holding folio n: quire and bifolium number from Zandbergen's file ($Q, $B)."""
    return SHEET[n]


def faint_ids():
    from PIL import Image
    paths = sorted((ROOT / "data/crops/unet_k").glob("*.png"))
    ink = np.array([(np.asarray(Image.open(p).convert("L")) < 128).mean() for p in paths])
    return {p.stem for p, share in zip(paths, ink) if share < np.percentile(ink, 10)}


def load(scribes, no_faint=False):
    rows = list(csv.DictReader((ROOT / "data/glyphs.csv").open()))
    dropped = {r["annotation_id"] for r in csv.DictReader((ROOT / "config/dropped_tilted_k.csv").open())}
    if no_faint:
        dropped |= faint_ids()
    keep = [i for i, r in enumerate(rows) if r["allograph"] == "k"
            and r["annotation_id"] not in dropped and r["scribe"] in scribes]
    k = [rows[i] for i in keep]
    group = np.array([bifolium(folio(r["page"])) for r in k])
    y = np.array([int(r["scribe"]) for r in k])
    unit_key = list(zip(group, [r["hand"] for r in k]))
    units = {u: i for i, u in enumerate(dict.fromkeys(unit_key))}
    member = np.array([units[u] for u in unit_key])
    unit_labels = np.array([y[member == i][0] for i in range(len(units))])
    folds = list(GroupKFold(n_splits=10).split(np.zeros(len(y)), y, group))
    return keep, k, y, member, unit_labels, folds


def embeddings_for(stem, k):
    """Embeddings of the given k rows, matched by annotation ID (works for full and k-only files)."""
    with (ROOT / f"data/embeddings/{stem}_ids.csv").open() as source:
        position = {r["annotation_id"]: i for i, r in enumerate(csv.DictReader(source))}
    return np.load(ROOT / f"data/embeddings/{stem}.npy")[[position[r["annotation_id"]] for r in k]]


def page_weights(rows):
    """1 / letters on the same page and scribe, so each page counts once within its scribe."""
    key = [(r["page"], r["scribe"]) for r in rows]
    n = Counter(key)
    return np.array([1 / n[k] for k in key])


def score(y, prediction, weights):
    return balanced_accuracy_score(y, prediction, sample_weight=weights)


def recall(y, prediction, weights, scribe):
    """Page-weighted share of one scribe's letters called that scribe."""
    mine = y == scribe
    return weights[mine & (prediction == scribe)].sum() / weights[mine].sum()


def sheet_interval(x, y, weights, member, group, observed, n=1000, seed=0):
    """95% interval of the score: each scribe's (sheet, hand) units drawn with replacement and the classifier refitted
    in 10 folds (fewer if it holds fewer than 10 sheets) that keep every copy of a sheet together. A resample holds only about 65% of the sheets, so it trains on
    less and scores lower; the interval is shifted so the resamples' median sits on the observed score."""
    rng = np.random.default_rng(seed)
    letters_of = [np.flatnonzero(member == u) for u in range(member.max() + 1)]
    units_of = [np.unique(member[y == s]) for s in np.unique(y)]
    draws = [np.concatenate([letters_of[u] for units in units_of for u in rng.choice(units, len(units))])
             for _ in range(n)]
    scores = Parallel(n_jobs=-1)(delayed(_resample_score)(x, y, weights, group, index) for index in draws)
    return np.quantile(scores, [0.025, 0.975]) + observed - np.median(scores)


def _resample_score(x, y, weights, group, index):
    """One resample's score; folds = 10, or the number of sheets drawn when fewer (every copy of a sheet kept together)."""
    folds = list(GroupKFold(n_splits=min(10, len(np.unique(group[index])))).split(index, y[index], group[index]))
    return score(y[index], held_out(x[index], y[index], folds, weights[index]), weights[index])


def shuffled_scores(x, unit_labels, member, folds, weights, n=200, seed=0):
    """Scores with the scribe labels shuffled between whole units, n times: the null for p."""
    rng = np.random.default_rng(seed)
    labels = [rng.permutation(unit_labels)[member] for _ in range(n)]
    return np.array(Parallel(n_jobs=-1)(delayed(_shuffled_score)(x, s, folds, weights) for s in labels))


def _shuffled_score(x, s, folds, weights):
    return score(s, held_out(x, s, folds, weights), weights)


def held_out(x, y, folds, weights=None, proba=False):
    """Held-out predictions. With weights, each training scribe gets equal total weight, spread by the given weights,
    and the total equals the summed weights (the number of pages for page weights), so more letters per page do not
    weaken the penalty."""
    out = np.zeros((len(y), len(np.unique(y)))) if proba else np.empty_like(y)
    for train, test in folds:
        if weights is None:
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(x[train], y[train])
        else:
            w = weights[train].copy()
            for s in np.unique(y[train]):
                w[y[train] == s] /= w[y[train] == s].sum()
            model = LogisticRegression(C=1.0, max_iter=2000).fit(x[train], y[train], sample_weight=w * weights[train].sum() / w.sum())
        out[test] = model.predict_proba(x[test]) if proba else model.predict(x[test])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="+")
    parser.add_argument("--scribes", default="1,2,3,4,5")
    parser.add_argument("--no-faint", action="store_true")
    args = parser.parse_args()
    scribes = args.scribes.split(",")
    keep, k, y, member, unit_labels, folds = load(scribes, args.no_faint)
    for stem in args.stems:
        x = embeddings_for(stem, k)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        weights = page_weights(k)
        observed = score(y, prediction := held_out(x, y, folds, weights), weights)
        group = np.array([bifolium(folio(r["page"])) for r in k])
        low, high = sheet_interval(x, y, weights, member, group, observed)
        null = shuffled_scores(x, unit_labels, member, folds, weights)
        print(f"{stem} scribes {args.scribes}: balanced accuracy {observed:.3f}, 95% interval [{low:.3f}, {high:.3f}]; "
              f"shuffled {null.mean():.3f} [{np.quantile(null, .025):.3f}, {np.quantile(null, .975):.3f}]; "
              f"p={(1 + np.sum(null >= observed)) / 201:.3f}")
        print("  per scribe:", " ".join(f"S{s}={recall(y, prediction, weights, int(s)):.2f} (n={np.sum(y == int(s))})"
                                        for s in scribes))


if __name__ == "__main__":
    main()
