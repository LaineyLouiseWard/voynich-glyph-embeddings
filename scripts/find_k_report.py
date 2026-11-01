"""Summarise find_k.py folds: Timm's k found on held-out pages, and k found per Davis scribe.

Reads outputs/finding/fold<N>/{detections,timm_match}.csv for every fold present, the same page split as find_k.py,
the placed word boxes and config/zl_page_labels.csv (`davis_hand`). A k is found when a detection scores at least
find_k.THRESHOLD and is among the word's top EVA-count detections. Page 115r has no `davis_hand`: lines 0 to 11 count as
Scribe 2 (Davis: first twelve lines) and the rest as Scribe 3, using the line numbers from align_words.py.
Output: outputs/finding/report.txt
Usage: python scripts/find_k_report.py
"""
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupKFold

import find_k as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    words = F.load_words()
    pages = sorted({w["page"] for w in words})
    splits = list(GroupKFold(n_splits=F.FOLDS).split(pages, groups=[F.folio(p) for p in pages]))
    hand = {r["page"]: r["davis_hand"] for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}
    total_timm = sum(1 for g in csv.DictReader((ROOT / "data/glyphs.csv").open()) if g["allograph"] == "k")

    def scribe(page, line):
        if page == "115r":
            return "2" if int(line) < 12 else "3"
        return hand.get(page) or "?"

    lines = []
    eva, found = defaultdict(int), defaultdict(int)
    timm_rows, folds_done, pages_done = [], [], set()
    for fold, (_, test_index) in enumerate(splits):
        out = F.OUT / f"fold{fold}"
        if not (out / "timm_match.csv").exists():
            continue
        folds_done.append(fold)
        test_pages = {pages[i] for i in test_index}
        pages_done |= test_pages
        timm_rows += list(csv.DictReader((out / "timm_match.csv").open()))
        line_of = {(w["page"], w["index"]): w["line"] for w in words}
        for w in words:
            if w["page"] in test_pages and w["n_k"] > 0:
                eva[scribe(w["page"], w["line"])] += w["n_k"]
        for d in csv.DictReader((out / "detections.csv").open()):
            if float(d["score"]) >= F.THRESHOLD and int(d["rank"]) < int(d["n_k"]):
                found[scribe(d["page"], line_of[(d["page"], d["word"])])] += 1

    iou = [float(r["iou"]) for r in timm_rows]
    lines += [f"Folds done: {folds_done}; held-out pages: {len(pages_done)} of {len(pages)} kept pages", "",
              f"1. Timm's {total_timm} k: {len(timm_rows)} are on held-out pages. Found at IoU >= 0.5: "
              f"{sum(v >= 0.5 for v in iou)}; at IoU >= 0.75: {sum(v >= 0.75 for v in iou)}.", "",
              "2. Per Davis scribe, held-out pages: k found / k in the word-box transcription"]
    for s in sorted(set(eva) | set(found)):
        lines.append(f"   Scribe {s}: {found[s]} / {eva[s]}")
    lines.append(f"   All: {sum(found.values())} / {sum(eva.values())}")
    (F.OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
