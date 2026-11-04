"""One table of every k for the analysis: Timm's upright k plus the k found by find_k.py.

Timm's letters keep his annotation_id and box. Found letters are detections scoring at least find_k.THRESHOLD and
among their word's top EVA-count detections, from every fold present (each page is found by the model that held it
out). A find is dropped as a repeat of Timm's letter when it overlaps one of his k boxes on that image by IoU >= 0.3
or its centre lies inside his box. Found letters get ids from 100001 up, in page, word and rank order; page, word_index and
rank also name their SBB files (scripts/sbb2022_found.py: <page>_<word>_<rank>).
Davis scribe: config/zl_page_labels.csv (`davis_hand`). 115r has no `davis_hand` (Davis: first twelve lines Scribe 2,
the rest Scribe 3), so its letters go by the line of their placed word (counted from 0, so lines 0-11 are Zandbergen's 1-12). Timm's letters take the line of the placed word
holding their centre; where his page name is not in Zandbergen's file (foldout images such as 67r) or the 115r line is
unknown, they keep his own label.
Output: data/all_k.csv
--letter iin builds data/all_iin.csv the same way from Timm's `iin` (no tilted list) and outputs/finding_iin/; its found
letters get ids from 200001 up.
Usage: python scripts/build_all_k.py [--letter k|iin]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import find_k as F

ROOT = Path(__file__).resolve().parents[1]
FIRST_ID = 100001
FIELDS = ["letter_id", "source", "page", "davis_scribe", "iipimage", "x0", "y0", "x1", "y1", "eva_word",
          "word_index", "rank", "line", "score", "fold"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", default="k", choices=("k", "iin"))
    L = parser.parse_args().letter
    hand = {r["page"]: r["davis_hand"] for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}

    def scribe(page, line, own=""):
        if page == "115r":
            return own if line == "" else "2" if int(line) < 12 else "3"
        return hand.get(page) or own

    words = [r for r in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open())]
    by_image = defaultdict(list)
    for w in words:
        by_image[w["iipimage"]].append(w)
    page_image = {w["page"]: w["iipimage"] for w in words}
    word_of = {(w["page"], w["index"]): w for w in words}

    dropped = ({r["annotation_id"] for r in csv.DictReader((ROOT / "config/dropped_tilted_k.csv").open())}
               if L == "k" else set())
    rows, timm_boxes = [], defaultdict(list)
    for g in csv.DictReader((ROOT / "data/glyphs.csv").open()):
        if g["allograph"] != L or g["annotation_id"] in dropped:
            continue
        box = tuple(float(g[k]) for k in ("x0", "y0", "x1", "y1"))
        timm_boxes[g["iipimage"]].append(box)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [w for w in by_image[g["iipimage"]] if w["kept"] == "1"
                  and (int(w["n_k"]) if L == "k" else F.COUNT[L](w["text"])) > 0
                  and float(w["x0"]) <= cx <= float(w["x1"]) and float(w["y0"]) <= cy <= float(w["y1"])]
        w = min(inside, key=lambda w: (float(w["x1"]) - float(w["x0"])) * (float(w["y1"]) - float(w["y0"])),
                default=None)
        line = w["line"] if w else ""
        rows.append(dict(letter_id=g["annotation_id"], source="timm", page=g["page"],
                         davis_scribe=scribe(g["page"], line, g["scribe"]),
                         iipimage=g["iipimage"], x0=g["x0"], y0=g["y0"], x1=g["x1"], y1=g["y1"],
                         eva_word=w["text"] if w else "", word_index=w["index"] if w else "", rank="", line=line,
                         score="", fold=""))

    found, repeats = [], 0
    for fold in range(F.FOLDS):
        path = F.out_dir(L) / f"fold{fold}" / "detections.csv"
        if not path.exists():
            continue
        for d in csv.DictReader(path.open()):
            if float(d["score"]) < F.THRESHOLD or int(d["rank"]) >= int(d[f"n_{L}"]):
                continue
            box = tuple(float(d[k]) for k in ("x0", "y0", "x1", "y1"))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            image = page_image[d["page"]]
            if any(iou(box, t) >= 0.3 or (t[0] <= cx <= t[2] and t[1] <= cy <= t[3]) for t in timm_boxes[image]):
                repeats += 1
                continue
            line = word_of[(d["page"], d["word"])]["line"]
            found.append(dict(source="found", page=d["page"], davis_scribe=scribe(d["page"], line), iipimage=image,
                              x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"], eva_word=d["text"],
                              word_index=d["word"], line=line, score=d["score"], fold=fold, rank=int(d["rank"])))
    found.sort(key=lambda r: (r["page"], int(r["word_index"]), r["rank"]))
    for i, r in enumerate(found):
        r["letter_id"] = str((FIRST_ID if L == "k" else 200001) + i)
    rows += found
    assert len({r["letter_id"] for r in rows}) == len(rows), "letter ids must be unique"
    with (ROOT / f"data/all_{L}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows) - len(found)} Timm {L}, {len(found)} found {L}, {repeats} finds dropped as repeats of Timm's letters")


if __name__ == "__main__":
    main()
