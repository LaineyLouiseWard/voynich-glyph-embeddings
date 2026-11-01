"""Raw SBB 2022-08-16 ink masks for the k found by scripts/find_k.py, cut to the found box (no polygon).

Same SBB step as sbb2022_masks.py (512 x 512 region around each box). Reads outputs/finding/fold<N>/detections.csv and
keeps a detection when its score reaches find_k's threshold and its rank is below the word's EVA count of k.
Outputs, named <page>_<word>_<rank>: data/crops/sbb2022_found_raw/ (0 = ink, 255 = background) and
data/crops/box_rgb_found/ (colour box crop), plus settings_fold<N>.json.
CPU: environment voynich-sbb, --workers 4. GPU: environment voynich-sbb-gpu, one copy, with TF_USE_LEGACY_KERAS=1 and
LD_LIBRARY_PATH listing every site-packages/nvidia/*/lib folder (notes/sbb_speed.md).
--skip-done leaves out letters whose mask already exists (to finish a run that stopped).
Usage: python scripts/sbb2022_found.py --fold N [--workers N] [--limit N] [--skip-done]
"""
import argparse
import ast
import csv
import json
from multiprocessing import get_context
from pathlib import Path

from PIL import Image

from sbb2022_masks import REGION, binarise_pages

ROOT = Path(__file__).resolve().parents[1]
RAW, RGB = ROOT / "data/crops/sbb2022_found_raw", ROOT / "data/crops/box_rgb_found"
THRESHOLD = next(ast.literal_eval(line.split("=")[1].strip()) for line in
                 (ROOT / "scripts/find_k.py").read_text().splitlines() if line.startswith("THRESHOLD ="))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()
    images = {(w["page"], w["index"]): w["iipimage"]
              for w in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open())}
    rows = [d for d in csv.DictReader((ROOT / f"outputs/finding/fold{args.fold}/detections.csv").open())
            if float(d["score"]) >= THRESHOLD and int(d["rank"]) < int(d["n_k"])]
    rows = rows[:args.limit] if args.limit else rows
    by_page, sizes, ids = {}, {}, []
    for d in rows:
        path = images[(d["page"], d["word"])]
        if path not in sizes:
            with Image.open(ROOT / "data/raw/digipal_project/images" / path) as page:
                sizes[path] = page.size
        width, height = sizes[path]
        box = (max(int(d["x0"]), 0), max(int(d["y0"]), 0), min(int(d["x1"]), width), min(int(d["y1"]), height))
        r = dict(annotation_id=f"{d['page']}_{d['word']}_{d['rank']}", **dict(zip(("x0", "y0", "x1", "y1"), box)))
        ids.append(r["annotation_id"])
        if not (args.skip_done and (RAW / f"{r['annotation_id']}.png").exists()):
            by_page.setdefault(path, []).append((r, None))
    assert len(set(ids)) == len(ids), "repeated letter IDs would share one output file"
    RAW.mkdir(parents=True, exist_ok=True)
    RGB.mkdir(parents=True, exist_ok=True)
    pages = list(by_page.items())
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(binarise_pages, [(pages[i::args.workers], RAW, RGB) for i in range(args.workers)])
    else:
        binarise_pages(pages, RAW, RGB)
    done = sum((RAW / f"{i}.png").exists() for i in ids)
    (RAW / f"settings_fold{args.fold}.json").write_text(json.dumps(dict(
        checkpoint="2022-08-16", detections=f"outputs/finding/fold{args.fold}/detections.csv", threshold=THRESHOLD,
        region=REGION, letters=done, polygon="none, found box only"), indent=2) + "\n")


if __name__ == "__main__":
    main()
