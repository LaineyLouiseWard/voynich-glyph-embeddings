"""Raw SBB 2022-08-16 ink masks for every upright k, cut to Timm's box and polygon.

Runs in the separate SBB environment (requirements-sbb.txt). SBB sees a 512 x 512 native-resolution
region around each box, kept inside the page. Outputs, named by annotation ID:
data/crops/sbb2022_raw/ (0 = ink, 255 = background, outside the polygon set to background) and
data/crops/box_rgb/ (the colour box crop, for review).
--workers N runs N copies of SBB side by side, each on its own pages; the masks are the same.
--letter iin does the same for Timm's `iin` boxes, into sbb2022_raw_iin/ and box_rgb_iin/ (no tilted list).
Usage: python scripts/sbb2022_masks.py [--limit N] [--workers N] [--letter k|iin]
"""
import argparse
import csv
import importlib.util
import json
import subprocess
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from archetype_db import glyph_table
from preprocess import polygon_top_left

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/models/sbb_binarization/sbb_binarize/sbb_binarize.py"
MODELS = ROOT / "data/models/sbb_2022"
RAW, RGB = ROOT / "data/crops/sbb2022_raw", ROOT / "data/crops/box_rgb"
REGION = 512
LETTER_BATCH, TILE_BATCH = 16, 16  # letters whose tiles go to SBB together; tiles per predict batch


def run_batched(engine, images):
    """engine.run on several images with all their tiles in one predict call; SBB's own tiling and stitching."""
    assert len(engine.models) == 1
    model, n_classes = engine.models[0][0], engine.models[0][3]
    predict, tiles = model.predict, []
    model.predict = lambda x, verbose=0: tiles.append(x) or np.zeros(x.shape[:3] + (n_classes,), np.float32)
    for image in images:
        engine.run(image=image)
    outputs = iter(np.split(predict(np.concatenate(tiles), batch_size=TILE_BATCH, verbose=0), len(tiles)))
    model.predict = lambda x, verbose=0: next(outputs)
    inks = [engine.run(image=image) for image in images]
    model.predict = predict
    return inks


def binarise_pages(pages, raw, rgb):
    spec = importlib.util.spec_from_file_location("sbb_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tf.config.experimental.enable_tensor_float_32_execution(False)
    engine = module.SbbBinarizer(str(MODELS))
    total, done = sum(len(page_rows) for _, page_rows in pages), 0
    for path, page_rows in pages:
        with Image.open(ROOT / "data/raw/digipal_project/images" / path) as page:
            page = page.convert("RGB")
            width, height = page.size
            jobs = []
            for r, points in page_rows:
                box = tuple(int(r[k]) for k in ("x0", "y0", "x1", "y1"))
                x = min(max((box[0] + box[2]) // 2 - REGION // 2, 0), width - REGION)
                y = min(max((box[1] + box[3]) // 2 - REGION // 2, 0), height - REGION)
                region = np.asarray(page.crop((x, y, x + REGION, y + REGION)))
                jobs.append((r, points, box, x, y, cv2.cvtColor(region, cv2.COLOR_RGB2BGR)))
            inks = [ink for start in range(0, len(jobs), LETTER_BATCH)
                    for ink in run_batched(engine, [j[5] for j in jobs[start:start + LETTER_BATCH]])]
            for (r, points, box, x, y, _), ink in zip(jobs, inks):
                ink = ink[box[1] - y:box[3] - y, box[0] - x:box[2] - x].astype(np.uint8)
                if points is not None:
                    inside = Image.new("L", (box[2] - box[0], box[3] - box[1]))
                    ImageDraw.Draw(inside).polygon([(px - box[0], py - box[1]) for px, py in points], fill=255)
                    ink[np.asarray(inside) == 0] = 255
                Image.fromarray(ink).save(raw / f"{r['annotation_id']}.png")
                page.crop(box).save(rgb / f"{r['annotation_id']}.png")
                done += 1
        print(f"{done}/{total}", flush=True)
    engine.end_session()
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--letter", default="k", choices=("k", "iin"))
    args = parser.parse_args()
    raw, rgb = (RAW, RGB) if args.letter == "k" else (Path(f"{RAW}_{args.letter}"), Path(f"{RGB}_{args.letter}"))
    dropped = ({r["annotation_id"] for r in csv.DictReader((ROOT / "config/dropped_tilted_k.csv").open())}
               if args.letter == "k" else set())
    rows = [r for r in csv.DictReader((ROOT / "data/glyphs.csv").open())
            if r["allograph"] == args.letter and r["annotation_id"] not in dropped]
    rows = rows[:args.limit] if args.limit else rows
    assert len({r["annotation_id"] for r in rows}) == len(rows), "repeated annotation IDs would share one output file"
    annotations = {r["annotation_id"]: r for r in map(lambda a: {k: str(v) for k, v in a.items()}, glyph_table())}
    raw.mkdir(parents=True, exist_ok=True)
    rgb.mkdir(parents=True, exist_ok=True)
    by_page = {}
    for r in rows:
        a = annotations[r["annotation_id"]]
        by_page.setdefault(r["iipimage"], []).append((r, polygon_top_left(a["geo_json"], int(a["height"]))))
    pages = list(by_page.items())
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            done = sum(pool.starmap(binarise_pages, [(pages[i::args.workers], raw, rgb)
                                                     for i in range(args.workers)]))
    else:
        done = binarise_pages(pages, raw, rgb)
    commit = subprocess.run(["git", "-C", str(SOURCE.parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (raw / "settings.json").write_text(json.dumps(dict(
        checkpoint="2022-08-16", weights="https://huggingface.co/SBB/sbb_binarization",
        code="https://github.com/qurator-spk/sbb_binarization", code_commit=commit,
        region=REGION, glyphs=done, polygon="outside set to background"), indent=2) + "\n")


if __name__ == "__main__":
    main()
