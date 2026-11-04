"""Blind review sheets for the `iin` cleaning (gate 2) and the `n` cut (gate 3).

clean: 360 found `iin` (seed 0) drawn at random from those the tests use (Scribes 1 to 3, faint removed), in random
order, labelled by review number only; 18 sheets of 20, colour box beside the cleaned mask (black kept, red removed).
cut: 50 cut letters (seed 0) from the same pool; 2 sheets of 25, the cleaned `iin` with the `n` in black, the rest grey,
and the cut column in blue.
Keys from review number to letter: outputs/cleaning/review_iin/<mode>_key.csv; open only after rating.
With a seed other than 0, files are named <mode>_s<seed>_*, and letters already in the seed-0 key of that mode are left out.
Usage: python scripts/iin_review_sheets.py clean|cut [seed]
"""
import csv
import sys

import numpy as np
from PIL import Image, ImageDraw

from classifier_test import ROOT
from davis_timm_tests import letters
from k_review_sheets import panel

CROPS, OUT = ROOT / "data/crops", ROOT / "outputs/cleaning/review_iin"
TILE_H, PANEL_W, GAP = 80, 110, 12


def main():
    mode = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    name_ = f"{mode}_s{seed}" if seed else mode
    pool = [r for r in letters(True, "iin") if r["source"] == "found" and not r["faint"] and r["scribe"] in ("1", "2", "3")]
    cut = {r["letter_id"]: int(r["cut"]) for r in csv.DictReader((ROOT / "outputs/cleaning/cut_n.csv").open())} \
        if mode == "cut" else {}
    if mode == "cut":
        pool = [r for r in pool if cut[r["letter_id"]] >= 0]
    if seed:
        seen = {r["letter_id"] for r in csv.DictReader((OUT / f"{mode}_key.csv").open())}
        pool = [r for r in pool if r["letter_id"] not in seen]
    total, cols, rows = (360, 4, 5) if mode == "clean" else (50, 5, 5)
    chosen = [pool[i] for i in np.random.default_rng(seed).choice(len(pool), total, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name_}_key.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_number", "letter_id", "page", "scribe"])
        for n, r in enumerate(chosen, 1):
            writer.writerow([n, r["letter_id"], r["page"], r["scribe"]])
    per = cols * rows
    for s in range(total // per):
        width = PANEL_W * (2 if mode == "clean" else 1) + GAP
        sheet = Image.new("RGB", (cols * width, rows * (TILE_H + 16)), "white")
        draw = ImageDraw.Draw(sheet)
        for n, r in enumerate(chosen[s * per:(s + 1) * per]):
            name = f"{r['page']}_{r['word_index']}_{r['rank']}"
            kept = np.asarray(Image.open(CROPS / "unet_box_iin" / f"{r['letter_id']}.png").convert("L")) < 128
            left, top = (n % cols) * width, (n // cols) * (TILE_H + 16)
            if mode == "clean":
                raw = np.asarray(Image.open(CROPS / "sbb2022_found_raw_iin" / f"{name}.png").convert("L")) < 128
                view = np.full(raw.shape + (3,), 255, np.uint8)
                view[kept] = (0, 0, 0)
                view[raw & ~kept] = (230, 30, 30)
                colour = Image.open(CROPS / "box_rgb_found_iin" / f"{name}.png").convert("RGB")
                sheet.paste(panel(colour, PANEL_W, TILE_H), (left, top))
                sheet.paste(panel(Image.fromarray(view), PANEL_W, TILE_H), (left + PANEL_W + 2, top))
            else:
                x = cut[r["letter_id"]]
                n_mask = np.asarray(Image.open(CROPS / "unet_box_n" / f"{r['letter_id']}.png").convert("L")) < 128
                view = np.full(kept.shape + (3,), 255, np.uint8)
                view[kept] = (170, 170, 170)
                start = kept.shape[1] - n_mask.shape[1]
                view[:, start:][n_mask] = (0, 0, 0)
                view[:, x] = np.where(view[:, x].sum(axis=1, keepdims=True) == 765, (60, 110, 255), view[:, x])
                sheet.paste(panel(Image.fromarray(view), PANEL_W, TILE_H), (left, top))
            draw.text((left + 2, top + TILE_H + 2), str(s * per + n + 1), fill="black")
        sheet.save(OUT / f"{name_}_{s:02d}.png")
    print(f"{total} letters from {len(pool)}, {total // per} sheets")


if __name__ == "__main__":
    main()
