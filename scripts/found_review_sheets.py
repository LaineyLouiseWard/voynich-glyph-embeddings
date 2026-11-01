"""Blind review sheets for the cleaning of found letters: colour box beside the cleaned mask, removed ink in red.

360 found letters (seed 0) drawn at random from those the tests use (Scribes 1-3, faint removed), kept in random order
and labelled by review number only, so the review is blind to scribe and page. The key from review number to letter
is written to outputs/cleaning/review/found_key.csv; open it only after rating.
Outputs: outputs/cleaning/review/found_00.png ... found_09.png, outputs/cleaning/review/found_key.csv.
Usage: python scripts/found_review_sheets.py
"""
import csv

import numpy as np
from PIL import Image, ImageDraw

from classifier_test import ROOT
from davis_timm_tests import letters
from k_review_sheets import COLS, OUT, PANEL_W, ROWS, TILE_H, panel

N = 360
CROPS = ROOT / "data/crops"


def main():
    pool = [r for r in letters(True) if r["source"] == "found" and not r["faint"] and r["scribe"] in ("1", "2", "3")]
    chosen = [pool[i] for i in np.random.default_rng(0).choice(len(pool), N, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "found_key.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_number", "letter_id", "page", "scribe"])
        for n, r in enumerate(chosen, 1):
            writer.writerow([n, r["letter_id"], r["page"], r["scribe"]])
    per = COLS * ROWS
    for s in range(N // per):
        sheet = Image.new("RGB", (COLS * (2 * PANEL_W + 12), ROWS * (TILE_H + 16)), "white")
        draw = ImageDraw.Draw(sheet)
        for n, r in enumerate(chosen[s * per:(s + 1) * per]):
            name = f"{r['page']}_{r['word_index']}_{r['rank']}"
            raw = np.asarray(Image.open(CROPS / "sbb2022_found_raw" / f"{name}.png").convert("L")) < 128
            kept = np.asarray(Image.open(CROPS / "unet_all_k" / f"{r['letter_id']}.png").convert("L")) < 128
            view = np.full(raw.shape + (3,), 255, np.uint8)
            view[kept] = (0, 0, 0)
            view[raw & ~kept] = (230, 30, 30)
            left, top = (n % COLS) * (2 * PANEL_W + 12), (n // COLS) * (TILE_H + 16)
            sheet.paste(panel(Image.open(CROPS / "box_rgb_found" / f"{name}.png").convert("RGB"), PANEL_W, TILE_H), (left, top))
            sheet.paste(panel(Image.fromarray(view), PANEL_W, TILE_H), (left + PANEL_W + 2, top))
            draw.text((left + 2, top + TILE_H + 2), str(s * per + n + 1), fill="black")
        sheet.save(OUT / f"found_{s:02d}.png")
    print(f"{N} letters from {len(pool)}, {N // per} sheets")


if __name__ == "__main__":
    main()
