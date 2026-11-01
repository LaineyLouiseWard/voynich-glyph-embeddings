"""Review sheets for the k cleaning: colour crop beside the cleaned mask, removed ink in red.

Ordered by annotation ID, labelled by ID only, so the review is blind to scribe and page.
Usage: python scripts/k_review_sheets.py [first_sheet last_sheet] [--ids config/k_clean_dev_ids.csv] [--name dev] [--kept unet_k]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW, KEPT, RGB = (ROOT / "data/crops" / d for d in ("sbb2022_raw", "sbb2022_k", "box_rgb"))
OUT = ROOT / "outputs/cleaning/review"
TILE_H, PANEL_W, COLS, ROWS = 120, 76, 6, 6


def panel(image, width, height):
    scale = min(width / image.width, height / image.height)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def main():
    args = sys.argv[1:]
    name = args.pop(args.index("--name") + 1) if "--name" in args else "sheet"
    if "--name" in args:
        args.remove("--name")
    kept_dir = KEPT
    if "--kept" in args:
        kept_dir = ROOT / "data/crops" / args.pop(args.index("--kept") + 1)
        args.remove("--kept")
    ids = sorted((int(p.stem) for p in kept_dir.glob("*.png")))
    if "--ids" in args:
        with open(ROOT / args.pop(args.index("--ids") + 1)) as source:
            wanted = {int(line) for line in source.read().split()[1:]}
        args.remove("--ids")
        ids = [i for i in ids if i in wanted]
    per = COLS * ROWS
    sheets = range(0, (len(ids) + per - 1) // per)
    if len(args) == 2:
        sheets = range(int(args[0]), int(args[1]) + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in sheets:
        sheet = Image.new("RGB", (COLS * (2 * PANEL_W + 12), ROWS * (TILE_H + 16)), "white")
        draw = ImageDraw.Draw(sheet)
        for n, ident in enumerate(ids[s * per:(s + 1) * per]):
            raw = np.asarray(Image.open(RAW / f"{ident}.png").convert("L")) < 128
            kept = np.asarray(Image.open(kept_dir / f"{ident}.png").convert("L")) < 128
            view = np.full(raw.shape + (3,), 255, np.uint8)
            view[kept] = (0, 0, 0)
            view[raw & ~kept] = (230, 30, 30)
            left, top = (n % COLS) * (2 * PANEL_W + 12), (n // COLS) * (TILE_H + 16)
            sheet.paste(panel(Image.open(RGB / f"{ident}.png").convert("RGB"), PANEL_W, TILE_H), (left, top))
            sheet.paste(panel(Image.fromarray(view), PANEL_W, TILE_H), (left + PANEL_W + 2, top))
            draw.text((left + 2, top + TILE_H + 2), str(ident), fill="black")
        sheet.save(OUT / f"{name}_{s:02d}.png")
    print(f"{len(ids)} glyphs, sheets {sheets.start}-{sheets.stop - 1}")


if __name__ == "__main__":
    main()
