"""README figures: three pages of the book, how every `k` is found and one followed to its numbers.

pages.png: folios 2r (Scribe 1, Currier A, herbal), 31r (Scribe 2, Currier B, herbal) and 103r (Scribe 3, Currier B,
starred paragraphs), framed in their scribe colours.
pipeline.png: top row, word boxes placed on the letter's page (data/word_boxes/aligned.csv), the `k` the detector found
there (data/all_k.csv, source "found") and the first 12 cut out, the chosen letter framed thick; bottom row, that letter's
crops in data/crops/ (sbb2022_found_raw, unet_all_k, unet_all), with kept ink black and ink the U-Net removed as
neighbouring letters grey, and its DINOv3 embedding.
Outputs: outputs/pages.png, outputs/pipeline.png
Usage: python scripts/pipeline_figure.py [letter_id]   (default 102980 on page 31r; must be among the first 12 found
on its page)
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.path import Path as ArrowPath
from PIL import Image

from classifier_test import embeddings_for

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data/crops"
IMAGES = ROOT / "data/raw/digipal_project/images"
MAGENTA = "#e377c2"
CYAN = "#17becf"
SCRIBE_COLOUR = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}


def plain(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def arrows(fig, axes):
    for left, right in zip(axes[:-1], axes[1:]):
        a, b = left.get_position(), right.get_position()
        fig.text((a.x1 + b.x0) / 2, (a.y0 + a.y1) / 2, "→", ha="center", va="center", fontsize=18)


def cleaning_row(axes, row):
    """Ink only, Neighbours removed and Model input for one found k (its row in data/all_k.csv)."""
    letter = row["letter_id"]
    raw = np.asarray(Image.open(CROPS / "sbb2022_found_raw" / f"{row['page']}_{row['word_index']}_{row['rank']}.png")
                     .convert("L")) < 128
    kept = np.asarray(Image.open(CROPS / "unet_all_k" / f"{letter}.png").convert("L")) < 128
    edited = np.full(raw.shape + (3,), 255, np.uint8)
    edited[raw & ~kept] = (150, 150, 150)
    edited[kept] = 0
    panels = [("Black and white by the\nStaatsbibliothek zu\nBerlin model (SBB)", Image.fromarray(np.where(raw, 0, 255).astype(np.uint8))),
              ("U-Net removed\nneighbouring ink", Image.fromarray(edited)),
              ("Scaled square, DINOv3 input", Image.open(CROPS / "unet_all" / f"{letter}.png").convert("L"))]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        plain(ax, title)


def embedding_panel(ax, letter):
    vector = embeddings_for("dinov3_unet_all", [{"annotation_id": letter}])[0].reshape(24, 32)
    limit = np.abs(vector).max()
    ax.imshow(vector, cmap="BrBG_r", vmin=-limit, vmax=limit)
    plain(ax, "768 numbers from DINOv3")


def pages_figure(pages=("2r", "31r", "103r")):
    """Pages side by side at the same height, titled and framed in their scribe's colour."""
    labels = {r["page"]: r for r in csv.DictReader((ROOT / "config/zl_page_labels.csv").open())}
    scans = {}
    for r in csv.DictReader((ROOT / "data/all_k.csv").open()):
        scans.setdefault(r["page"], r["iipimage"])
    images = []
    for page_name in pages:
        with Image.open(IMAGES / scans[page_name]) as scan:
            images.append(scan.convert("RGB").resize((round(scan.width * 1200 / scan.height), 1200)))
    fig, axes = plt.subplots(1, len(pages), figsize=(13, 6),
                             gridspec_kw={"width_ratios": [page.width for page in images], "wspace": 0.2})
    for ax, page_name, page in zip(axes, pages, images):
        ax.imshow(page)
        scribe, language = labels[page_name]["davis_hand"], labels[page_name]["language"]
        plain(ax, f"Folio {page_name}: Davis's Scribe {scribe}, Currier {language}")
        for spine in ax.spines.values():
            spine.set_color(SCRIBE_COLOUR[scribe])
            spine.set_linewidth(4)
    fig.savefig(ROOT / "outputs/pages.png", dpi=100, bbox_inches="tight", facecolor="white")


def finding(axes, page_name, letter):
    """Word boxes on the page, the k found in them, and the first 12 found cut out, `letter` framed thick.
    Returns that letter's frame in the cut-out grid."""
    words = [r for r in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open())
             if r["page"] == page_name and r["kept"] == "1"]
    found = [r for r in csv.DictReader((ROOT / "data/all_k.csv").open())
             if r["page"] == page_name and r["source"] == "found"]
    box = lambda r: tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1"))
    with Image.open(IMAGES / words[0]["iipimage"]) as scan:
        page = scan.convert("RGB")
    axes[0].imshow(page)
    for r in words:
        a, b, c, d = box(r)
        axes[0].add_patch(Rectangle((a, b), c - a, d - b, fill=False, linewidth=0.6,
                                    edgecolor=CYAN if int(r["n_k"]) else "0.55"))
    plain(axes[0], "Word outlines from The Voynichese Project")

    top = min(box(r)[1] for r in found)
    region = [r for r in words if box(r)[1] >= top - 10 and box(r)[3] <= top + 420]
    rx0, ry0 = min(box(r)[0] for r in region) - 30, top - 40
    rx1, ry1 = max(box(r)[2] for r in region) + 30, top + 440
    axes[1].imshow(page.crop((int(rx0), int(ry0), int(rx1), int(ry1))))
    for r in region:
        if int(r["n_k"]):
            a, b, c, d = box(r)
            axes[1].add_patch(Rectangle((a - rx0, b - ry0), c - a, d - b, fill=False, linewidth=1, edgecolor=CYAN))
    for r in found:
        a, b, c, d = box(r)
        if b >= ry0 and d <= ry1:
            axes[1].add_patch(Rectangle((a - rx0, b - ry0), c - a, d - b, fill=False, linewidth=1.6, edgecolor=MAGENTA))
    plain(axes[1], "Each k detected (Faster R-CNN)")
    key = [Patch(fill=False, edgecolor=CYAN, linewidth=1.5, label="word whose transcription contains a k"),
           Patch(fill=False, edgecolor="0.55", linewidth=1.5, label="other word"),
           Patch(fill=False, edgecolor=MAGENTA, linewidth=1.5, label="k detected by Faster R-CNN")]
    axes[1].legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    tiles, frames = [], []
    for i, r in enumerate(found[:12]):
        a, b, c, d = (int(v) for v in box(r))
        tile = page.crop((a, b, c, d))
        tile.thumbnail((90, 120))
        canvas = Image.new("RGB", (100, 130), "white")
        ox, oy = (100 - tile.width) // 2, (130 - tile.height) // 2
        canvas.paste(tile, (ox, oy))
        tiles.append(np.asarray(canvas))
        frames.append((i % 4 * 100 + ox - 0.5, i // 4 * 130 + oy - 0.5, tile.width, tile.height))
    while len(tiles) % 4:
        tiles.append(np.full((130, 100, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    axes[2].imshow(grid)
    for r, (a, b, w, h) in zip(found, frames):
        axes[2].add_patch(Rectangle((a, b), w, h, fill=False, linewidth=3 if r["letter_id"] == letter else 1,
                                    edgecolor=MAGENTA))
    plain(axes[2], "First 12 detected k")
    return next(f for r, f in zip(found, frames) if r["letter_id"] == letter)


def pipeline_figure(letter="102980"):
    """Top row: how each k on the letter's page is found. Bottom row: that k cleaned and turned into 768 numbers."""
    row = next(r for r in csv.DictReader((ROOT / "data/all_k.csv").open()) if r["letter_id"] == letter)
    fig = plt.figure(figsize=(13, 8.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.5, 2.1], hspace=0.42)
    top = [fig.add_subplot(s) for s in grid[0].subgridspec(1, 3, wspace=0.25, width_ratios=[1, 1.6, 1])]
    bottom = [fig.add_subplot(s) for s in grid[1].subgridspec(1, 4, wspace=0.35, width_ratios=[0.75, 0.75, 1, 4 / 3])]
    a, b, w, h = finding(top, row["page"], letter)
    cleaning_row(bottom[:3], row)
    embedding_panel(bottom[3], letter)
    fig.canvas.draw()
    arrows(fig, top)
    arrows(fig, bottom)
    to_figure = fig.transFigure.inverted()
    x0, y0 = to_figure.transform(top[2].transData.transform((a + w / 2, b + h)))
    key_bottom = top[1].get_legend().get_window_extent().transformed(to_figure).y0
    title_top = bottom[0].title.get_window_extent().transformed(to_figure).y1
    level = (key_bottom + title_top) / 2
    x1 = (bottom[0].get_position().x0 + bottom[0].get_position().x1) / 2
    path = ArrowPath([(x0, y0), (x0, level), (x1, level), (x1, title_top + 0.005)])
    fig.add_artist(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=18, linewidth=1.5, color="black",
                                   transform=fig.transFigure))
    fig.savefig(ROOT / "outputs/pipeline.png", dpi=150, bbox_inches="tight", facecolor="white")


def main():
    pipeline_figure(sys.argv[1] if len(sys.argv) > 1 else "102980")
    pages_figure()


if __name__ == "__main__":
    main()
