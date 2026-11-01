"""Crop every annotated glyph and write its metadata."""
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

from archetype_db import glyph_table, load_tables
from preprocess import polygon_top_left, preprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/digipal_project/images"
SCRIBES = {"HerbalA": 1, "Pharma": 1, "HerbalB": 2, "Balneology": 2,
           "Herbal - Scribe 3": 3, "Stars": 3, "Cosmological": 4, "Scribe 5": 5}
FIELDS = ["annotation_id", "image_id", "page", "allograph", "hand", "scribe",
          "x0", "y0", "x1", "y1", "box_w", "box_h", "iipimage"]


def main():
    Image.MAX_IMAGE_PIXELS = None
    hands = load_tables({"digipal_hand"})["digipal_hand"]
    assert {h["label"]: int(h["num"]) for h in hands} == SCRIBES
    rows = glyph_table()
    assert len(rows) == 1684
    assert len({r["annotation_id"] for r in rows}) == len(rows)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["iipimage"]].append(row)
    for variant in ("grey", "binary", "clean"):
        (ROOT / "data/crops" / variant).mkdir(parents=True, exist_ok=True)
    metadata = []
    for path, annotations in sorted(by_image.items()):
        with Image.open(IMAGES / path) as page:
            expected = (int(annotations[0]["width"]), int(annotations[0]["height"]))
            if page.size != expected:
                print(f"Size mismatch {annotations[0]['locus']}: database {expected}, image {page.size}; using database coordinates without scaling.")
            for row in annotations:
                pts = polygon_top_left(row["geo_json"], int(row["height"]))
                assert all(0 <= x <= page.width and 0 <= y <= page.height for x, y in pts), row["annotation_id"]
                for variant in ("grey", "binary", "clean"):
                    crop, box = preprocess(page, pts, binary=variant == "binary",
                                           clean=variant == "clean")
                    crop.save(ROOT / "data/crops" / variant / f"{row['annotation_id']}.png")
                x0, y0, x1, y1 = box
                metadata.append(dict(annotation_id=row["annotation_id"], image_id=row["image_id"],
                                     page=row["locus"], allograph=row["allograph"], hand=row["hand"],
                                     scribe=SCRIBES[row["hand"]], x0=x0, y0=y0, x1=x1, y1=y1,
                                     box_w=x1-x0, box_h=y1-y0, iipimage=path))
    with (ROOT / "data/glyphs.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(metadata, key=lambda r: int(r["annotation_id"])))
    print(f"Saved {len(metadata)} grey, binary and clean crops from {len(by_image)} page images.")


if __name__ == "__main__":
    main()
