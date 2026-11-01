"""Pad box-size black-and-white masks to a centred square on white and resize to 224 (nearest neighbour).

Usage: python scripts/square_crops.py unet_k unet   (reads data/crops/unet_k/, writes data/crops/unet/)
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    source, target = (ROOT / "data/crops" / name for name in sys.argv[1:3])
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.png"):
        with Image.open(path) as image:
            mask = image.convert("L")
        side = max(mask.size)
        square = Image.new("L", (side, side), 255)
        square.paste(mask, ((side - mask.width) // 2, (side - mask.height) // 2))
        square.resize((224, 224), Image.Resampling.NEAREST).save(target / path.name)


if __name__ == "__main__":
    main()
