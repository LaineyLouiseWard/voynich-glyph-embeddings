"""Contour-hinge feature (Bulacu & Schomaker 2007, IEEE TPAMI 29(4), section 4.2) for each cleaned k.

For every pixel on every inner and outer ink contour, the two legs to the contour pixels 5 steps before and after give
two angles in [0, 360); each is binned into 24 directions; unordered pairs give 24 * 25 / 2 = 300 bins. The histogram
is normalised to a probability distribution and square-rooted (Euclidean distance then matches Hellinger distance).
Reads data/crops/unet_k/, writes data/embeddings/hinge_unet.npy and hinge_unet_ids.csv. Needs OpenCV.
With --all: every letter in data/all_k.csv from data/crops/unet_all_k/, written as hinge_unet_all.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPSILON, DIRECTIONS = 5, 24
PAIRS = [(i, j) for i in range(DIRECTIONS) for j in range(i, DIRECTIONS)]
INDEX = {pair: n for n, pair in enumerate(PAIRS)}


def hinge(ink):
    counts = np.zeros(len(PAIRS))
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour[:, 0, :].astype(float)
        if len(points) < 2 * EPSILON + 1:
            continue
        ahead = np.roll(points, -EPSILON, axis=0) - points
        behind = np.roll(points, EPSILON, axis=0) - points
        a1 = np.degrees(np.arctan2(-ahead[:, 1], ahead[:, 0])) % 360
        a2 = np.degrees(np.arctan2(-behind[:, 1], behind[:, 0])) % 360
        b1 = (a1 // (360 / DIRECTIONS)).astype(int) % DIRECTIONS
        b2 = (a2 // (360 / DIRECTIONS)).astype(int) % DIRECTIONS
        for i, j in zip(np.minimum(b1, b2), np.maximum(b1, b2)):
            counts[INDEX[(i, j)]] += 1
    return np.sqrt(counts / max(counts.sum(), 1))


def main():
    every = "--all" in sys.argv
    table, key, crops, stem = (("data/all_k.csv", "letter_id", "unet_all_k", "hinge_unet_all") if every else
                               ("data/glyphs.csv", "annotation_id", "unet_k", "hinge_unet"))
    with (ROOT / table).open() as source:
        ids = [r[key] for r in csv.DictReader(source) if (ROOT / "data/crops" / crops / f"{r[key]}.png").exists()]
    from PIL import Image
    features = np.stack([hinge(np.asarray(Image.open(ROOT / "data/crops" / crops / f"{i}.png").convert("L")) < 128)
                         for i in ids]).astype(np.float32)
    out = ROOT / "data/embeddings"
    np.save(out / f"{stem}.npy", features)
    with (out / f"{stem}_ids.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id"])
        writer.writerows([i] for i in ids)
    print(features.shape, "bins used:", int((features > 0).any(axis=0).sum()))


if __name__ == "__main__":
    main()
