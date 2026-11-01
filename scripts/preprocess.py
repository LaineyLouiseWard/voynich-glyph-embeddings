"""Polygon crops with shared contrast, padding and resize rules."""
import ast

import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.measure import label


def polygon_top_left(geo_json, height):
    """Convert Archetype's bottom-origin polygon to image coordinates."""
    points = ast.literal_eval(geo_json)["geometry"]["coordinates"][0]
    return [(x, height - (height + y if y < 0 else y)) for x, y in points]


def preprocess(page, pts, size=224, pad=0, binary=False, clean=False):
    xs, ys = np.asarray(pts).T
    box = (max(int(np.floor(min(xs))) - pad, 0),
           max(int(np.floor(min(ys))) - pad, 0),
           min(int(np.ceil(max(xs))) + pad, page.width),
           min(int(np.ceil(max(ys))) + pad, page.height))
    crop = np.asarray(page.crop(box).convert("L"), dtype=float)
    mask = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
    ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in pts], fill=1)
    inside = np.asarray(mask, dtype=bool)
    lo, hi = np.percentile(crop[inside], [2, 98])
    crop[~inside] = hi
    crop = (np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    if binary or clean:
        crop = np.where(crop > threshold_otsu(crop), 255, 0).astype(np.uint8)
    crop[~inside] = 255
    h, w = crop.shape
    if clean:
        components = label(crop == 0, connectivity=2)
        central = np.unique(components[h // 4:h - h // 4, w // 4:w - w // 4])
        central = central[central != 0]
        if len(central):
            areas = np.bincount(components.ravel())
            largest = areas[central].max()
            keep = central[areas[central] >= 0.15 * largest]
            crop = np.where(np.isin(components, keep), 0, 255).astype(np.uint8)
    side = max(h, w)
    square = np.full((side, side), 255, dtype=np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop
    resample = Image.Resampling.NEAREST if binary or clean else Image.Resampling.BICUBIC
    return Image.fromarray(square).resize((size, size), resample), box
