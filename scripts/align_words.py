"""Place The Voynichese Project (2014) word boxes on Timm's page images, with no letter labels.

Each word-box page was drawn on a reference image 1,500 px high. Each page is placed only on Timm's image of the same
name (pages without one are skipped). Over a grid of scales, a filled-word-box template is slid over a text-ink map of
the image (normalised cross-correlation); the 10 best rough placements are refined by contrast (ink inside each word
box minus ink in a ring around it). Timm image x = s * X + tx, y = s * Y + ty, with (X, Y) in reference pixels.

Test (Timm's `iin` and `sh` only; his `k` reported apart): share of his boxes whose centre lies in a placed word whose
EVA holds that letter, at the fitted placement and at controls: plain image-size scaling, all words moved one line up
and down, and each word's text moved to the next word on its line.
A page is kept when its contrast is at least KEEP times the best placement elsewhere and its image is not a foldout
(wider than tall).
Outputs: data/word_boxes/aligned.csv, outputs/finding/alignment.txt, outputs/finding/overlays/<page>.png.
Usage: python scripts/align_words.py [--pages 10v 94v ...]
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from archetype_db import load_tables

ROOT = Path(__file__).resolve().parents[1]
WORDS = ROOT / "data/raw/voynichese"
IMAGES = ROOT / "data/raw/digipal_project/images"
OUT_CSV, OUT = ROOT / "data/word_boxes/aligned.csv", ROOT / "outputs/finding"
SCALES = np.exp(np.arange(np.log(2.0), np.log(3.6), 0.01))  # 1% steps
WORK = 0.5          # working pixels per reference pixel
INK_REDUCE = 4      # ink map stored at 1/4 of Timm's resolution
MARGIN = 20         # reference pixels around the word region
TOP = 10            # rough placements refined per page
RING = 0.4          # ring width around each word box, in box heights
KEEP = 1.5          # a page is kept when its contrast is at least this many times the runner-up
LETTERS = {"iin": lambda t: "iin" in t, "sh": lambda t: "sh" in t,
           "k": lambda t: t.count("k") - t.count("ckh") > 0}


def read_words(page):
    root = ET.parse(WORDS / f"f{page}.xml").getroot()
    words = [dict(index=int(w.get("index")), text=w.text or "",
                  box=np.array([int(w.get(a)) for a in ("x", "y", "width", "height")], float)) for w in root]
    words = [w for w in words if w["box"][2] > 0 and w["box"][3] > 0]
    return words, (int(root.get("width")), int(root.get("height")))


def lines_of(words):
    """Group words into lines: a new line starts when the next word starts left of the current one."""
    line, out = 0, []
    for prev, w in zip([None] + words[:-1], words):
        if prev is not None and w["box"][0] < prev["box"][0] - 20:
            line += 1
        out.append(line)
    return out


def ink_map(path):
    with Image.open(IMAGES / path) as im:
        im = im.convert("RGB")
        small = np.asarray(im.resize((im.width // INK_REDUCE, im.height // INK_REDUCE), Image.Resampling.BOX),
                           np.float32)
    grey = small.mean(axis=2)
    background = cv2.GaussianBlur(grey, (0, 0), 6)
    return np.clip(background - grey, 0, None)


def template(words):
    boxes = np.array([w["box"] for w in words])
    x0, y0 = boxes[:, 0].min() - MARGIN, boxes[:, 1].min() - MARGIN
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max() + MARGIN, (boxes[:, 1] + boxes[:, 3]).max() + MARGIN
    t = np.zeros((int((y1 - y0) * WORK) + 1, int((x1 - x0) * WORK) + 1), np.float32)
    for bx, by, bw, bh in boxes:
        t[int((by - y0) * WORK):int((by + bh - y0) * WORK) + 1, int((bx - x0) * WORK):int((bx + bw - x0) * WORK) + 1] = 1
    return t, (x0, y0)


def peaks(ncc, count, radius):
    """Up to `count` local maxima of a score map, at least `radius` apart."""
    ncc, out = ncc.copy(), []
    for _ in range(count):
        v, u = np.unravel_index(np.argmax(ncc), ncc.shape)
        out.append((float(ncc[v, u]), u, v))
        ncc[max(v - radius, 0):v + radius + 1, max(u - radius, 0):u + radius + 1] = -1
    return out


def coarse(words, inks):
    """Rough placements (score, image, s, tx, ty): the best few peaks per image and scale, top TOP overall."""
    t, (x0, y0) = template(words)
    radius = int(min(t.shape) / 4)
    found = []
    for path, ink in inks.items():
        for s in SCALES:
            f = WORK / s * INK_REDUCE
            work = cv2.resize(ink, (int(ink.shape[1] * f), int(ink.shape[0] * f)), interpolation=cv2.INTER_AREA)
            if work.shape[0] < t.shape[0] or work.shape[1] < t.shape[1]:
                continue
            ncc = cv2.matchTemplate(work, t, cv2.TM_CCOEFF_NORMED)
            for score, u, v in peaks(ncc, 3, radius):
                found.append((score, path, float(s), u * s / WORK - s * x0, v * s / WORK - s * y0))
    found.sort(key=lambda r: -r[0])
    kept = []
    for c in found:
        if all(c[1] != k[1] or abs(c[3] - k[3]) + abs(c[4] - k[4]) > 200 for k in kept):
            kept.append(c)
        if len(kept) == TOP:
            break
    return kept


def contrast(words, integral, std, s, tx, ty):
    """Median over words of (ink inside the box - ink in a ring around it), in ink standard deviations."""
    h, w = integral.shape[0] - 1, integral.shape[1] - 1

    def mean(x0, y0, x1, y1):
        x0, y0 = np.clip(x0, 0, w).astype(int), np.clip(y0, 0, h).astype(int)
        x1, y1 = np.clip(x1, 0, w).astype(int), np.clip(y1, 0, h).astype(int)
        area = np.maximum((x1 - x0) * (y1 - y0), 1)
        total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return total, area

    b = np.array([w_["box"] for w_ in words])
    x0, y0 = (s * b[:, 0] + tx) / INK_REDUCE, (s * b[:, 1] + ty) / INK_REDUCE
    x1, y1 = (s * (b[:, 0] + b[:, 2]) + tx) / INK_REDUCE, (s * (b[:, 1] + b[:, 3]) + ty) / INK_REDUCE
    m = RING * (y1 - y0)
    inside, a_in = mean(x0, y0, x1, y1)
    outer, a_out = mean(x0 - m, y0 - m, x1 + m, y1 + m)
    ring = (outer - inside) / np.maximum(a_out - a_in, 1)
    return float(np.median(inside / a_in - ring) / std)


def refine(words, candidate, ink):
    """Local search around a rough placement for the best contrast."""
    integral = cv2.integral(ink.astype(np.float64))
    std = float(ink.std())
    _, path, s0, tx0, ty0 = candidate
    best = (-np.inf, s0, tx0, ty0)
    for s in s0 * np.array([0.99, 0.995, 1.0, 1.005, 1.01]):
        for dx in range(-24, 25, 4):
            for dy in range(-24, 25, 4):
                c = contrast(words, integral, std, s, tx0 + dx, ty0 + dy)
                if c > best[0]:
                    best = (c, s, tx0 + dx, ty0 + dy)
    return best


def placed(words, s, tx, ty, dy=0.0, texts=None):
    out = []
    for i, w in enumerate(words):
        bx, by, bw, bh = w["box"]
        out.append(dict(text=w["text"] if texts is None else texts[i],
                        box=(s * bx + tx, s * (by + dy) + ty, s * (bx + bw) + tx, s * (by + bh + dy) + ty)))
    return out


def hits(glyphs, boxes):
    """Per letter: number of Timm boxes whose centre is in a placed word holding that letter, and total."""
    counts = defaultdict(lambda: [0, 0])
    for g in glyphs:
        cx, cy = (g["x0"] + g["x1"]) / 2, (g["y0"] + g["y1"]) / 2
        ok = any(b["box"][0] <= cx <= b["box"][2] and b["box"][1] <= cy <= b["box"][3] and LETTERS[g["allograph"]](b["text"])
                 for b in boxes)
        counts[g["allograph"]][0] += ok
        counts[g["allograph"]][1] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="*")
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    images = {r["locus"]: r for r in load_tables({"digipal_image"})["digipal_image"]}
    glyphs = defaultdict(list)
    for g in csv.DictReader((ROOT / "data/glyphs.csv").open()):
        if g["allograph"] in LETTERS:
            glyphs[g["iipimage"]].append({**g, **{k: float(g[k]) for k in ("x0", "y0", "x1", "y1")}})
    pages = args.pages or sorted(p.stem[1:] for p in WORDS.glob("f*.xml"))
    inks, rows, report, skipped = {}, [], [], []
    placed_by_image = defaultdict(lambda: defaultdict(list))
    for page in pages:
        words, (ref_w, ref_h) = read_words(page)
        if page not in images:
            skipped.append(page)
            print(f"{page}: no Timm image of that name, skipped", flush=True)
            continue
        candidates = [images[page]["iipimage"]]
        for path in candidates:
            if path not in inks:
                inks[path] = ink_map(path)
        ranked = []
        for c in coarse(words, {p: inks[p] for p in candidates}):
            ranked.append((*refine(words, c, inks[c[1]]), c[1], c[0]))
        ranked.sort(key=lambda r: -r[0])
        if not ranked:
            skipped.append(page)
            print(f"{page}: word layout larger than the image at every scale, skipped", flush=True)
            continue
        score, s, tx, ty, path, rough = ranked[0]
        other = next((r[0] for r in ranked[1:] if r[4] != path or abs(r[2] - tx) + abs(r[3] - ty) > 40), np.nan)
        locus = next(loc for loc, r in images.items() if r["iipimage"] == path)
        foldout = int(images[locus]["width"]) > int(images[locus]["height"])
        kept = bool((np.isnan(other) or score >= KEEP * other) and not foldout)
        line = lines_of(words)
        line_tops = defaultdict(list)
        for w, l in zip(words, line):
            line_tops[l].append(w["box"][1] + w["box"][3] / 2)
        centres = sorted(np.median(v) for v in line_tops.values())
        spacing = float(np.median(np.diff(centres))) if len(centres) > 1 else float(np.median([w["box"][3] for w in words]))
        nxt = [words[i + 1]["text"] if i + 1 < len(words) and line[i + 1] == line[i] else "" for i in range(len(words))]
        timm = images[locus]
        variants = {
            "fitted": placed(words, s, tx, ty),
            "size ratio": [dict(text=w["text"], box=(w["box"][0] * int(timm["width"]) / ref_w,
                                                     w["box"][1] * int(timm["height"]) / ref_h,
                                                     (w["box"][0] + w["box"][2]) * int(timm["width"]) / ref_w,
                                                     (w["box"][1] + w["box"][3]) * int(timm["height"]) / ref_h))
                           for w in words] if page == locus else None,
            "line up": placed(words, s, tx, ty, dy=-spacing),
            "line down": placed(words, s, tx, ty, dy=spacing),
            "next word's text": placed(words, s, tx, ty, texts=nxt),
        }
        for name, boxes in variants.items():
            if boxes is not None and kept:
                placed_by_image[path][name].extend(boxes)
        report.append((page, locus, score, other, s, tx, ty, spacing * s, kept, sum(r_["text"].count("k") - r_["text"].count("ckh") for r_ in words)))
        for w, l, b in zip(words, line, variants["fitted"]):
            rows.append(dict(page=page, image=locus, iipimage=path, kept=int(kept), index=w["index"], line=l, text=w["text"],
                             n_k=w["text"].count("k") - w["text"].count("ckh"),
                             x0=round(b["box"][0], 1), y0=round(b["box"][1], 1),
                             x1=round(b["box"][2], 1), y1=round(b["box"][3], 1)))
        print(f"{page}: {locus} contrast {score:.2f} runner-up {other:.2f} scale {s:.2f} {'kept' if kept else 'dropped'}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["Word-box pages placed on Timm images", "",
             "page  image  contrast  runner-up  scale  line spacing (Timm px)  kept  EVA k"]
    lines += [f"{p}  {loc}  {sc:.2f}  {ru:.2f}  {s:.2f}  {sp:.0f}  {int(kp)}  {nk}" for p, loc, sc, ru, s, tx, ty, sp, kp, nk in report]
    lines += ["", f"Pages kept: {sum(r[8] for r in report)} of {len(report)} placed; EVA k on kept pages: "
              f"{sum(r[9] for r in report if r[8])} of {sum(r[9] for r in report)}",
              f"Skipped, no Timm image of that name or no placement ({len(skipped)}): {' '.join(skipped)}"]
    lines += ["", "Timm boxes on images with a kept page whose centre lies in a placed word holding that letter",
              "(all Timm boxes on those images in the denominator)", ""]
    for name in ("fitted", "size ratio", "line up", "line down", "next word's text"):
        total = defaultdict(lambda: [0, 0])
        for path, by_name in placed_by_image.items():
            if name not in by_name:
                continue
            for letter, (a, n) in hits(glyphs[path], by_name[name]).items():
                total[letter][0] += a
                total[letter][1] += n
        lines.append(name + ": " + ", ".join(f"{l} {a}/{n}" for l, (a, n) in sorted(total.items())))
    lines += ["", "Per image, fitted (iin, sh, k):"]
    for path, by_name in placed_by_image.items():
        h = hits(glyphs[path], by_name["fitted"])
        loc = next(loc for loc, r in images.items() if r["iipimage"] == path)
        lines.append(f"{loc}: " + ", ".join(f"{l} {h[l][0]}/{h[l][1]}" for l in ("iin", "sh", "k") if l in h))
    (OUT / "alignment.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[len(report) + 4:]))

    overlays = OUT / "overlays"
    overlays.mkdir(exist_ok=True)
    from PIL import ImageDraw
    for path, by_name in placed_by_image.items():
        with Image.open(IMAGES / path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            for b in by_name["fitted"]:
                draw.rectangle(b["box"], outline=(0, 0, 255), width=4)
            for g in glyphs[path]:
                draw.rectangle((g["x0"], g["y0"], g["x1"], g["y1"]),
                               outline=(255, 0, 0) if g["allograph"] == "k" else (0, 160, 0), width=5)
            loc = next(loc for loc, r in images.items() if r["iipimage"] == path)
            im.resize((im.width // 4, im.height // 4)).save(overlays / f"{loc}.png")


if __name__ == "__main__":
    main()
