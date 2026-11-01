"""Find `k` inside transcribed words with a detector trained on Timm's `k` boxes.

Inputs: word boxes placed on Timm's images (data/word_boxes/aligned.csv, kept pages only) and Timm's boxes
(data/glyphs.csv). Each word is cut with half a word height of padding on every side, at native scan resolution. The
padding is left as scanned: greying out neighbouring `k`-words would cover part of the target `k` for 79 of 344 Timm `k`.

Training, folds grouped by folio (letters of one folio never in both): positives are words holding exactly one
EVA `k` with a Timm `k` centre inside, the box being Timm's; negatives are words holding no `k` whose padded crop touches
no word holding `k`, half of them with another tall letter (`t`, `p`, `f`). Detector: torchvision Faster R-CNN ResNet-50 FPN v2, COCO weights.
At use, a detection counts only if its centre lies inside the unpadded word box. Crops stay at native scale on a
paper-coloured canvas just large enough for their batch (multiples of 32 px); training batches group crops of similar
size, and each training image samples 128 candidate boxes.

Measure on held-out folios: each Timm `k` against the best detection in its word (IoU >= 0.5 and >= 0.75), the
same without keeping at most the EVA count per word, the same with the detection moved onto Timm's centre (size
errors only), and per `k`-word the number of detections
above the score threshold against the EVA count.
Outputs: outputs/finding/fold<N>/ (detections.csv, timm_match.csv, summary.txt, gallery.png).
Usage: python scripts/find_k.py [--fold 0] [--quick]
"""
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupKFold
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/digipal_project/images"
OUT = ROOT / "outputs/finding"
PAD = 0.5                     # padding on every side, in word heights
CANVAS = (1024, 384)          # width, height at native resolution; larger crops are shrunk to fit
FOLDS, EPOCHS, BATCH, SEED = 5, 15, 4, 0
ROIS = 128                    # candidate boxes sampled per training image (torchvision's default is 512)
NEG_PER_POS = 3
THRESHOLD = 0.5


def folio(page):
    return int(re.search(r"\d+", page).group())


def n_k(text):
    return text.count("k") - text.count("ckh")


def load_words():
    words = [r for r in csv.DictReader((ROOT / "data/word_boxes/aligned.csv").open()) if r["kept"] == "1"]
    for w in words:
        for key in ("x0", "y0", "x1", "y1"):
            w[key] = float(w[key])
        w["n_k"] = int(w["n_k"])
    return words


def crop_word(page, word):
    """Padded crop (uint8, shrunk to fit the canvas if needed); returns it, the paper colour, origin and scale."""
    h = word["y1"] - word["y0"]
    x0, y0 = max(int(word["x0"] - PAD * h), 0), max(int(word["y0"] - PAD * h), 0)
    x1, y1 = min(int(word["x1"] + PAD * h), page.width), min(int(word["y1"] + PAD * h), page.height)
    crop = page.crop((x0, y0, x1, y1))
    paper = tuple(int(v) for v in np.median(np.asarray(crop).reshape(-1, 3), axis=0))
    scale = min(1.0, CANVAS[0] / crop.width, CANVAS[1] / crop.height)
    if scale < 1:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.BILINEAR)
    return np.asarray(crop), paper, (x0, y0), scale


def touches_k_word(word, page_words):
    """True if the padded crop of `word` overlaps any other word whose EVA holds `k`."""
    h = word["y1"] - word["y0"]
    c = (word["x0"] - PAD * h, word["y0"] - PAD * h, word["x1"] + PAD * h, word["y1"] + PAD * h)
    return any(o is not word and o["n_k"] > 0 and min(o["x1"], c[2]) > max(o["x0"], c[0])
               and min(o["y1"], c[3]) > max(o["y0"], c[1]) for o in page_words)


def canvas_size(crops):
    """Smallest (width, height), in multiples of 32 px, that holds every crop of a batch."""
    return tuple(-(-max(c.shape[i] for c in crops) // 32) * 32 for i in (1, 0))


def to_tensor(crop, paper, size):
    """The crop on a paper-coloured canvas of the given size, as a float tensor; built per batch to keep memory low."""
    canvas = np.empty((size[1], size[0], 3), np.uint8)
    canvas[:] = paper
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    return torch.from_numpy(canvas.astype(np.float32) / 255).permute(2, 0, 1)


def model(device):
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_batch_size_per_image=ROIS)
    m.transform.resize = lambda image, target=None: (image, target)  # keep native scale
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="test run: 40 positives, 1 epoch, 100 held-out words")
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = OUT / (f"fold{args.fold}" + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    words = load_words()
    by_image = defaultdict(list)
    for w in words:
        by_image[w["iipimage"]].append(w)
    timm = [g for g in csv.DictReader((ROOT / "data/glyphs.csv").open())
            if g["allograph"] == "k" and g["iipimage"] in by_image]
    for g in timm:
        for key in ("x0", "y0", "x1", "y1"):
            g[key] = float(g[key])
        cx, cy = (g["x0"] + g["x1"]) / 2, (g["y0"] + g["y1"]) / 2
        inside = [w for w in by_image[g["iipimage"]] if w["n_k"] > 0
                  and w["x0"] <= cx <= w["x1"] and w["y0"] <= cy <= w["y1"]]
        g["word"] = min(inside, key=lambda w: (w["x1"] - w["x0"]) * (w["y1"] - w["y0"])) if inside else None

    pages = sorted({w["page"] for w in words})
    groups = [folio(p) for p in pages]
    split = list(GroupKFold(n_splits=FOLDS).split(pages, groups=groups))[args.fold]
    test_pages = {pages[i] for i in split[1]}
    print(f"Fold {args.fold}: {len(pages) - len(test_pages)} training pages, {len(test_pages)} held-out pages")

    # training crops
    positives = [g for g in timm if g["word"] is not None and g["word"]["n_k"] == 1 and g["word"]["page"] not in test_pages]
    clean = [w for w in words if w["n_k"] == 0 and w["page"] not in test_pages
             and not touches_k_word(w, by_image[w["iipimage"]])]
    tall = [w for w in clean if re.search(r"[tpf]", w["text"])]
    plain = [w for w in clean if not re.search(r"[tpf]", w["text"])]
    if args.quick:
        positives = [positives[i] for i in sorted(rng.choice(len(positives), min(40, len(positives)), replace=False))]
    n_neg = NEG_PER_POS * len(positives)
    negatives = [tall[i] for i in rng.choice(len(tall), n_neg // 2, replace=False)] + \
                [plain[i] for i in rng.choice(len(plain), n_neg - n_neg // 2, replace=False)]
    jobs = defaultdict(list)
    for g in positives:
        jobs[g["iipimage"]].append(("pos", g["word"], g))
    for w in negatives:
        jobs[w["iipimage"]].append(("neg", w, None))
    train = []
    for path, items in jobs.items():
        with Image.open(IMAGES / path) as page:
            page = page.convert("RGB")
            for kind, w, g in items:
                crop, paper, (x0, y0), scale = crop_word(page, w)
                boxes = torch.zeros((0, 4))
                if kind == "pos":
                    boxes = torch.tensor([[(g["x0"] - x0) * scale, (g["y0"] - y0) * scale,
                                           (g["x1"] - x0) * scale, (g["y1"] - y0) * scale]])
                train.append((crop, paper, boxes))
    print(f"Training crops: {len(positives)} with a Timm k, {len(negatives)} without k "
          f"({n_neg // 2} with t, p or f); loaded in {time.perf_counter() - start_time:.0f} s", flush=True)

    net = model(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for epoch in range(1 if args.quick else EPOCHS):
        net.train()
        order = rng.permutation(len(train))
        batches = []
        for start in range(0, len(order), 8 * BATCH):
            chunk = sorted(order[start:start + 8 * BATCH], key=lambda i: train[i][0].shape[:2])
            batches += [chunk[i:i + BATCH] for i in range(0, len(chunk), BATCH)]
        total = 0.0
        for index in rng.permutation(len(batches)):
            batch = [train[i] for i in batches[index]]
            size = canvas_size([b[0] for b in batch])
            images = [to_tensor(b[0], b[1], size).to(device) for b in batch]
            targets = [dict(boxes=b[2].to(device), labels=torch.ones(len(b[2]), dtype=torch.int64, device=device))
                       for b in batch]
            loss = sum(net(images, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        print(f"epoch {epoch + 1}: loss {total / len(train):.3f}, at {time.perf_counter() - start_time:.0f} s", flush=True)

    # held-out k-words
    net.eval()
    detections, gallery = [], []
    test_words = [w for w in words if w["page"] in test_pages and w["n_k"] > 0]
    if args.quick:
        test_words = [test_words[i] for i in sorted(rng.choice(len(test_words), min(100, len(test_words)), replace=False))]
    found = {}
    with torch.inference_mode():
        for path in sorted({w["iipimage"] for w in test_words}):
            with Image.open(IMAGES / path) as page:
                page = page.convert("RGB")
                for w in [w for w in test_words if w["iipimage"] == path]:
                    crop, paper, (x0, y0), scale = crop_word(page, w)
                    pred = net([to_tensor(crop, paper, canvas_size([crop])).to(device)])[0]
                    kept = []
                    for box, score in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy()):
                        bx = (box[0] / scale + x0, box[1] / scale + y0, box[2] / scale + x0, box[3] / scale + y0)
                        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                        if w["x0"] <= cx <= w["x1"] and w["y0"] <= cy <= w["y1"]:
                            kept.append((float(score), bx))
                    found[id(w)] = kept
                    for rank, (score, bx) in enumerate(kept):
                        detections.append(dict(page=w["page"], word=w["index"], text=w["text"], n_k=w["n_k"],
                                               rank=rank, score=round(score, 4),
                                               x0=round(bx[0]), y0=round(bx[1]), x1=round(bx[2]), y1=round(bx[3])))
                    above = [d for d in kept if d[0] >= THRESHOLD][:w["n_k"]]
                    if above and len(gallery) < 2000:
                        gallery.append((path, above[0][1], w["text"]))

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    matches = []
    held = [g for g in timm if g["page"] in test_pages or (g["word"] is not None and g["word"]["page"] in test_pages)]
    for g in held:
        t = (g["x0"], g["y0"], g["x1"], g["y1"])
        every = [c for c in found.get(id(g["word"]), []) if c[0] >= THRESHOLD] if g["word"] is not None else []
        cands = every[:g["word"]["n_k"]] if every else []
        best = max(cands, key=lambda c: iou(t, c[1]), default=None)
        row = dict(annotation_id=g["annotation_id"], page=g["page"], in_k_word=g["word"] is not None,
                   iou=0.0, iou_centred=0.0, score=0.0,
                   iou_no_count_rule=round(max((iou(t, c[1]) for c in every), default=0.0), 3))
        if best is not None:
            b = best[1]
            w_, h_ = b[2] - b[0], b[3] - b[1]
            cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
            row.update(iou=round(iou(t, b), 3), score=round(best[0], 3),
                       iou_centred=round(iou(t, (cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2)), 3))
        matches.append(row)

    for name, rows in (("detections.csv", detections), ("timm_match.csv", matches)):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    ious = np.array([m["iou"] for m in matches])
    centred = np.array([m["iou_centred"] for m in matches])
    in_word = np.array([m["in_k_word"] for m in matches])
    agree = defaultdict(int)
    for w in test_words:
        n = len([d for d in found[id(w)] if d[0] >= THRESHOLD])
        agree["equal" if n == w["n_k"] else "fewer" if n < w["n_k"] else "more"] += 1
    lines = [f"Fold {args.fold}: {len(test_pages)} held-out pages, {len(held)} Timm k on them, "
             f"{in_word.sum()} with their centre in a k-word",
             f"Timm k found at IoU >= 0.5: {(ious >= 0.5).sum()} of {len(held)}; IoU >= 0.75: {(ious >= 0.75).sum()}",
             f"Without the at-most-EVA-count rule: IoU >= 0.5 {(np.array([m['iou_no_count_rule'] for m in matches]) >= 0.5).sum()}, "
             f">= 0.75 {(np.array([m['iou_no_count_rule'] for m in matches]) >= 0.75).sum()}",
             f"Same, detection moved onto Timm's centre: IoU >= 0.5 {(centred >= 0.5).sum()}, >= 0.75 {(centred >= 0.75).sum()}",
             f"Median IoU of found: {np.median(ious[ious > 0]) if (ious > 0).any() else 0:.3f}",
             f"Held-out k-words: {len(test_words)}; detections above {THRESHOLD} equal to EVA count: {agree['equal']}, "
             f"fewer: {agree['fewer']}, more: {agree['more']}",
             f"Detections above {THRESHOLD} kept (at most EVA count per word): "
             f"{sum(min(len([d for d in found[id(w)] if d[0] >= THRESHOLD]), w['n_k']) for w in test_words)}"]
    lines.append(f"Run time: {time.perf_counter() - start_time:.0f} s")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # gallery: a seeded random 48 found k on held-out pages, with 30 px context
    pick = [gallery[i] for i in sorted(rng.choice(len(gallery), min(48, len(gallery)), replace=False))]
    tiles = []
    for path, b, text in pick:
        with Image.open(IMAGES / path) as page:
            tile = page.convert("RGB").crop((int(b[0]) - 30, int(b[1]) - 30, int(b[2]) + 30, int(b[3]) + 30))
            ImageDraw.Draw(tile).rectangle((30, 30, 30 + b[2] - b[0], 30 + b[3] - b[1]), outline=(255, 0, 0), width=2)
            tile.thumbnail((140, 160))
            tiles.append(tile)
    sheet = Image.new("RGB", (8 * 145, 6 * 165), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 8) * 145, (i // 8) * 165))
    sheet.save(out / "gallery.png")


if __name__ == "__main__":
    main()
