"""Target-letter segmenter for k: a small U-Net labels each pixel as this k's ink, other ink, or background.

Training letters are those where the rule in k_clean.py removed only detached pieces: kept ink = k, removed ink =
other. Composites add removed pieces from other letters, often slid until they touch the k. Five folds grouped by
folio: every letter is predicted by a model that never saw its folio, and composite tests use held-out folios only.
Inputs per pixel: ink darkness from the colour crop (2nd to 98th percentile stretch) and the SBB 2022 ink mask.
Outputs: data/crops/unet_k/ (0 = ink, 255 = background, box size), outputs/cleaning/unet_k/letters.csv, summary.json.
load_letters("iin") reads Timm's `iin` (sbb2022_raw_iin/, box_rgb_iin/, the rule in iin_clean.py in sbb2022_iin/); an
`iin` is easy when the rule split no piece, since its `i` strokes are separate pieces; letters larger than the canvas
are left out.
Usage: python scripts/k_segmenter.py
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data/crops"
OUT_MASKS, OUT = CROPS / "unet_k", ROOT / "outputs/cleaning/unet_k"
SIZE, FOLDS, COMPOSITES, EPOCHS, BATCH, SEED = 144, 5, 6000, 12, 32, 0
TOUCH_SHARE, PLAIN_SHARE, MIN_DONOR = 0.7, 0.2, 15
WEIGHTS = (0.2, 1.0, 1.0)  # background, target ink, other ink


def load_letters(letter="k"):
    pages = {r["annotation_id"]: r["page"] for r in csv.DictReader((ROOT / "data/glyphs.csv").open())}
    suffix = "" if letter == "k" else f"_{letter}"
    letters = []
    for path in sorted((CROPS / f"sbb2022_raw{suffix}").glob("*.png"), key=lambda p: int(p.stem)):
        grey = np.asarray(Image.open(CROPS / f"box_rgb{suffix}" / path.name).convert("L"), dtype=np.float32)
        low, high = np.percentile(grey, [2, 98])
        dark = np.clip((high - grey) / max(high - low, 1), 0, 1)
        ink = np.asarray(Image.open(path).convert("L")) < 128
        kept = np.asarray(Image.open(CROPS / f"sbb2022_{letter}" / path.name).convert("L")) < 128
        if letter != "k" and max(ink.shape) > SIZE:
            continue
        parts = label(ink, connectivity=2)
        shared = set(np.unique(parts[kept])) & set(np.unique(parts[ink & ~kept])) - {0}
        easy = kept.any() and not shared and (letter != "k" or label(kept, connectivity=2).max() == 1)
        page = pages[path.stem]
        letters.append(dict(id=path.stem, folio=int(re.search(r"\d+", page).group()), dark=dark, ink=ink,
                            kept=kept, easy=bool(easy)))
    return letters


def donors(letters):
    pieces = []
    for letter in letters:
        if not letter["easy"]:
            continue
        removed = label(letter["ink"] & ~letter["kept"], connectivity=2)
        h, w = removed.shape
        for r in regionprops(removed):
            if r.area < MIN_DONOR:
                continue
            t, l, b, rr = r.bbox
            mask = removed[t:b, l:rr] == r.label
            pieces.append(dict(mask=mask, dark=letter["dark"][t:b, l:rr] * mask,
                               centre=((t + b) / 2 / h, (l + rr) / 2 / w)))
    return pieces


def canvas(letter):
    h, w = letter["ink"].shape
    top, left = (SIZE - h) // 2, (SIZE - w) // 2
    dark, ink, target = (np.zeros((SIZE, SIZE), t) for t in (np.float32, bool, np.int64))
    dark[top:top + h, left:left + w] = letter["dark"]
    ink[top:top + h, left:left + w] = letter["ink"]
    target[top:top + h, left:left + w] = np.where(letter["kept"], 1, np.where(letter["ink"], 2, 0))
    return dark, ink, target, (top, left, h, w)


def paste(dark, ink, target, piece, rng, box):
    top, left, h, w = box
    ph, pw = piece["mask"].shape
    y = int(round(top + piece["centre"][0] * h - ph / 2))
    x = int(round(left + piece["centre"][1] * w - pw / 2))
    if rng.random() < TOUCH_SHARE:
        k = binary_dilation(target == 1, iterations=1)
        ky, kx = np.argwhere(target == 1).mean(axis=0)
        step = np.array([ky - (y + ph / 2), kx - (x + pw / 2)])
        step = step / (np.abs(step).max() + 1e-6)
        position = np.array([y, x], dtype=float)
        for _ in range(SIZE):
            cy, cx = np.round(position).astype(int)
            if not (0 <= cy and 0 <= cx and cy + ph <= SIZE and cx + pw <= SIZE):
                break
            window = (slice(cy, cy + ph), slice(cx, cx + pw))
            if (target[window] == 1)[piece["mask"]].any():
                break
            y, x = cy, cx
            if k[window][piece["mask"]].any():
                break
            position += step
    if not (0 <= y and 0 <= x and y + ph <= SIZE and x + pw <= SIZE):
        return
    window = (slice(y, y + ph), slice(x, x + pw))
    free = piece["mask"] & (target[window] != 1)
    dark[window] = np.where(free, np.maximum(dark[window], piece["dark"]), dark[window])
    ink[window] |= free
    target[window] = np.where(free, 2, target[window])


def composites(letters, pieces, count, rng):
    easy = [l for l in letters if l["easy"]]
    xs, ys = np.zeros((count, 2, SIZE, SIZE), np.float32), np.zeros((count, SIZE, SIZE), np.int64)
    for n in range(count):
        dark, ink, target, box = canvas(easy[rng.integers(len(easy))])
        if rng.random() >= PLAIN_SHARE:
            for _ in range(rng.integers(1, 3)):
                paste(dark, ink, target, pieces[rng.integers(len(pieces))], rng, box)
        shift = rng.integers(-4, 5, size=2)
        dark, ink, target = (np.roll(a, tuple(shift), axis=(0, 1)) for a in (dark, ink, target))
        xs[n] = (np.clip(dark * rng.uniform(0.7, 1.3), 0, 1), ink)
        ys[n] = target
    return torch.from_numpy(xs), torch.from_numpy(ys)


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8, base * 16]
        self.down = nn.ModuleList([Block(2, widths[0])] + [Block(a, b) for a, b in zip(widths, widths[1:])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths, widths[1:])])
        self.merge = nn.ModuleList([Block(2 * a, a) for a in widths[:-1]])
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(len(self.up))):
            x = self.merge[i](torch.cat([self.up[i](x), skips.pop()], dim=1))
        return self.head(x)


def train(x, y, device):
    torch.manual_seed(SEED)
    model = UNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(x) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=steps)
    weights = torch.tensor(WEIGHTS, device=device)
    generator = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x) - BATCH + 1, BATCH):
            index = order[start:start + BATCH]
            loss = F.cross_entropy(model(x[index].to(device)), y[index].to(device), weight=weights)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    return model.eval()


@torch.no_grad()
def predict(model, x, device):
    return torch.cat([F.softmax(model(x[i:i + 64].to(device)), dim=1).cpu() for i in range(0, len(x), 64)])


def composite_scores(prob, y):
    pred = prob.argmax(dim=1)
    true_k, pred_k, other = y == 1, pred == 1, y == 2
    inter = (true_k & pred_k).sum(dim=(1, 2)).float()
    union = (true_k | pred_k).sum(dim=(1, 2)).float()
    return dict(target_iou=float((inter / union.clamp(min=1)).mean()),
                k_ink_lost=float(((true_k & ~pred_k).sum() / true_k.sum())),
                other_ink_called_k=float(((other & pred_k).sum() / other.sum().clamp(min=1))))


def main():
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    letters = load_letters()
    folds = list(GroupKFold(n_splits=FOLDS).split(letters, groups=[l["folio"] for l in letters]))
    OUT_MASKS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    for fold, (train_index, test_index) in enumerate(folds):
        train_letters = [letters[i] for i in train_index]
        test_letters = [letters[i] for i in test_index]
        x, y = composites(train_letters, donors(train_letters), COMPOSITES, rng)
        model = train(x, y, device)
        test_x, test_y = composites(test_letters, donors(test_letters), 500, rng)
        scores.append(dict(fold=fold, train_letters=len(train_letters),
                           train_easy=sum(l["easy"] for l in train_letters),
                           **composite_scores(predict(model, test_x, device), test_y)))
        real = [canvas(l) for l in test_letters]
        real_x = torch.from_numpy(np.stack([np.stack([d, i.astype(np.float32)]) for d, i, _, _ in real]))
        prob = predict(model, real_x, device)
        for letter, (_, ink, _, (top, left, h, w)), p in zip(test_letters, real, prob):
            pred = p.argmax(dim=0).numpy()[top:top + h, left:left + w]
            conf = p.max(dim=0).values.numpy()[top:top + h, left:left + w]
            k = pred == 1
            Image.fromarray(np.where(k, 0, 255).astype(np.uint8)).save(OUT_MASKS / f"{letter['id']}.png")
            on_ink = letter["ink"]
            rows.append(dict(annotation_id=letter["id"], fold=fold, easy=int(letter["easy"]),
                             k_pixels=int(k.sum()), k_outside_sbb_ink=int((k & ~on_ink).sum()),
                             unsure_ink_share=round(float((conf[on_ink] < 0.7).mean()) if on_ink.any() else 0, 4),
                             mean_confidence_k=round(float(conf[k].mean()) if k.any() else 0, 4)))
        print(scores[-1], flush=True)
    with (OUT / "letters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["annotation_id"])))
    (OUT / "summary.json").write_text(json.dumps(dict(
        folds=scores, settings=dict(size=SIZE, folds=FOLDS, composites_per_fold=COMPOSITES, epochs=EPOCHS,
                                    batch=BATCH, seed=SEED, touch_share=TOUCH_SHARE, plain_share=PLAIN_SHARE,
                                    class_weights=WEIGHTS, torch=torch.__version__, device=device)), indent=2) + "\n")


if __name__ == "__main__":
    main()
