"""Which parts of a k identify its scribe: occlusion maps for Scribes 1, 2 and 3 (main version, faint letters removed).

A classifier (logistic regression on DINOv3 CLS embeddings, 10 folds grouped by sheet of parchment) gives each letter
the probability of its own scribe from a model that never saw its sheet. A 32 x 32 white patch is slid over the
224 x 224 cleaned crop with stride 16; each occluded image is re-embedded, and importance = drop in that probability.
Maps are moved into the same aligned frame as scripts/average_k.py and averaged per scribe.
Outputs: outputs/letter_shapes/occlusion_k.png, outputs/letter_shapes/occlusion_k.npz. Needs a GPU for reasonable speed.
With --all: every k in data/all_k.csv (crops in data/crops/unet_all/, masks in unet_all_k/), outputs occlusion_k_all.*.
--sample N maps a seeded random N letters per scribe; the classifier still trains on every letter.
--replot redraws the figure from the saved .npz without rerunning (counts default to every non-faint letter).
Usage: python scripts/occlusion_k.py [n] [--all] [--sample N] [--replot]   (n = only the first n letters, for a trial)
"""
import argparse
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoModel

from average_k import HEIGHT, SIZE
from classifier_test import ROOT, bifolium, embeddings_for, folio
from davis_timm_tests import letters

PATCH, STRIDE, SCRIBES = 32, 16, ("1", "2", "3")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def embed(model, batch):
    """CLS embeddings of a normalised batch on the device: float16 on a GPU, redone in float32 if that overflows."""
    with torch.inference_mode():
        with torch.autocast(batch.device.type, torch.float16, enabled=batch.is_cuda):
            e = model(pixel_values=batch).last_hidden_state[:, 0].float()
        if not torch.isfinite(e).all():
            e = model(pixel_values=batch).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(e, dim=1).cpu().numpy()


def to_aligned(values, ink):
    """Move a box-frame array into the frame of average_k.aligned(): ink box, 100 px tall, ink centroid centred."""
    rows, cols = np.nonzero(ink)
    crop_v = values[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    crop_i = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)
    width = max(1, round(crop_i.shape[1] * HEIGHT / crop_i.shape[0]))
    resize = lambda a: np.asarray(Image.fromarray(a.astype(np.float32)).resize((width, HEIGHT), Image.Resampling.BILINEAR))
    canvas_v, canvas_i = np.zeros((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    top, left = (SIZE - HEIGHT) // 2, (SIZE - width) // 2
    canvas_v[top:top + HEIGHT, left:left + width] = resize(crop_v)
    canvas_i[top:top + HEIGHT, left:left + width] = resize(crop_i * 255) / 255
    cy, cx = center_of_mass(canvas_i)
    move = (SIZE / 2 - cy, SIZE / 2 - cx)
    return shift(canvas_v, move, order=1), shift(canvas_i, move, order=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    crops, masks, suffix = ("unet_all", "unet_all_k", "_all") if args.all else ("unet", "unet_k", "")
    if args.replot:
        saved = np.load(ROOT / f"outputs/letter_shapes/occlusion_k{suffix}.npz")
        every = [r for r in letters(args.all) if not r["faint"]]
        counts = {s: int(saved[f"n_{s}"]) if f"n_{s}" in saved else sum(r["scribe"] == s for r in every) for s in SCRIBES}
        draw({s: saved[f"average_{s}"] for s in SCRIBES}, {s: saved[f"importance_{s}"] for s in SCRIBES}, counts, suffix)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = [r for r in letters(args.all) if not r["faint"] and r["scribe"] in SCRIBES]
    x = embeddings_for("dinov3_unet_all" if args.all else "dinov3_unet", k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([int(r["scribe"]) for r in k])
    group = np.array([bifolium(folio(r["page"])) for r in k])
    fold_of = np.empty(len(k), int)
    models = []
    for n, (train, test) in enumerate(GroupKFold(n_splits=10).split(x, y, group)):
        models.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(x[train], y[train]))
        fold_of[test] = n
    dino = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(device).eval()
    starts = range(0, 224 - PATCH + 1, STRIDE)
    positions = [(t, l) for t in starts for l in starts]
    covered = torch.zeros(len(positions), 1, 224, 224, dtype=torch.bool, device=device)
    for n, (top, left) in enumerate(positions):
        covered[n, :, top:top + PATCH, left:left + PATCH] = True
    white = ((torch.ones(3, 1, 1) - MEAN) / STD).to(device)
    sums = {s: [np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE)), 0] for s in SCRIBES}
    check = []
    chosen = list(range(len(k)))
    if args.sample:
        rng = np.random.default_rng(0)
        chosen = sorted(i for s in SCRIBES for i in rng.choice(np.flatnonzero(y == int(s)), args.sample, replace=False))
    chosen = chosen[:args.n] if args.n else chosen
    for done, i in enumerate(chosen):
        r = k[i]
        image = np.asarray(Image.open(ROOT / "data/crops" / crops / f"{r['annotation_id']}.png").convert("RGB"),
                           dtype=np.float32) / 255
        # A patch over pixels that are already white leaves the image unchanged: its probability is the original's.
        change = [n for n, (top, left) in enumerate(positions) if (image[top:top + PATCH, left:left + PATCH] < 1).any()]
        original = ((torch.from_numpy(image).permute(2, 0, 1) - MEAN) / STD).to(device)
        e = embed(dino, torch.cat([original[None], torch.where(covered[change], white, original)]))
        check.append(float(e[0] @ x[i]))
        model = models[fold_of[i]]
        own = list(model.classes_).index(y[i])
        q = model.predict_proba(e)[:, own]
        p = np.full(len(positions) + 1, q[0])
        p[np.array(change, int) + 1] = q[1:]
        drop = np.zeros((224, 224))
        count = np.zeros((224, 224))
        for n, (top, left) in enumerate((t, l) for t in starts for l in starts):
            drop[top:top + PATCH, left:left + PATCH] += p[0] - p[n + 1]
            count[top:top + PATCH, left:left + PATCH] += 1
        drop /= np.maximum(count, 1)
        ink = np.asarray(Image.open(ROOT / "data/crops" / masks / f"{r['annotation_id']}.png").convert("L")) < 128
        h, w = ink.shape
        side = max(h, w)
        square = np.asarray(Image.fromarray(drop.astype(np.float32)).resize((side, side), Image.Resampling.BILINEAR))
        top0, left0 = (side - h) // 2, (side - w) // 2
        values, mask = to_aligned(square[top0:top0 + h, left0:left0 + w], ink)
        sums[r["scribe"]][0] += values
        sums[r["scribe"]][1] += mask
        sums[r["scribe"]][2] += 1
        if (done + 1) % 50 == 0:
            print(f"{done + 1}/{len(chosen)}", flush=True)
    print(f"GPU vs cached embedding cosine: min {min(check):.4f}, median {np.median(check):.4f}")
    if args.n:
        return
    importance = {s: v[0] / v[2] for s, v in sums.items()}
    average = {s: v[1] / v[2] for s, v in sums.items()}
    np.savez(ROOT / f"outputs/letter_shapes/occlusion_k{suffix}.npz", **{f"importance_{s}": importance[s] for s in SCRIBES},
             **{f"average_{s}": average[s] for s in SCRIBES}, **{f"n_{s}": sums[s][2] for s in SCRIBES})

    draw(average, importance, {s: sums[s][2] for s in SCRIBES}, suffix)


def draw(average, importance, counts, suffix):
    """Average k per scribe (top) and its occlusion map (bottom): brown where covering lowers the own-scribe
    probability, teal where it raises it; panels framed in the scribe colours of letters_map.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from letters_map import COLOURS
    limit = max(np.abs(importance[s]).max() for s in SCRIBES)
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6))
    for c, s in enumerate(SCRIBES):
        axes[0, c].imshow(average[s], cmap="Greys", vmin=0, vmax=1)
        axes[0, c].set_title(f"Scribe {s} (n={counts[s]})", fontsize=9)
        axes[1, c].imshow(average[s], cmap="Greys", vmin=0, vmax=1, alpha=0.35)
        axes[1, c].imshow(importance[s], cmap="BrBG_r", vmin=-limit, vmax=limit, alpha=0.8)
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(tuple(v / 255 for v in COLOURS[s]))
                spine.set_linewidth(3)
    for ax in axes.ravel():
        ax.set_xlim(30, 130)
        ax.set_ylim(140, 20)
    fig.tight_layout()
    fig.savefig(ROOT / f"outputs/letter_shapes/occlusion_k{suffix}.png", dpi=150)

if __name__ == "__main__":
    main()
