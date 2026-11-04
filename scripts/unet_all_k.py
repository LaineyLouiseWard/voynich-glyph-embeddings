"""Clean every letter in data/all_k.csv with the k U-Net: keep this k's ink, drop neighbour ink.

The five fold models are trained exactly as in k_segmenter.py (same letters, folds, composites, seed and order) and
saved to data/models/unet_k/fold<N>.pt; saved models are loaded instead of retrained. Each letter is cleaned by the
model whose training never saw its folio: Timm's letters by the fold that held them out, found letters on a folio with
Timm letters by that folio's fold, others by fold = folio mod 5. Input per pixel, as before: darkness from the colour
box crop (2nd to 98th percentile stretch) and the SBB ink mask: Timm's letters from sbb2022_masks.py (his outlines
fill his boxes for all 425, so these are box-only too), found letters from sbb2022_found.py. Letters larger than the
canvas are shrunk to fit and the mask is enlarged back.
Outputs: data/crops/unet_all_k/<letter_id>.png (0 = ink, 255 = background, box size), outputs/cleaning/unet_all_k/letters.csv.
--letter iin: models trained on Timm's `iin` (k_segmenter.load_letters("iin")) and saved to data/models/unet_iin/, with
their scores on held-out composites printed; letters from data/all_iin.csv and the `_iin` SBB folders; outputs
data/crops/unet_box_iin/ and outputs/cleaning/unet_all_iin/.
Usage: python scripts/unet_all_k.py [--letter k|iin]
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold

import k_segmenter as K

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data/models/unet_k"
CROPS = ROOT / "data/crops"
OUT_MASKS, OUT = ROOT / "data/crops/unet_all_k", ROOT / "outputs/cleaning/unet_all_k"


def fold_models(device, letter="k"):
    """The five k_segmenter fold models and the folios each one held out."""
    rng = np.random.default_rng(K.SEED)
    letters = K.load_letters(letter)
    folds = list(GroupKFold(n_splits=K.FOLDS).split(letters, groups=[l["folio"] for l in letters]))
    models_dir = MODELS if letter == "k" else ROOT / f"data/models/unet_{letter}"
    models_dir.mkdir(parents=True, exist_ok=True)
    models, held_out = [], {}
    for fold, (train_index, test_index) in enumerate(folds):
        train_letters = [letters[i] for i in train_index]
        test_letters = [letters[i] for i in test_index]
        path = models_dir / f"fold{fold}.pt"
        # composites are drawn every time, so the random stream matches k_segmenter.py for every fold
        x, y = K.composites(train_letters, K.donors(train_letters), K.COMPOSITES, rng)
        trained = not path.exists()
        if not trained:
            model = K.UNet().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            model = K.train(x, y, device)
            torch.save(model.state_dict(), path)
        test_x, test_y = K.composites(test_letters, K.donors(test_letters), 500, rng)
        if trained:
            print(dict(fold=fold, train_letters=len(train_letters), train_easy=sum(l["easy"] for l in train_letters),
                       **K.composite_scores(K.predict(model, test_x, device), test_y)), flush=True)
        models.append(model)
        for letter in test_letters:
            held_out[letter["folio"]] = fold
        print(f"fold {fold} ready", flush=True)
    return models, held_out


def inputs(r, letter="k"):
    """Darkness and SBB ink for one all_k row: Timm's from sbb2022_masks.py, found from sbb2022_found.py."""
    suffix = "" if letter == "k" else f"_{letter}"
    if r["source"] == "timm":
        raw, rgb, name = CROPS / f"sbb2022_raw{suffix}", CROPS / f"box_rgb{suffix}", r["letter_id"]
    else:
        raw, rgb, name = (CROPS / f"sbb2022_found_raw{suffix}", CROPS / f"box_rgb_found{suffix}",
                          f"{r['page']}_{r['word_index']}_{r['rank']}")
    grey = np.asarray(Image.open(rgb / f"{name}.png").convert("L"), dtype=np.float32)
    low, high = np.percentile(grey, [2, 98])
    dark = np.clip((high - grey) / max(high - low, 1), 0, 1)
    ink = np.asarray(Image.open(raw / f"{name}.png").convert("L")) < 128
    return dark, ink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", default="k", choices=("k", "iin"))
    L = parser.parse_args().letter
    out_masks, out = (OUT_MASKS, OUT) if L == "k" else (CROPS / f"unet_box_{L}", ROOT / f"outputs/cleaning/unet_all_{L}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, held_out = fold_models(device, L)
    rows = list(csv.DictReader((ROOT / f"data/all_{L}.csv").open()))
    out_masks.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in rows:
        dark, ink = inputs(r, L)
        h, w = ink.shape
        scale = min(1.0, K.SIZE / max(h, w))
        if scale < 1:
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            dark_in = np.asarray(Image.fromarray(dark).resize(size, Image.Resampling.BILINEAR))
            ink_in = np.asarray(Image.fromarray(ink).resize(size, Image.Resampling.NEAREST))
        else:
            dark_in, ink_in = dark, ink
        d, i, _, (top, left, hh, ww) = K.canvas(dict(dark=dark_in, ink=ink_in, kept=np.zeros_like(ink_in)))
        folio = int(re.search(r"\d+", r["page"]).group())
        fold = held_out.get(folio, folio % K.FOLDS)
        x = torch.from_numpy(np.stack([d, i.astype(np.float32)])[None])
        p = K.predict(models[fold], x, device)[0]
        pred = p.argmax(dim=0).numpy()[top:top + hh, left:left + ww]
        conf = p.max(dim=0).values.numpy()[top:top + hh, left:left + ww]
        k = pred == 1
        if scale < 1:
            k = np.asarray(Image.fromarray(k).resize((w, h), Image.Resampling.NEAREST))
            conf = np.asarray(Image.fromarray(conf).resize((w, h), Image.Resampling.NEAREST))
        Image.fromarray(np.where(k, 0, 255).astype(np.uint8)).save(out_masks / f"{r['letter_id']}.png")
        out_rows.append(dict(letter_id=r["letter_id"], source=r["source"], fold=fold, shrunk=int(scale < 1),
                             box_pixels=h * w, k_pixels=int(k.sum()), k_outside_sbb_ink=int((k & ~ink).sum()),
                             unsure_ink_share=round(float((conf[ink] < 0.7).mean()) if ink.any() else 0, 4),
                             mean_confidence_k=round(float(conf[k].mean()) if k.any() else 0, 4)))
    with (out / "letters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{len(out_rows)} letters cleaned, {sum(r['shrunk'] for r in out_rows)} shrunk to fit")


if __name__ == "__main__":
    main()
