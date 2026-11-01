"""Embed crops with frozen DINO tokens or pixel PCA. Variant unet_all takes its letters from data/all_k.csv."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]


def dino_embeddings(ids, model_name, variant, device="cpu"):
    torch.manual_seed(0)
    model_id = {"dinov2": "facebook/dinov2-base",
                "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "dinov3patch": "facebook/dinov3-vitb16-pretrain-lvd1689m"}[model_name]
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    model.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            crops = []
            for annotation_id in ids[start:start + 32]:
                path = ROOT / "data/crops" / variant / f"{annotation_id}.png"
                with Image.open(path) as image:
                    assert image.size == (224, 224), path
                    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                crop = torch.from_numpy(pixels).permute(2, 0, 1)
                crops.append((crop - mean) / std)
            tokens = model(pixel_values=torch.stack(crops).to(device)).last_hidden_state
            if model_name == "dinov3patch":
                patches = tokens[:, 1 + model.config.num_register_tokens:]
                grid = 224 // model.config.patch_size
                patches = patches.transpose(1, 2).reshape(len(crops), -1, grid, grid)
                features = torch.nn.functional.adaptive_avg_pool2d(patches, (4, 4)).flatten(1)
            else:
                features = tokens[:, 0]
            batches.append(torch.nn.functional.normalize(features, dim=1).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def pixel_embeddings(ids, variant):
    crops = []
    for annotation_id in ids:
        path = ROOT / "data/crops" / variant / f"{annotation_id}.png"
        with Image.open(path) as image:
            pixels = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
            crops.append(np.asarray(pixels, dtype=np.float32).ravel())
    scaled = StandardScaler().fit_transform(np.stack(crops))
    reduced = PCA(n_components=50, random_state=0).fit_transform(scaled)
    return normalize(reduced).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("dinov2", "dinov3", "dinov3patch", "pixel"))
    parser.add_argument("variant", choices=("grey", "binary", "clean", "unet", "sbb2022", "unet_all"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    table, key = ("data/all_k.csv", "letter_id") if args.variant == "unet_all" else ("data/glyphs.csv", "annotation_id")
    with (ROOT / table).open(newline="", encoding="utf-8") as source:
        ids = [row[key] for row in csv.DictReader(source)
               if (ROOT / "data/crops" / args.variant / f"{row[key]}.png").exists()]
    if args.model == "pixel":
        embeddings = pixel_embeddings(ids, args.variant)
    else:
        embeddings = dino_embeddings(ids, args.model, args.variant, args.device)
    output = ROOT / "data/embeddings"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.variant}"
    np.save(output / f"{stem}.npy", embeddings)
    with (output / f"{stem}_ids.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["annotation_id"])
        writer.writerows((annotation_id,) for annotation_id in ids)
    print(embeddings.shape)


if __name__ == "__main__":
    main()
