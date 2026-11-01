"""Family tree of the letters, after Denby 2020 (GRL, Fig. 2): Ward hierarchical clustering cut into 12 families.

Every k the analysis uses (Scribes 1-5, faint removed), as L2-normalised DINOv3 embeddings (dinov3_unet_all). The top
panel is the tree of the last 11 joins; under each of the 12 families sit the share of its letters from each scribe
and 8 of its letters drawn at random (seed 0), framed by scribe in the colours of letters_map.py.
Outputs: outputs/unsupervised_classification/letter_dendrogram.png, outputs/unsupervised_classification/letter_dendrogram.txt (scribe counts per family).
Usage: python scripts/letter_dendrogram.py
"""
import matplotlib
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

from classifier_test import ROOT, embeddings_for
from davis_timm_tests import letters
from letters_map import COLOURS, tile

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES, EXAMPLES, TILE = 12, 8, 46


def main():
    k = [r for r in letters(True) if not r["faint"]]
    x = embeddings_for("dinov3_unet_all", k)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    tree = linkage(x, method="ward")
    nodes = {}
    stack = [to_tree(tree)]
    while stack:
        node = stack.pop()
        nodes[node.id] = node
        stack += [c for c in (node.left, node.right) if c is not None]

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(3, 1, height_ratios=(2, 0.6, EXAMPLES * 0.55), hspace=0.05)
    top, bar, examples = (fig.add_subplot(g) for g in grid)
    leaves = dendrogram(tree, truncate_mode="lastp", p=FAMILIES, ax=top, color_threshold=0, above_threshold_color="k",
                        show_leaf_counts=True, leaf_font_size=9)["leaves"]
    top.set_yticks([])
    for side in ("top", "right", "left"):
        top.spines[side].set_visible(False)

    rng = np.random.default_rng(0)
    montage = Image.new("RGB", (FAMILIES * TILE * 10 // 10, EXAMPLES * TILE), "white")
    lines = ["family (left to right): letters; Scribe 1, 2, 3, 4, 5"]
    for n, leaf in enumerate(leaves):
        members = np.array(nodes[leaf].pre_order())
        counts = [int(sum(k[i]["scribe"] == s for i in members)) for s in COLOURS]
        lines.append(f"{n + 1}: {len(members)}; " + ", ".join(f"{c} ({100 * c / len(members):.0f}%)" for c in counts))
        bottom = 0.0
        for s, c in zip(COLOURS, counts):
            bar.bar(10 * n + 5, c / len(members), bottom=bottom, width=8, color=tuple(v / 255 for v in COLOURS[s]))
            bottom += c / len(members)
        for row, i in enumerate(rng.choice(members, min(EXAMPLES, len(members)), replace=False)):
            montage.paste(Image.fromarray(tile(k[i]["annotation_id"], k[i]["scribe"])).resize((TILE - 4, TILE - 4)),
                          (n * TILE + 2, row * TILE + 2))
    bar.set_xlim(top.get_xlim())
    bar.set_ylim(0, 1)
    bar.set_axis_off()
    examples.imshow(np.asarray(montage), extent=(0, 10 * FAMILIES, 0, 10 * EXAMPLES * 10 / 10), aspect="auto")
    examples.set_xlim(top.get_xlim())
    examples.set_axis_off()
    for s in COLOURS:
        bar.bar(0, 0, color=tuple(v / 255 for v in COLOURS[s]), label=f"Scribe {s}")
    fig.legend(*bar.get_legend_handles_labels(), frameon=False, loc="upper right", fontsize=9)
    fig.savefig(ROOT / "outputs/unsupervised_classification/letter_dendrogram.png", dpi=100, bbox_inches="tight")
    (ROOT / "outputs/unsupervised_classification/letter_dendrogram.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
