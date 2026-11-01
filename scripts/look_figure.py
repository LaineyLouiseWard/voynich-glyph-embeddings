"""What the average k of each scribe looks like and which parts of it the model uses, in one figure.

Top row: the average k of Scribes 1, 2 and 3 (share of letters with ink at each pixel) and Scribe 1 minus Scribes 2
and 3, from outputs/letter_shapes/average_k_all.npy (scripts/average_k.py --all). Bottom row: the occlusion maps of
outputs/letter_shapes/occlusion_k_all.npz (scripts/occlusion_k.py --all) over the faint average: brown where covering a patch lowers
the model's probability for the letter's Davis scribe, teal where it raises it. Arrows mark the strongest brown point
of Scribes 1 and 3 and the strongest teal point of Scribe 2 inside the crop, each label placed where the average has
no ink. Counts: n_1..3 from the .npz, else every non-faint k in data/all_k.csv. No model or GPU.
Output: outputs/letter_shapes/look.png.
Usage: python scripts/look_figure.py
"""
import matplotlib
import numpy as np
from matplotlib.patches import Patch

from average_k import FRAMES
from classifier_test import ROOT
from davis_timm_tests import letters
from letters_map import COLOURS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIBES = ("1", "2", "3")
X0, X1, Y0, Y1 = 33, 143, 22, 142  # crop: every pixel of each average with ink share above 0.02 lies inside
# (scribe, label, "max" = strongest brown or "min" = strongest teal)
NOTES = [("1", "uses the\ncrossbar join", "max"), ("2", "covering\nraises\nprobability", "min"), ("3", "uses\nthe loop", "max")]


def free_spot(ax, average, target, label):
    """Centre (x, y) nearest the target, at least 25 px from it, where the label's box covers no ink (share < 0.05)."""
    text = ax.text(0, 0, label, fontsize=8, ha="center", va="center", linespacing=1.0)
    box = text.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
    text.remove()
    half_w, half_h = abs(box.width) / 2 + 2, abs(box.height) / 2 + 2
    best = None
    for y in np.arange(Y0 + half_h, Y1 - half_h):
        for x in np.arange(X0 + half_w, X1 - half_w):
            inside = average[int(y - half_h):int(y + half_h) + 1, int(x - half_w):int(x + half_w) + 1]
            distance = np.hypot(x - target[1], y - target[0])
            if inside.max() < 0.05 and distance >= 25 and (best is None or distance < best[0]):
                best = (distance, x, y)
    return best[1], best[2]


def main():
    means = np.load(ROOT / "outputs/letter_shapes/average_k_all.npy")  # Scribe 1, Scribe 2, Scribe 3, Scribes 2 and 3
    saved = np.load(ROOT / "outputs/letter_shapes/occlusion_k_all.npz")
    every = [r for r in letters(True) if not r["faint"]]
    counts = {s: int(saved[f"n_{s}"]) if f"n_{s}" in saved else sum(r["scribe"] == s for r in every) for s in SCRIBES}
    importance = {s: saved[f"importance_{s}"] for s in SCRIBES}
    limit = max(np.abs(importance[s]).max() for s in SCRIBES)
    colour = {s: tuple(v / 255 for v in COLOURS[s]) for s in SCRIBES}
    brbg = plt.get_cmap("BrBG_r")

    fig, axes = plt.subplots(2, 4, figsize=(8.4, 5.2))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.02, wspace=0.08, hspace=0.22)
    for c, s in enumerate(SCRIBES):
        average = means[c]
        axes[0, c].imshow(average, cmap="Greys", vmin=0, vmax=1)
        axes[0, c].set_title(f"Scribe {s} (n={counts[s]:,})", fontsize=10)
        axes[1, c].imshow(average, cmap="Greys", vmin=0, vmax=1, alpha=0.35)
        axes[1, c].imshow(importance[s], cmap=brbg, vmin=-limit, vmax=limit, alpha=0.8)
    diff = means[0] - means[3]
    blue_orange = LinearSegmentedColormap.from_list("blue_orange", [FRAMES["Scribe 2"], "white", FRAMES["Scribe 1"]])
    diff_limit = np.abs(diff).max()
    axes[0, 3].imshow(diff, cmap=blue_orange, vmin=-diff_limit, vmax=diff_limit)
    axes[0, 3].set_title("Scribe 1 minus 2 and 3", fontsize=10)
    axes[0, 3].legend(handles=[Patch(color=FRAMES["Scribe 1"], label="more often inked\nby Scribe 1"),
                               Patch(color=FRAMES["Scribe 2"], label="more often inked\nby Scribes 2 and 3")],
                      loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=9, frameon=False,
                      handlelength=1.0, handleheight=1.0, borderaxespad=0.2)
    axes[0, 0].set_ylabel("Average k", fontsize=11)
    axes[1, 0].set_ylabel("Regions the classifier relies on", fontsize=11)

    for ax in axes[:, :3].ravel().tolist() + [axes[0, 3]]:
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y1, Y0)
        ax.set_xticks([])
        ax.set_yticks([])
    for c, s in enumerate(SCRIBES):
        for ax in axes[:, c]:
            for spine in ax.spines.values():
                spine.set_color(colour[s])
                spine.set_linewidth(3)
    for spine in axes[0, 3].spines.values():
        spine.set_color("0.7")

    key = axes[1, 3]
    key.axis("off")
    key.legend(handles=[Patch(color=brbg(0.85), label="covering lowers\nthe probability\n(the classifier\nrelies on\nthis region)"),
                        Patch(color=brbg(0.15), label="covering raises\nthe probability")],
               loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=9, frameon=False,
               handlelength=1.2, handleheight=1.2, labelspacing=1.2)

    fig.canvas.draw()
    for s, label, pick in NOTES:
        c = SCRIBES.index(s)
        ax = axes[1, c]
        window = importance[s][Y0:Y1, X0:X1]
        flat = window.argmax() if pick == "max" else window.argmin()
        row, col = np.unravel_index(flat, window.shape)
        target = (row + Y0, col + X0)
        x, y = free_spot(ax, means[c], target, label)
        ax.annotate(label, xy=(target[1], target[0]), xytext=(x, y), fontsize=8, ha="center", va="center",
                    linespacing=1.0, arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkA=4, shrinkB=1))
        print(f"Scribe {s}: {pick} importance {importance[s][target]:+.3f} at row {target[0]}, column {target[1]}; "
              f"label at ({x:.0f}, {y:.0f})")
    fig.savefig(ROOT / "outputs/letter_shapes/look.png", dpi=200)


if __name__ == "__main__":
    main()
