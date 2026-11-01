"""The scribe tests in one chart: each comparison's held-out score with its 95% interval and p, for DINOv3 and the
contour-hinge cross-check, against the shuffled-label score.

Reads the DINOv3 results on every letter (faint removed): Scribe 1 vs 2, Scribe 1 vs 2 on herbal pages and Scribe 1 vs
3 from outputs/classifier_tests/pairs_and_blend_dinov3_unet_all.txt; Scribe 2 vs 3 on B pages from
outputs/classifier_tests/davis_timm_tests_dinov3_unet_all.txt ("[without faint] text type fixed"). Output: outputs/classifier_tests/scribe_tests.png.
Usage: python scripts/tests_figure.py
"""
import re

import matplotlib

from classifier_test import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMBER = r"([0-9.]+)"


def parse(text, start):
    line = next(l for l in text.splitlines() if l.startswith(start))
    score, low, high = (float(v) for v in re.search(
        rf"{NUMBER}, 95% interval \[{NUMBER}, {NUMBER}\]", line.replace("balanced accuracy ", "")).groups())
    shuffled = float(re.search(rf"shuffled {NUMBER}", line).group(1))
    p = float(re.search(rf"p={NUMBER}", line).group(1))
    return score, low, high, shuffled, p


def main():
    out = ROOT / "outputs/classifier_tests"
    rows = []
    for stem in ("dinov3_unet_all", "hinge_unet_all"):
        pairs = (out / f"pairs_and_blend_{stem}.txt").read_text()
        tests = (out / f"davis_timm_tests_{stem}.txt").read_text()
        rows.append([parse(pairs, "Scribe 1 vs 2"), parse(pairs, "herbal pages only"), parse(pairs, "Scribe 1 vs 3"),
                     parse(tests, "[without faint] text type fixed")])
    names = ["Scribe 1 vs 2", "Scribe 1 vs 2, herbal pages", "Scribe 1 vs 3", "Scribe 2 vs 3, Currier B pages"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for y, (dino, hinge) in enumerate(reversed(list(zip(rows[0], rows[1])))):
        for (score, low, high, shuffled, p), dy, face, width in ((dino, 0.13, "black", 2), (hinge, -0.13, "white", 1.2)):
            ax.plot([100 * low, 100 * high], [y + dy, y + dy], color="black", linewidth=width)
            ax.plot(100 * score, y + dy, "o", markerfacecolor=face, markeredgecolor="black", markersize=7)
        ax.plot(100 * dino[3], y, "|", color="0.55", markersize=22, markeredgewidth=2)
        ax.text(100 * max(dino[2], hinge[2]) + 1.5, y + 0.13, f"{100 * dino[0]:.1f}%, p = {dino[4]:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), list(reversed(names)))
    ax.axvline(50, color="0.7", linestyle=":", linewidth=1)
    ax.text(50.5, len(names) - 0.5, "50%: chance level", fontsize=7, color="0.45", va="top")
    ax.set_xlim(35, 100)
    ax.set_xlabel("correct scribe assignments on sheets left out of training\n(balanced accuracy, %)")
    ax.plot([], [], "o-", color="black", linewidth=2, markersize=7, label="DINOv3, with 95% interval")
    ax.plot([], [], "o-", color="black", markerfacecolor="white", linewidth=1.2, markersize=7,
            label="contour-hinge, with 95% interval")
    ax.plot([], [], "|", color="0.55", markersize=12, markeredgewidth=2, label="shuffled labels")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/classifier_tests/scribe_tests.png", dpi=150)


if __name__ == "__main__":
    main()
