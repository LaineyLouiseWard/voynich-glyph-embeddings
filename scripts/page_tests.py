"""Two page tests on a sheet held back from training (settings in notes/n_settings.md).

f115r: Davis (2020) gives its first twelve lines to Scribe 2 and the rest to Scribe 3 (build_all_k.py splits the page by
the line of each placed word). A classifier trained on Scribe 2 and 3 letters from every other sheet assigns each f115r
letter; the share assigned to Scribe 2 is reported for lines 1 to 12 and for the rest (letters with no placed line are left
out).
ff58r/58v: Currier A pages that Davis gives to Scribe 3. A classifier trained on Scribe 1 and 3 letters from every other
sheet; the share of their letters assigned to Scribe 3.
Faint letters removed; logistic regression with every page weighted equally within its scribe and the scribes weighted
equally (classifier_test.held_out).
Output: outputs/classifier_tests/page_tests_<stem>.txt
Usage: python scripts/page_tests.py dinov3_unet_all dinov3_unet_all_iin dinov3_unet_all_n
"""
import sys

import numpy as np

from classifier_test import ROOT, bifolium, embeddings_for, folio, held_out, page_weights
from davis_timm_tests import letters, stem_letter


def assign(k, x, scribes, target):
    """Classifier on `scribes` from every sheet but the target's; the target letters' predicted scribes."""
    assert target, "no letters on the target page"
    sheet = bifolium(folio(k[target[0]]["page"]))
    train = [i for i, r in enumerate(k) if r["scribe"] in scribes and bifolium(folio(r["page"])) != sheet]
    index = train + target
    y = np.array([int(k[i]["scribe"]) for i in index])
    folds = [(np.arange(len(train)), np.arange(len(train), len(index)))]
    prediction = held_out(x[index], y, folds, page_weights([k[i] for i in index]))
    return prediction[len(train):], len(train)


def main():
    for stem in sys.argv[1:]:
        k = [r for r in letters(*stem_letter(stem)) if not r["faint"]]
        x = embeddings_for(stem, k)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        lines = [f"{stem}, faint letters removed"]
        f115 = [i for i, r in enumerate(k) if r["page"].lstrip("f") == "115r" and r["line"] != ""]
        called, n_train = assign(k, x, ("2", "3"), f115)
        for part, early in (("lines 1 to 12 (Davis: Scribe 2)", True), ("lines 13 on (Davis: Scribe 3)", False)):
            mine = np.array([(int(k[i]["line"]) < 12) == early for i in f115])
            lines.append(f"f115r {part}: {np.sum(called[mine] == 2)} of {mine.sum()} assigned to Scribe 2 "
                         f"({np.mean(called[mine] == 2):.2f}); trained on {n_train} Scribe 2 and 3 letters")
        f58 = [i for i, r in enumerate(k) if r["page"].lstrip("f") in ("58r", "58v")]
        called, n_train = assign(k, x, ("1", "3"), f58)
        lines.append(f"ff58r/58v (Currier A, Davis: Scribe 3): {np.sum(called == 3)} of {len(f58)} assigned to Scribe 3 "
                     f"({np.mean(called == 3):.2f}); trained on {n_train} Scribe 1 and 3 letters")
        text = "\n".join(lines)
        (ROOT / f"outputs/classifier_tests/page_tests_{stem}.txt").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
