# Settings for the `iin` and `n` tests, fixed before any classifier run

**Letters.** Every `iin` in `data/all_iin.csv` (Timm's 854 boxes plus the letters found by `find_k.py --letter iin`),
cleaned by the `iin` U-Net (`unet_all_k.py --letter iin`). The `n` alone is each cleaned `iin` cut at the start of its
last upright stroke (`cut_n.py`); letters with no cut are left out of the `n`. The `n` is the primary result if the cut
passed its review (more than 5 of 50 in the wrong place drops it), because Davis judges the `n`; otherwise the whole
`iin` is. Both are reported either way. With two strokes in the lower half, the cut is kept only if their widths are within
1.5 times (`RATIO` in `cut_n.py`); set after the first cut review (6 of 50 wrong), and a second review of 50 other
letters found 0 wrong.

**Letters with joined neighbour ink.** Letters whose cleaned `iin` ink touches the box's top or left edge are left out
(`touches_top_or_left` in `davis_timm_tests.py`): 2,282 of 3,113 kept. Set after the first blind review failed (62 of 360
bad) and before any classifier run; a second blind review of 360 other letters (seed 1, rated by a separate agent) found
24 bad (6.7%), under the 10% limit.

**Faint letters.** The 10% of letters with the smallest share of ink after cleaning are removed, measured on the cleaned
`iin` (`data/crops/unet_box_iin/`), so the `iin` and `n` tests use the same letters.

**Unchanged from `k`.** DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, class token, 768 numbers) on the squares of
`square_crops.py` (padded to a centred square, 224 pixels, nearest neighbour), each embedding scaled to length 1.
Logistic regression (C = 1), every page weighted equally within its scribe and the two scribes weighted equally; 10 folds
by sheet of parchment (quire and bifolium from Zandbergen's file); balanced accuracy; 95% range from 1,000 resamples of
each scribe's sheets; p from 200 shuffles of the scribe labels between sheets, p = (1 + shuffles at least as high) / 201.

**Scribe pairs, as for `k`.** Scribe 1 vs 2, 1 vs 3, 2 vs 3 (all pages), 1 vs 2 on herbal pages (folios 1 to 66):
`pairs_and_blend.py`. Scribe 2 vs 3 on Currier B pages, and the classifier trained on Scribes 1 and 2 applied to
Scribes 3 to 5: `davis_timm_tests.py`. Scribes 4 and 5 are left out of the pair tests. Scribe 2 vs 3 reads "different"
if p is under 0.05, "alike" if the top of the 95% range is under 60%, otherwise "can't tell". (`classifier_test.py`'s
own command reads only Timm's `k`; its functions are what both scripts use.)

**Two extra tests** (`page_tests.py`, faint letters removed, same classifier, trained once with the whole target sheet
held back):
- f115r: trained on Scribe 2 and 3 letters from every other sheet; the share of f115r letters assigned to Scribe 2, for
  lines 1 to 12 (Davis: Scribe 2) and for lines 13 on (Davis: Scribe 3), for `k`, `iin` and `n`. Letters with no placed
  line are left out.
- ff58r/58v (Currier A, Davis's Scribe 3): trained on Scribe 1 and 3 letters from every other sheet; the share of their
  letters assigned to Scribe 3.

No setting changes after the first result.
