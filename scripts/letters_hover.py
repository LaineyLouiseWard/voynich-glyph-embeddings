"""Interactive map of every letter: hover a point to see the cleaned k or n, its scribe, page and Currier language.

Same view as scripts/letters_map.py: PCA (2 components) fitted on the page-hand unit means of
outputs/unsupervised_classification/units_dinov3_unet_all.csv, and every k in data/all_k.csv
projected onto it from its L2-normalised DINOv3 embedding; a switch at the top shows the n map instead, made the same
way from the n letters' own units (units_dinov3_unet_all_n.csv), embeddings (dinov3_unet_all_n) and crops
(data/crops/unet_all_n/). Points are coloured blue, orange, green, red and purple for Scribes 1 to 5; the faintest
10%, left out of the tests, are grey and hidden until switched on in the legend; clicking a legend entry shows or hides
it, and the axes stay fixed to the range of all letters. A switch above the plot recolours the same letters by Currier
language instead: A brown and B grey, as in scripts/page_spread.py, and letters with no language light grey.
Thumbnails (96 px) are embedded, so the file stands alone.
Output: outputs/maps/letters_map.html.
Usage: python scripts/letters_hover.py
"""
import base64
import csv
import io
import json

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from sklearn.decomposition import PCA

from classifier_test import ROOT, embeddings_for
from davis_timm_tests import letters

COLOURS = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c", "4": "#d62728", "5": "#9467bd"}
LANGUAGE = {"A": "#8c564b", "B": "#7f7f7f"}
LETTERS = {"k": ("dinov3_unet_all", "unet_all"), "n": ("dinov3_unet_all_n", "unet_all_n")}
THUMB = 96
HOVER_JS = """
var plot = document.getElementById('{plot_id}');
plot.on('plotly_hover', function (data) {
  var c = data.points[0].customdata;
  var preview = document.getElementById('preview'), caption = document.getElementById('caption');
  preview.src = 'data:image/png;base64,' + THUMBS['{plot_id}'][c[3]];
  caption.textContent = 'Scribe ' + c[0] + ', page ' + c[1] + ', Currier ' + c[2] + ', letter ' + c[3];
});
"""
SWITCH_JS = """
function show(letter) {
  for (var l of ['k', 'n']) {
    document.getElementById(l + '_box').style.display = l == letter ? 'block' : 'none';
    document.getElementById(l + '_button').style.fontWeight = l == letter ? 'bold' : 'normal';
  }
}
"""


def thumbnail(letter_id, crops):
    img = Image.open(ROOT / "data/crops" / crops / f"{letter_id}.png").convert("L").resize((THUMB, THUMB))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def figure(letter):
    stem, crops = LETTERS[letter]
    units = list(csv.DictReader(open(ROOT / f"outputs/unsupervised_classification/units_{stem}.csv")))
    u = np.array([[float(r[f"e{i + 1}"]) for i in range(768)] for r in units])
    pca = PCA(2).fit(u)
    k = letters(True, letter)
    x = embeddings_for(stem, k)
    z = pca.transform(x / np.linalg.norm(x, axis=1, keepdims=True))
    fig = go.Figure()
    thumbs = {}
    for s, colour in COLOURS.items():
        idx = [i for i, r in enumerate(k) if r["scribe"] == s and not r["faint"]]
        custom = [[s, k[i]["page"], k[i]["language"] or "?", k[i]["annotation_id"]] for i in idx]
        thumbs.update((k[i]["annotation_id"], thumbnail(k[i]["annotation_id"], crops)) for i in idx)
        fig.add_trace(go.Scatter(x=z[idx, 0], y=z[idx, 1], mode="markers", name=f"Scribe {s} ({len(idx)})",
                                   marker=dict(color=colour, size=5, opacity=0.6), customdata=custom,
                                   hovertemplate="Scribe %{customdata[0]}, page %{customdata[1]}<extra></extra>"))
    shown = [i for i, r in enumerate(k) if r["scribe"] in COLOURS and not r["faint"]]
    groups = [(f"Currier {a}", c, [i for i in shown if k[i]["language"] == a]) for a, c in LANGUAGE.items()]
    unknown = [i for i in shown if k[i]["language"] not in LANGUAGE]
    if unknown:
        groups.append(("Currier unknown", "#c7c7c7", unknown))
    for name, colour, idx in groups:
        custom = [[k[i]["scribe"], k[i]["page"], k[i]["language"] or "?", k[i]["annotation_id"]] for i in idx]
        fig.add_trace(go.Scatter(x=z[idx, 0], y=z[idx, 1], mode="markers", name=f"{name} ({len(idx)})", visible=False,
                                 marker=dict(color=colour, size=5, opacity=0.6), customdata=custom,
                                 hovertemplate="Currier %{customdata[2]}, page %{customdata[1]}<extra></extra>"))
    idx = [i for i, r in enumerate(k) if r["faint"]]
    custom = [[k[i]["scribe"], k[i]["page"], k[i]["language"] or "?", k[i]["annotation_id"]] for i in idx]
    thumbs.update((k[i]["annotation_id"], thumbnail(k[i]["annotation_id"], crops)) for i in idx)
    fig.add_trace(go.Scatter(x=z[idx, 0], y=z[idx, 1], mode="markers", name=f"Faint, left out ({len(idx)})",
                             marker=dict(color="#999999", size=5, opacity=0.6), customdata=custom, visible="legendonly",
                             hovertemplate="Scribe %{customdata[0]}, page %{customdata[1]} (faint)<extra></extra>"))
    by_scribe = [True] * len(COLOURS) + [False] * len(groups) + ["legendonly"]
    by_language = [False] * len(COLOURS) + [True] * len(groups) + ["legendonly"]
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, xanchor="left", y=1.02, yanchor="bottom",
                                        buttons=[dict(label="Colour by scribe", method="restyle",
                                                      args=[{"visible": by_scribe}]),
                                                 dict(label="Colour by Currier A/B", method="restyle",
                                                      args=[{"visible": by_language}])])])
    pad = 0.03 * (z.max(axis=0) - z.min(axis=0))
    fig.update_xaxes(range=[z[:, 0].min() - pad[0], z[:, 0].max() + pad[0]], autorange=False)
    fig.update_yaxes(range=[z[:, 1].min() - pad[1], z[:, 1].max() + pad[1]], autorange=False)
    fig.update_layout(xaxis_title="PC 1", yaxis_title="PC 2", template="simple_white", width=900, height=700,
                      legend=dict(itemsizing="constant"))
    return fig, thumbs


def main():
    divs, thumbs = [], {}
    for n, letter in enumerate(LETTERS):
        fig, thumbs[letter] = figure(letter)
        html = fig.to_html(include_plotlyjs="cdn" if n == 0 else False, full_html=False, div_id=letter,
                           post_script=HOVER_JS)
        divs.append(f"<div id='{letter}_box' style='display:{'block' if n == 0 else 'none'}'>{html}</div>")
    buttons = "".join(f"<button id='{letter}_button' onclick=\"show('{letter}')\" style='font-size:16px;margin-right:8px;"
                      f"font-weight:{'bold' if n == 0 else 'normal'}'>Letter {letter}</button>"
                      for n, letter in enumerate(LETTERS))
    page = ("<html><head><meta charset='utf-8'><title>Voynich k and n maps</title>"
            f"<script>var THUMBS = {json.dumps(thumbs)};{SWITCH_JS}</script></head>"
            "<body style='font-family:sans-serif;display:flex;gap:24px;align-items:flex-start'>"
            f"<div><div>{buttons}</div>{''.join(divs)}</div><div style='padding-top:80px'>"
            f"<img id='preview' width='{2 * THUMB}' height='{2 * THUMB}' style='border:1px solid #ccc'>"
            "<p id='caption'>Hover over a point</p></div></body></html>")
    (ROOT / "outputs/maps/letters_map.html").write_text(page)


if __name__ == "__main__":
    main()
