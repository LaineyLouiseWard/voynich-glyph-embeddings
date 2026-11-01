"""Parse the Archetype pg_dump (COPY blocks) into lists of dicts, no Postgres needed."""
import re
from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "data/raw/digipal_project/archetype.sql"
COPY_RE = re.compile(r"^COPY (\S+) \((.*)\) FROM stdin;$")


def _unescape(v):
    if v == r"\N":
        return None
    return (v.replace(r"\t", "\t").replace(r"\n", "\n").replace(r"\r", "\r")
             .replace("\\\\", "\\"))


def load_tables(wanted=None, path=SQL):
    tables, cur, cols = {}, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if cur is None:
                m = COPY_RE.match(line)
                if m:
                    name = m.group(1).split(".")[-1]
                    if wanted is None or name in wanted:
                        cur, cols = name, [c.strip().strip('"') for c in m.group(2).split(",")]
                        tables[cur] = []
            elif line == r"\.":
                cur = None
            else:
                tables[cur].append(dict(zip(cols, map(_unescape, line.split("\t")))))
    return tables


def glyph_table():
    """One row per annotation: glyph type, hand, page and polygon."""
    t = load_tables({"digipal_annotation", "digipal_graph", "digipal_idiograph", "digipal_allograph",
                     "digipal_character", "digipal_hand", "digipal_image"})
    by = lambda name: {r["id"]: r for r in t[name]}
    graph, idio, allo, char, hand, img = (by(n) for n in ("digipal_graph", "digipal_idiograph",
                                                         "digipal_allograph", "digipal_character",
                                                         "digipal_hand", "digipal_image"))
    rows = []
    for a in t["digipal_annotation"]:
        g = graph.get(a["graph_id"])
        al = allo.get(idio[g["idiograph_id"]]["allograph_id"]) if g else None
        im = img.get(a["image_id"])
        rows.append({
            "annotation_id": a["id"], "image_id": a["image_id"], "graph_id": a["graph_id"],
            "allograph": al["name"] if al else None,
            "character": char[al["character_id"]]["name"] if al else None,
            "hand": hand[g["hand_id"]]["label"] if g and g["hand_id"] else None,
            "locus": im["locus"] if im else None, "iipimage": im["iipimage"] if im else None,
            "width": im["width"] if im else None, "height": im["height"] if im else None,
            "geo_json": a["geo_json"], "rotation": a.get("rotation"),
        })
    return rows
