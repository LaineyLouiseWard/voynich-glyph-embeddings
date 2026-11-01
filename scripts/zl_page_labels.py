"""Page labels from Zandbergen's IVTFF transcription (https://www.voynich.nu/data/ZL3b-n.txt, version 3b, 13 May 2025).

Reads data/raw/zandbergen/ZL3b-n.txt and writes config/zl_page_labels.csv: one row per page with the header
variables quire ($Q), page in quire ($P), bifolium in quire ($B), illustration ($I), Currier language ($L), Davis hand ($H), Currier hand ($C).
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"Q": "quire", "P": "page_in_quire", "B": "bifolium", "I": "illustration", "L": "language", "H": "davis_hand",
          "C": "currier_hand"}


def main():
    rows = []
    for line in (ROOT / "data/raw/zandbergen/ZL3b-n.txt").open(encoding="latin-1"):
        match = re.match(r"^<f(\w+)>\s+<!(.*)>", line)
        if match:
            values = dict(re.findall(r"\$(\w)=(\w+)", match.group(2)))
            rows.append({"page": match.group(1), **{name: values.get(key, "") for key, name in FIELDS.items()}})
    with (ROOT / "config/zl_page_labels.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["page", *FIELDS.values()])
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "pages")


if __name__ == "__main__":
    main()
