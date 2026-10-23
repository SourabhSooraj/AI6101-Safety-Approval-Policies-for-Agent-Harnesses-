"""
Draws the full TCR@k plot on the test split from two evaluate.py outputs:
k = 0, 1, 5, unlimited and the cost ablation (results/a8-ablations/k-sweep/)
and k = 3 (results/b8-headline/, from b8). Run from the repository root:

    python results/a8-ablations/tcr-at-k/plot_combined.py

allow_all never escalates, so k doesn't change it; it is drawn at every k
as a flat reference.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from harness.evaluation import Rate  # noqa: E402
from scripts.evaluate import plot_tcr_at_k, write_csv  # noqa: E402

SOURCES = [
    ROOT / "results" / "a8-ablations" / "k-sweep" / "tcr_at_k.csv",
    ROOT / "results" / "b8-headline" / "tcr_at_k.csv",
]
ALLOW_ALL = "allow_all (no gate; k has no effect)"
ORDER = [
    "hybrid (value, tau0=1.0)",
    "no_budget (value, tau0=1.0)",
    "escalate_all",
    ALLOW_ALL,
    "hybrid (cost, tau0=0.05)",
]


def load_rows() -> list[dict]:
    rows = []
    for path in SOURCES:
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                total = int(float(record["tcr_n"]))
                row = {
                    "family": record["family"],
                    "k": record["k"] if record["k"] == "unlimited" else int(record["k"]),
                    "tcr": Rate(round(float(record["tcr"]) * total), total),
                }
                if record["family"] == "allow_all":
                    rows.extend({**row, "family": ALLOW_ALL, "k": k} for k in (0, 1, 3, 5))
                else:
                    rows.append(row)
    rows.sort(key=lambda row: (ORDER.index(row["family"]), 99 if row["k"] == "unlimited" else row["k"]))
    return rows


def main() -> None:
    rows = load_rows()
    out_dir = Path(__file__).resolve().parent
    plot_tcr_at_k(rows, out_dir)
    write_csv(
        out_dir / "tcr_at_k.csv",
        [
            {"family": row["family"], "k": row["k"], **{f"tcr_{key}": value for key, value in row["tcr"].as_dict().items()}}
            for row in rows
        ],
    )
    for row in rows:
        print(f"{row['family']:40s} k={row['k']!s:9s} {row['tcr'].successes}/{row['tcr'].total}")


if __name__ == "__main__":
    main()
