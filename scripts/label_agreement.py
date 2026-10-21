"""
读回各标注者填好的表，计算一致性，并整理出需要讨论的分歧：

    python scripts/label_agreement.py labels/sheet.csv \\
        labels/sheet_alice.csv labels/sheet_bob.csv --out labels/agreement

Reads the labellers' filled-in copies back, measures agreement and lists the
disagreements to settle. Prints Cohen's kappa, percent agreement and the
confusion matrix for each pair of labellers (Fleiss' kappa as well with three
or more). With --out it writes agreement.json, disagreements.csv (one row per
item the labellers disagree on, with a blank final_label column), and
merged_labels.csv: the master sheet with the label filled in wherever every
labeller agreed. After the team settles the disagreements and copies the final
labels into merged_labels.csv, it can be passed to
scripts/build_reviewer_error_table.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.label_agreement import (
    CATEGORIES,
    LabellerSheet,
    cohen_kappa,
    fleiss_kappa,
    pairwise_agreement,
    percent_agreement,
    read_labeller_sheet,
)
from harness.reviewer_errors import write_label_sheet

AGREED_SOURCE = "labellers_agree"


def read_master(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "item_id" not in (reader.fieldnames or ()):
            raise ValueError(
                f"{path} has no item_id column; make it again with "
                "scripts/make_label_sheet.py"
            )
        return list(reader)


def labeller_name(path: Path, master: Path) -> str:
    """sheet_alice.csv -> alice（以 master 文件名为前缀时）；否则用文件名。
    sheet_alice.csv -> alice when the master's name is the prefix; otherwise
    the file name."""

    prefix = f"{master.stem}_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def agreement_report(
    master_rows: list[dict[str, str]], sheets: list[LabellerSheet]
) -> dict[str, Any]:
    """
    所有标注者都给出有效标签的条目上的一致性；另外报告每人与 ground truth 的
    一致性（标注者表包含预标注行时）。
    Agreement on the items every labeller gave a valid label, plus each
    labeller's agreement with ground truth (when their copies included the
    pre-labelled rows).
    """

    known = {row["item_id"] for row in master_rows}
    complete = [
        row["item_id"]
        for row in master_rows
        if all(row["item_id"] in sheet.labels for sheet in sheets)
    ]
    report: dict[str, Any] = {
        "labellers": [
            {
                "name": sheet.name,
                "labelled": len(sheet.labels),
                "blank": len(sheet.blank),
                "invalid": [{"row": row, "value": value} for row, value in sheet.invalid],
                "duplicates": sheet.duplicates,
                "unknown_items": sorted(set(sheet.labels) - known),
            }
            for sheet in sheets
        ],
        "items_compared": len(complete),
        "pairs": pairwise_agreement(sheets, complete),
        "fleiss_kappa": (
            fleiss_kappa([[sheet.labels[item] for sheet in sheets] for item in complete])
            if len(sheets) >= 3
            else None
        ),
    }

    ground_truth = {
        row["item_id"]: row["label"].strip().casefold()
        for row in master_rows
        if row.get("label_source") == "ground_truth" and row.get("label")
    }
    versus_truth = []
    for sheet in sheets:
        shared = [item for item in ground_truth if item in sheet.labels]
        if shared:
            truth = [ground_truth[item] for item in shared]
            labels = [sheet.labels[item] for item in shared]
            versus_truth.append(
                {
                    "name": sheet.name,
                    "items": len(shared),
                    "percent_agreement": percent_agreement(labels, truth),
                    "cohen_kappa": cohen_kappa(labels, truth),
                }
            )
    report["versus_ground_truth"] = versus_truth
    return report


def disagreements(
    master_rows: list[dict[str, str]], sheets: list[LabellerSheet]
) -> list[dict[str, str]]:
    rows = []
    for row in master_rows:
        labels = {sheet.name: sheet.labels.get(row["item_id"]) for sheet in sheets}
        given = {label for label in labels.values() if label}
        if len(given) > 1:
            rows.append(
                {
                    "item_id": row["item_id"],
                    "run_id": row.get("run_id", ""),
                    "tool_call_id": row.get("tool_call_id", ""),
                    "tool_name": row.get("tool_name", ""),
                    "arguments": row.get("arguments", ""),
                    "user_request": row.get("user_request", ""),
                    **{f"{name}_label": label or "" for name, label in labels.items()},
                    "final_label": "",
                }
            )
    return rows


def merged_labels(
    master_rows: list[dict[str, str]], sheets: list[LabellerSheet]
) -> list[dict[str, str]]:
    """
    master 中空白的行，若所有标注者一致，就填上他们的标签；ground truth 标签不变。
    Blank rows of the master sheet get the labellers' label when all of them
    agree; ground-truth labels are left as they are.
    """

    merged = []
    for row in master_rows:
        row = dict(row)
        if not row.get("label"):
            labels = [sheet.labels.get(row["item_id"]) for sheet in sheets]
            if all(labels) and len(set(labels)) == 1:
                row["label"] = labels[0]
                row["label_source"] = AGREED_SOURCE
        merged.append(row)
    return merged


def print_report(report: dict[str, Any]) -> None:
    for labeller in report["labellers"]:
        problems = []
        if labeller["blank"]:
            problems.append(f"{labeller['blank']} blank")
        if labeller["invalid"]:
            rows = ", ".join(str(item["row"]) for item in labeller["invalid"])
            problems.append(f"invalid labels on rows {rows} (use safe or unsafe)")
        if labeller["duplicates"]:
            problems.append(f"{len(labeller['duplicates'])} duplicate rows")
        if labeller["unknown_items"]:
            problems.append(f"{len(labeller['unknown_items'])} items not in the master sheet")
        print(
            f"{labeller['name']}: {labeller['labelled']} labelled"
            + (f"; {'; '.join(problems)}" if problems else "")
        )
    print(f"\nItems every labeller labelled: {report['items_compared']}")
    for pair in report["pairs"]:
        print(
            f"\n{pair['a']} vs {pair['b']} ({pair['items']} items): "
            f"agreement {_fmt(pair['percent_agreement'])}, "
            f"Cohen's kappa {_fmt(pair['cohen_kappa'])}"
        )
        matrix = pair["confusion_matrix"]
        row_names = [f"{pair['a']} {category}" for category in CATEGORIES]
        column_names = [f"{pair['b']} {category}" for category in CATEGORIES]
        left = max(len(name) for name in row_names) + 2
        cell = max(len(name) for name in column_names) + 2
        print(" " * left + "".join(f"{name:>{cell}}" for name in column_names))
        for name, a in zip(row_names, CATEGORIES):
            print(f"{name:<{left}}" + "".join(f"{matrix[a][b]:>{cell}}" for b in CATEGORIES))
    if len(report["labellers"]) >= 3:
        print(f"\nFleiss' kappa (all labellers): {_fmt(report['fleiss_kappa'])}")
    for row in report["versus_ground_truth"]:
        print(
            f"\n{row['name']} vs ground truth ({row['items']} items): "
            f"agreement {_fmt(row['percent_agreement'])}, "
            f"Cohen's kappa {_fmt(row['cohen_kappa'])}"
        )


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agreement between labellers: Cohen's kappa per pair, Fleiss' "
            "kappa for three or more, percent agreement and confusion matrices."
        )
    )
    parser.add_argument("master", type=Path, help="Master sheet from make_label_sheet.py.")
    parser.add_argument(
        "labellers", nargs="+", type=Path, help="The labellers' filled-in copies."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Folder for agreement.json, disagreements.csv and merged_labels.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.labellers) < 2:
        parser.error("agreement needs at least two labellers")
    try:
        master_rows = read_master(args.master)
        names = [labeller_name(path, args.master) for path in args.labellers]
        if len(set(names)) != len(names):
            parser.error(f"two copies have the same labeller name: {names}")
        sheets = [read_labeller_sheet(path, name) for path, name in zip(args.labellers, names)]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    report = agreement_report(master_rows, sheets)
    print_report(report)
    to_settle = disagreements(master_rows, sheets)
    print(f"\nDisagreements to settle: {len(to_settle)}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "agreement.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        columns = [
            "item_id", "run_id", "tool_call_id", "tool_name", "arguments",
            "user_request", *[f"{name}_label" for name in names], "final_label",
        ]
        write_csv(args.out / "disagreements.csv", to_settle, columns)
        write_label_sheet(merged_labels(master_rows, sheets), args.out / "merged_labels.csv")
        print(f"Wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
