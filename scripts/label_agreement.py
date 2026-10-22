"""
读回各标注者填好的表，计算一致性，并整理出需要讨论的分歧：

    python scripts/label_agreement.py labels/sheet.csv \\
        labels/sheet_alice.csv labels/sheet_bob.csv --out labels/agreement

Reads the labellers' filled-in copies back, measures agreement and lists the
disagreements to settle. Prints Cohen's kappa, percent agreement and the
confusion matrix for each pair of labellers (Fleiss' kappa as well with three
or more), on the rows that needed a human: ground-truth rows are only used to
check each labeller against ground truth.

--all-rows also reports agreement on every row all labellers labelled,
ground-truth rows included (make the copies with make_label_sheet.py
--include-prelabelled). It sits next to the main figures, which stay on the
rows that needed a human: when those are few, the all-rows kappa is the one
with enough items, though ground-truth rows are easier and push it up.

--reviewer-logs adds the simulated reviewer's verdicts (ALLOW = safe,
DENY = unsafe) and reports each labeller and the agreed human label against
the reviewer, next to human-human agreement on the same rows (the ToolEmu
comparison).

With --out it writes agreement.json, disagreements.csv (one row per item the
labellers disagree on, with a final_label column to fill in), and
merged_labels.csv: the master sheet with every label all labellers agreed on.
After the team fills final_label in disagreements.csv, run again with
--settled <that file> to add the settled labels to merged_labels.csv, ready
for scripts/build_reviewer_error_table.py.
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
    SettledSheet,
    compare,
    fleiss_kappa,
    pairwise_agreement,
    read_labeller_sheet,
    read_settled,
    reviewer_labels,
)
from harness.reviewer_errors import write_label_sheet
from scripts.make_label_sheet import read_reviews, reviewer_log_paths

AGREED_SOURCE = "labellers_agree"
SETTLED_SOURCE = "settled"
GROUND_TRUTH_SOURCE = "ground_truth"
REVIEWER_NAME = "reviewer"
AGREED_HUMANS_NAME = "agreed humans"

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


def human_items(master_rows: list[dict[str, str]]) -> list[str]:
    """
    需要人工判断的行（master 中没有 ground truth 标签）。
    The rows that needed a human (no ground-truth label in the master sheet).
    """

    return [row["item_id"] for row in master_rows if not row.get("label")]


def ground_truth_labels(master_rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        row["item_id"]: row["label"].strip().casefold()
        for row in master_rows
        if row.get("label_source") == GROUND_TRUTH_SOURCE and row.get("label")
    }


def agreement_report(
    master_rows: list[dict[str, str]],
    sheets: list[LabellerSheet],
    reviewer: dict[str, str] | None = None,
    reviewer_models: set[Any] | None = None,
    all_rows: bool = False,
) -> dict[str, Any]:
    """
    人与人的一致性只在需要人工判断的行上计算：ground truth 行（只读调用、完全
    匹配）大家都会标对，放进来会抬高 κ。ground truth 行只用于检查每人与 ground
    truth 的一致性。给出 reviewer 标签时，在同一批行上比较人与 reviewer、人与人。
    Human-human agreement is computed only on the rows that needed a human:
    ground-truth rows (reads, exact matches) are ones everyone gets right, and
    would push kappa up. Ground-truth rows are used only to check each labeller
    against ground truth. With reviewer labels, humans vs the reviewer and
    humans vs each other are compared on the same rows.

    all_rows=True 时另外给出所有人都标注过的全部行（含 ground truth 行）上的
    κ，放在 "all_rows" 下；需要人工判断的行很少时，这个 κ 才有足够的样本。
    With all_rows=True, kappa on every row everyone labelled (ground-truth rows
    included) is added under "all_rows"; when few rows needed a human, this is
    the kappa with enough items.
    """

    known = {row["item_id"] for row in master_rows}
    needs_human = human_items(master_rows)
    complete = [
        item for item in needs_human if all(item in sheet.labels for sheet in sheets)
    ]
    truth = ground_truth_labels(master_rows)
    report: dict[str, Any] = {
        "labellers": [
            {
                "name": sheet.name,
                "labelled": len(sheet.labels),
                "ground_truth_rows": sum(item in truth for item in sheet.labels),
                "blank": len(sheet.blank),
                "invalid": [{"row": row, "value": value} for row, value in sheet.invalid],
                "duplicates": sheet.duplicates,
                "unknown_items": sorted(set(sheet.labels) - known),
            }
            for sheet in sheets
        ],
        "rows_needing_a_human": len(needs_human),
        "items_compared": len(complete),
        "pairs": pairwise_agreement(sheets, complete),
        "fleiss_kappa": _fleiss(sheets, complete),
        "versus_ground_truth": [
            {"name": sheet.name, **compare(sheet.name, sheet.labels, "ground truth", truth, truth)}
            for sheet in sheets
            if any(item in truth for item in sheet.labels)
        ],
    }
    if all_rows:
        everyone = [
            row["item_id"]
            for row in master_rows
            if all(row["item_id"] in sheet.labels for sheet in sheets)
        ]
        report["all_rows"] = {
            "items_compared": len(everyone),
            "ground_truth_rows": sum(item in truth for item in everyone),
            "pairs": pairwise_agreement(sheets, everyone),
            "fleiss_kappa": _fleiss(sheets, everyone),
        }
    if reviewer is not None:
        report["reviewer"] = reviewer_report(
            sheets, complete, reviewer, reviewer_models or set(), known, truth
        )
    return report


def _fleiss(sheets: list[LabellerSheet], items: list[str]) -> float | None:
    if len(sheets) < 3 or not items:
        return None
    return fleiss_kappa([[sheet.labels[item] for sheet in sheets] for item in items])


def reviewer_report(
    sheets: list[LabellerSheet],
    complete: list[str],
    reviewer: dict[str, str],
    models: set[Any],
    known: set[str],
    truth: dict[str, str],
) -> dict[str, Any]:
    """
    人与模拟 reviewer 的一致性（ToolEmu 的做法：把人–AI κ 与人–人 κ 比较）。
    两者都只用所有人都标注过、reviewer 也审核过的需人工判断的行。
    Agreement between humans and the simulated reviewer (as in ToolEmu:
    compare human-AI kappa with human-human kappa). Both use only the rows that
    needed a human, that every labeller labelled and the reviewer judged.
    """

    items = [item for item in complete if item in reviewer]
    agreed = {
        item: sheets[0].labels[item]
        for item in items
        if len({sheet.labels[item] for sheet in sheets}) == 1
    }
    return {
        # 旧日志没有记录 reviewer 模型。 / Older logs don't record the reviewer model.
        "models": sorted(str(model) if model else "unknown model" for model in models),
        "reviews": len(reviewer),
        "ground_truth_rows": sum(item in truth for item in reviewer),
        "unknown_items": sum(item not in known for item in reviewer),
        "items_compared": len(items),
        "labellers_vs_reviewer": [
            compare(sheet.name, sheet.labels, REVIEWER_NAME, reviewer, items)
            for sheet in sheets
        ],
        "agreed_humans_vs_reviewer": compare(
            AGREED_HUMANS_NAME, agreed, REVIEWER_NAME, reviewer, items
        ),
        "human_pairs_same_items": pairwise_agreement(sheets, items),
        "fleiss_kappa_same_items": _fleiss(sheets, items),
    }


def disagreements(
    master_rows: list[dict[str, str]],
    sheets: list[LabellerSheet],
    settled: SettledSheet | None = None,
) -> list[dict[str, str]]:
    """
    需要人工判断的行中，标注者意见不一致的条目；ground truth 行已有定论，不列出。
    final_label 带上 --settled 中已填的值，重新运行不会丢失讨论结果。
    Items the labellers disagree on, among the rows that needed a human;
    ground-truth rows are already decided and are not listed. final_label keeps
    any value from --settled, so running again never loses settled labels.
    Each labeller's comment sits next to their label, to help settle the row.
    """

    settled_labels = settled.labels if settled else {}
    rows = []
    for row in master_rows:
        if row.get("label"):
            continue
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
                    **{
                        column: value
                        for sheet in sheets
                        for column, value in (
                            (f"{sheet.name}_label", labels[sheet.name] or ""),
                            (f"{sheet.name}_comment", sheet.comments.get(row["item_id"], "")),
                        )
                    },
                    "final_label": settled_labels.get(row["item_id"], ""),
                }
            )
    return rows


def merged_labels(
    master_rows: list[dict[str, str]],
    sheets: list[LabellerSheet],
    settled: SettledSheet | None = None,
) -> list[dict[str, str]]:
    """
    master 中空白的行：所有标注者一致时填上他们的标签，否则填讨论后的 final_label；
    ground truth 标签不变。
    Blank rows of the master sheet get the labellers' label when all of them
    agree, otherwise the settled final_label; ground-truth labels are left as
    they are.
    """

    settled_labels = settled.labels if settled else {}
    merged = []
    for row in master_rows:
        row = dict(row)
        if not row.get("label"):
            labels = [sheet.labels.get(row["item_id"]) for sheet in sheets]
            if all(labels) and len(set(labels)) == 1:
                row["label"] = labels[0]
                row["label_source"] = AGREED_SOURCE
            elif row["item_id"] in settled_labels:
                row["label"] = settled_labels[row["item_id"]]
                row["label_source"] = SETTLED_SOURCE
        merged.append(row)
    return merged


def _print_pair(pair: dict[str, Any], matrix: bool = True) -> None:
    print(
        f"{pair['a']} vs {pair['b']} ({pair['items']} items): "
        f"agreement {_fmt(pair['percent_agreement'])}, "
        f"Cohen's kappa {_fmt(pair['cohen_kappa'])}"
    )
    if not matrix:
        return
    counts = pair["confusion_matrix"]
    row_names = [f"{pair['a']} {category}" for category in CATEGORIES]
    column_names = [f"{pair['b']} {category}" for category in CATEGORIES]
    left = max(len(name) for name in row_names) + 2
    cell = max(len(name) for name in column_names) + 2
    print(" " * left + "".join(f"{name:>{cell}}" for name in column_names))
    for name, a in zip(row_names, CATEGORIES):
        print(f"{name:<{left}}" + "".join(f"{counts[a][b]:>{cell}}" for b in CATEGORIES))


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
            + (f" ({labeller['ground_truth_rows']} of them ground-truth rows)"
               if labeller["ground_truth_rows"] else "")
            + (f"; {'; '.join(problems)}" if problems else "")
        )
    print(
        f"\nRows that needed a human: {report['rows_needing_a_human']}; "
        f"labelled by everyone: {report['items_compared']}"
    )
    for pair in report["pairs"]:
        print()
        _print_pair(pair)
    if len(report["labellers"]) >= 3:
        print(f"\nFleiss' kappa (all labellers): {_fmt(report['fleiss_kappa'])}")
    every = report.get("all_rows")
    if every is not None:
        print(
            f"\nAll rows everyone labelled: {every['items_compared']} "
            f"({every['ground_truth_rows']} of them ground-truth rows)"
        )
        for pair in every["pairs"]:
            print()
            _print_pair(pair)
        if len(report["labellers"]) >= 3:
            print(f"\nFleiss' kappa (all labellers, all rows): {_fmt(every['fleiss_kappa'])}")
    if report["versus_ground_truth"]:
        print("\nAgainst ground truth (ground-truth rows only):")
        for row in report["versus_ground_truth"]:
            _print_pair(row, matrix=False)

    reviewer = report.get("reviewer")
    if reviewer is None:
        return
    print(
        f"\nSimulated reviewer ({', '.join(reviewer['models'])}): "
        f"{reviewer['reviews']} reviews; compared on {reviewer['items_compared']} "
        "rows that needed a human and everyone labelled"
    )
    for pair in reviewer["labellers_vs_reviewer"]:
        print()
        _print_pair(pair)
    print()
    _print_pair(reviewer["agreed_humans_vs_reviewer"])
    print("\nHuman-human on the same rows (for comparison):")
    for pair in reviewer["human_pairs_same_items"]:
        _print_pair(pair, matrix=False)
    if reviewer["fleiss_kappa_same_items"] is not None:
        print(f"Fleiss' kappa: {_fmt(reviewer['fleiss_kappa_same_items'])}")


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agreement between labellers: Cohen's kappa per pair, Fleiss' "
            "kappa for three or more, percent agreement and confusion matrices, "
            "on the rows that needed a human. Optionally against the simulated "
            "reviewer too."
        )
    )
    parser.add_argument("master", type=Path, help="Master sheet from make_label_sheet.py.")
    parser.add_argument(
        "labellers", nargs="+", type=Path, help="The labellers' filled-in copies."
    )
    parser.add_argument(
        "--reviewer-logs",
        nargs="+",
        help=(
            "reviewer_decisions.jsonl files, run folders or batch folders: add "
            "the simulated reviewer (ALLOW = safe, DENY = unsafe; failed "
            "reviews are left out) and compare it with the labellers."
        ),
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help=(
            "Also report agreement on every row all labellers labelled, "
            "ground-truth rows included (copies made with --include-prelabelled)."
        ),
    )
    parser.add_argument(
        "--allow-mixed-models",
        action="store_true",
        help="Compare reviewer logs from more than one reviewer model anyway.",
    )
    parser.add_argument(
        "--settled",
        type=Path,
        help=(
            "disagreements.csv with final_label filled in: adds the settled "
            "labels to merged_labels.csv."
        ),
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
        if REVIEWER_NAME in names:
            parser.error(f"'{REVIEWER_NAME}' is reserved for --reviewer-logs")
        sheets = [read_labeller_sheet(path, name) for path, name in zip(args.labellers, names)]
        settled = read_settled(args.settled) if args.settled else None
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if settled is not None:
        if settled.invalid:
            rows = ", ".join(str(row) for row, _ in settled.invalid)
            parser.error(f"final_label on rows {rows} of {args.settled} must be safe or unsafe")
        to_check = set(human_items(master_rows))
        unknown = sorted(set(settled.labels) - to_check)
        if unknown:
            parser.error(
                f"{args.settled} settles items that are not rows needing a human "
                f"in {args.master}: {', '.join(unknown[:5])}"
            )

    reviewer = models = None
    if args.reviewer_logs:
        reviewer, models = reviewer_labels(read_reviews(reviewer_log_paths(args.reviewer_logs)))
        if not reviewer:
            parser.error("the reviewer logs have no successful reviews")
        if len(models) > 1 and not args.allow_mixed_models:
            parser.error(
                "the reviewer logs come from more than one reviewer model "
                f"({', '.join(sorted(map(str, models)))}); compare one model at a "
                "time, or pass --allow-mixed-models"
            )

    report = agreement_report(master_rows, sheets, reviewer, models, all_rows=args.all_rows)
    print_report(report)
    to_settle = disagreements(master_rows, sheets, settled)
    still_open = sum(not row["final_label"] for row in to_settle)
    print(f"\nDisagreements: {len(to_settle)}; still to settle: {still_open}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "agreement.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        columns = [
            "item_id", "run_id", "tool_call_id", "tool_name", "arguments",
            "user_request",
            *[column for name in names for column in (f"{name}_label", f"{name}_comment")],
            "final_label",
        ]
        write_csv(args.out / "disagreements.csv", to_settle, columns)
        write_label_sheet(
            merged_labels(master_rows, sheets, settled), args.out / "merged_labels.csv"
        )
        print(f"Wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
