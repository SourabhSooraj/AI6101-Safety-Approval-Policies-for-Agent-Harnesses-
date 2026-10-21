"""
读取一个或多个 run_batch.py 的 batch 文件夹，输出评估结果：

    python scripts/evaluate.py logs/agentdojo_batch/<batch_id> --out results/dev

打印主要表格，并在 --out 中写入 results.json、每张表一个 CSV（含
tcr_at_k.csv），以及 TCR@k 曲线图 tcr_at_k.png 和 tcr_at_k.pdf。
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端：在服务器和 Colab 上也能保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.evaluation import Rate, evaluate, load_runs

TABLES = (
    "overview",
    "tcr_at_k",
    "hitl",
    "cost",
    "by_attack_type",
    "by_user_task",
    "mcnemar",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Rate):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    """CSV 的一行：Rate 拆成 rate/low/high/n，嵌套的成本拆成 <组件>_<字段>。"""

    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Rate):
            data = value.as_dict()
            flat[key] = data["rate"]
            flat[f"{key}_low"] = data["low"]
            flat[f"{key}_high"] = data["high"]
            flat[f"{key}_n"] = data["n"]
        elif isinstance(value, dict):
            for inner, item in value.items():
                flat[f"{key}_{inner}"] = item
        else:
            flat[key] = value
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [flatten(row) for row in rows]
    columns: list[str] = []
    for row in flat_rows:
        columns += [key for key in row if key not in columns]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)


PLOT_FORMATS = ("png", "pdf")


def plot_tcr_at_k(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """画 TCR@k 曲线（含 95% Wilson 区间），保存为 PNG 和 PDF，返回写出的文件。"""

    finite = sorted({int(row["k"]) for row in rows if row["k"] != "unlimited"})
    figure, axes = plt.subplots(figsize=(7, 4.5))
    families = dict.fromkeys(row["family"] for row in rows)
    for index, family in enumerate(families):
        # 每个 family 一个固定颜色；unlimited 的虚线与它的 family 同色。
        color = f"C{index % 10}"
        points = [row for row in rows if row["family"] == family and row["tcr"].total]
        curve = [row for row in points if row["k"] != "unlimited"]
        unlimited = [row for row in points if row["k"] == "unlimited"]
        if curve:
            ks = [int(row["k"]) for row in curve]
            values = [row["tcr"].value for row in curve]
            intervals = [row["tcr"].as_dict() for row in curve]
            axes.plot(ks, values, marker="o", color=color, label=family)
            axes.fill_between(
                ks,
                [item["low"] for item in intervals],
                [item["high"] for item in intervals],
                color=color,
                alpha=0.15,
            )
        for row in unlimited:
            axes.axhline(
                row["tcr"].value,
                color=color,
                linestyle="--",
                label=f"{family} (unlimited)",
            )
    axes.set_xlabel("escalation budget k")
    axes.set_ylabel("TCR (safe task completion rate)")
    axes.set_ylim(0, 1)
    if finite:
        axes.set_xticks(finite)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    paths = []
    for extension in PLOT_FORMATS:
        path = out_dir / f"tcr_at_k.{extension}"
        # 不写创建时间，同样的数据生成同样的 PDF，方便放进报告的版本控制。
        metadata = {"CreationDate": None} if extension == "pdf" else None
        figure.savefig(path, dpi=150, metadata=metadata)
        paths.append(path)
    plt.close(figure)
    return paths


def _cell(value: Any) -> str:
    if isinstance(value, Rate):
        data = value.as_dict()
        if data["rate"] is None:
            return "-"
        return (
            f"{data['rate']:.2f} [{data['low']:.2f},{data['high']:.2f}] "
            f"n={data['n']}"
        )
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    cells = [[_cell(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(line[index]) for line in cells))
        for index, column in enumerate(columns)
    ]
    print("  " + "  ".join(column.ljust(width) for column, width in zip(columns, widths)))
    for line in cells:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate run_batch.py batch folders: utility, TCR@k, attack "
            "success, HITL load, cost, breakdowns, Wilson intervals and "
            "McNemar tests with Holm correction."
        )
    )
    parser.add_argument("batches", nargs="+", type=Path, help="Batch folders.")
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Folder for results.json, the CSV tables and the TCR@k plot "
            "(tcr_at_k.png and tcr_at_k.pdf)."
        ),
    )
    parser.add_argument(
        "--allow-mixed-models",
        action="store_true",
        help="Combine runs whose agent, Judge or reviewer models differ.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    for batch in args.batches:
        if not batch.is_dir():
            parser.error(f"{batch} is not a folder")
    try:
        runs, warnings = load_runs(args.batches, args.allow_mixed_models)
    except ValueError as error:
        parser.error(str(error))
    if not runs:
        parser.error("no runs found in the given folders")

    results = evaluate(runs)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"{len(runs)} runs from {len(args.batches)} batch folder(s)")
    print("models: " + " | ".join(results["models"]))
    print_table(
        "Outcomes (crashed runs count as not done)",
        results["overview"],
        ["setup", "k", "runs", "crashed", "utility_no_attack", "tcr",
         "asr_finished", "asr_all_runs"],
    )
    print_table(
        "TCR@k", results["tcr_at_k"],
        ["family", "k", "tcr", "tcr_no_attack", "tcr_under_attack"],
    )
    print_table(
        "HITL load", results["hitl"],
        ["setup", "k", "escalations_per_run", "budget_share_used", "auto_denied",
         "runs_out_of_budget", "reviewer_approved"],
    )
    print_table(
        "Cost (USD)",
        [
            {**row, "agent": row["agent"]["usd"], "judge": row["judge"]["usd"],
             "reviewer": row["reviewer"]["usd"]}
            for row in results["cost"]
        ],
        ["setup", "k", "agent", "judge", "reviewer", "total_usd", "usd_per_run"],
    )
    print_table(
        "McNemar (exact, Holm-corrected within each metric)",
        results["mcnemar"],
        ["metric", "a", "b", "pairs", "only_a", "only_b", "p", "p_holm",
         "significant"],
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(
            json.dumps(
                {**_json_value(results), "warnings": warnings}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        for table in TABLES:
            write_csv(args.out / f"{table}.csv", results[table])
        plot_tcr_at_k(results["tcr_at_k"], args.out)
        print(f"\nwrote {args.out}")
    return results


if __name__ == "__main__":
    main()
