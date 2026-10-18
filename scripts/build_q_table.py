import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.budget_allocator import DEFAULT_ALLOW_THRESHOLD
from harness.q_estimator import estimate_q_table, models_in_logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate q_table from dev JSONL logs."
    )
    parser.add_argument(
        "log_paths",
        nargs="+",
        help=(
            "reviewer_decisions.jsonl files, or folders (such as a batch folder) "
            "searched recursively for them."
        ),
    )
    parser.add_argument(
        "--group-by",
        default="tool_name",
        help="Record field used to group similar calls.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum reviewer samples required for a group.",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=DEFAULT_ALLOW_THRESHOLD,
        help=(
            "Risk score below which the automatic decision is ALLOW. "
            "Must match the BudgetAllocator allow_threshold used in the run."
        ),
    )
    parser.add_argument(
        "--allow-mixed-models",
        action="store_true",
        help=(
            "Allow logs from different Judge/reviewer models. Off by default so "
            "local Ollama logs never end up in a DeepSeek q table."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()
    log_paths = expand_log_paths(args.log_paths)
    if not log_paths:
        raise SystemExit("No reviewer_decisions.jsonl files found.")

    try:
        q_table = estimate_q_table(
            log_paths,
            group_by=args.group_by,
            baseline_threshold=args.baseline_threshold,
            min_samples=args.min_samples,
            allow_mixed_models=args.allow_mixed_models,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for judge_model, reviewer_model in sorted(
        models_in_logs(log_paths), key=str
    ):
        print(
            f"logs include judge_model={judge_model} "
            f"reviewer_model={reviewer_model}",
            file=sys.stderr,
        )
    serialized = json.dumps(q_table, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {len(q_table)} q values to {output_path}")
    else:
        print(serialized)


def expand_log_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("reviewer_decisions.jsonl")))
        else:
            expanded.append(path)
    return expanded


if __name__ == "__main__":
    main()
