"""Estimate the reviewer's error rates from labelled dev escalations.

Output is the JSON that run_agentdojo_e2e.py reads with
--allocator cost --reviewer-error-table.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.reviewer_errors import (
    error_table_json,
    estimate_reviewer_errors,
    read_labels,
)
from scripts.make_label_sheet import read_reviews, reviewer_log_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate reviewer false-approve and false-deny rates per tool."
    )
    parser.add_argument(
        "reviewer_logs",
        nargs="+",
        help=(
            "reviewer_decisions.jsonl files, run directories, or batch "
            "directories (searched recursively)."
        ),
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Filled-in CSV from make_label_sheet.py.",
    )
    parser.add_argument(
        "--group-by",
        default="tool_name",
        help="Reviewer log field used to group calls.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help=(
            "Fewer labelled calls than this of one kind (safe or unsafe) in a "
            "group: that rate falls back to the pooled estimate."
        ),
    )
    parser.add_argument(
        "--allow-mixed-models",
        action="store_true",
        help=(
            "Allow reviews from different Judge/reviewer models. Off by default "
            "so local Ollama logs never end up in a DeepSeek error table."
        ),
    )
    parser.add_argument(
        "--output",
        help="JSON output path. Prints to stdout when omitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reviews = read_reviews(reviewer_log_paths(args.reviewer_logs))
    labels = read_labels(args.labels)
    try:
        estimates = estimate_reviewer_errors(
            reviews,
            labels,
            group_by=args.group_by,
            min_samples=args.min_samples,
            allow_mixed_models=args.allow_mixed_models,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    pooled = estimates["default"]
    print(
        f"Labelled reviews used: {pooled.unsafe_count} unsafe, "
        f"{pooled.safe_count} safe (of {len(reviews)} reviews).",
        file=sys.stderr,
    )
    for group, estimate in sorted(estimates.items()):
        print(
            f"  {group}: unsafe {estimate.unsafe_approved}/{estimate.unsafe_count} "
            f"approved, safe {estimate.safe_denied}/{estimate.safe_count} denied "
            f"-> false_approve {estimate.rates.false_approve:.3f}, "
            f"false_deny {estimate.rates.false_deny:.3f}",
            file=sys.stderr,
        )

    serialized = json.dumps(error_table_json(estimates), indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {len(estimates)} entries to {output_path}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
