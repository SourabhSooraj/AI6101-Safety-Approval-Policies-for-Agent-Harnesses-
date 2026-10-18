"""Make a CSV labelling sheet of the calls the simulated reviewer judged.

The sheet never shows the reviewer's verdict, so humans label blind.
With --ground-truth, calls that AgentDojo's ground truth settles are
pre-labelled; the rest are left blank for a human.
"""

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.ground_truth_labels import GroundTruthCall, label_call
from harness.reviewer_errors import read_reviewer_log, write_label_sheet
from harness.tool_metadata import BANKING_TOOL_METADATA

REVIEWER_LOG_NAME = "reviewer_decisions.jsonl"

# (user task id, injection task id or None) -> (user calls, injection calls)
GroundTruthLookup = Callable[
    [str, str | None],
    tuple[list[GroundTruthCall], list[GroundTruthCall]],
]


def reviewer_log_paths(paths: Iterable[str]) -> list[Path]:
    """Accept reviewer logs, run directories, or batch directories."""

    resolved = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir() and not (path / REVIEWER_LOG_NAME).is_file():
            # run_batch.py 的文件夹：每个 episode 子文件夹各有一份 reviewer 日志。
            found = sorted(path.rglob(REVIEWER_LOG_NAME))
            if not found:
                raise SystemExit(f"No reviewer log under {path}")
            resolved.extend(found)
            continue
        if path.is_dir():
            path = path / REVIEWER_LOG_NAME
        if not path.is_file():
            raise SystemExit(f"No reviewer log at {path}")
        resolved.append(path)
    return resolved


def read_reviews(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Successful reviews from every log, once per (run_id, tool_call_id)."""

    seen = set()
    reviews = []
    for path in paths:
        for record in read_reviewer_log(path):
            key = (str(record.get("run_id")), str(record.get("tool_call_id")))
            if key in seen:
                continue
            seen.add(key)
            reviews.append(record)
    return reviews


def build_rows(
    reviews: Iterable[dict[str, Any]],
    ground_truth: GroundTruthLookup | None = None,
) -> list[dict[str, Any]]:
    read_only_tools = frozenset(
        name
        for name, metadata in BANKING_TOOL_METADATA.items()
        if metadata.side_effect == "read"
    )
    rows = []
    for review in reviews:
        arguments = review.get("arguments") or {}
        label, note = None, ""
        if ground_truth is not None:
            user_calls, injection_calls = ground_truth(
                str(review.get("task_id")),
                review.get("injection_task_id"),
            )
            label, note = label_call(
                str(review.get("tool_name")),
                arguments,
                user_calls,
                injection_calls,
                read_only_tools=read_only_tools,
            )
        rows.append(
            {
                "run_id": review.get("run_id"),
                "tool_call_id": review.get("tool_call_id"),
                "task_id": review.get("task_id"),
                "injection_task_id": review.get("injection_task_id") or "",
                "tool_name": review.get("tool_name"),
                "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                "user_request": review.get("user_request", ""),
                "label": label or "",
                "label_source": "ground_truth" if label else "",
                "note": note,
            }
        )
    return rows


def agentdojo_ground_truth(version: str, suite_name: str) -> GroundTruthLookup:
    """Ground-truth calls from AgentDojo's default environment, cached per task."""

    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(version, suite_name)
    environment = suite.load_and_inject_default_environment({})
    cache: dict[str, list[GroundTruthCall]] = {}

    def calls(task) -> list[GroundTruthCall]:
        return [
            GroundTruthCall(function=call.function, args=dict(call.args))
            for call in task.ground_truth(environment.model_copy(deep=True))
        ]

    def lookup(task_id: str, injection_task_id: str | None):
        if task_id not in cache:
            cache[task_id] = calls(suite.get_user_task_by_id(task_id))
        injection_calls: list[GroundTruthCall] = []
        if injection_task_id:
            key = f"injection:{injection_task_id}"
            if key not in cache:
                cache[key] = calls(
                    suite.get_injection_task_by_id(injection_task_id)
                )
            injection_calls = cache[key]
        return cache[task_id], injection_calls

    return lookup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a blind labelling sheet of reviewed calls."
    )
    parser.add_argument(
        "reviewer_logs",
        nargs="+",
        help=f"{REVIEWER_LOG_NAME} files, run directories, or batch directories.",
    )
    parser.add_argument("--output", required=True, help="CSV path to write.")
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Pre-label calls that AgentDojo's ground truth settles. Dev runs only.",
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="banking")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reviews = read_reviews(reviewer_log_paths(args.reviewer_logs))
    ground_truth = (
        agentdojo_ground_truth(args.benchmark_version, args.suite)
        if args.ground_truth
        else None
    )
    rows = build_rows(reviews, ground_truth)
    count = write_label_sheet(rows, args.output)
    labelled = sum(1 for row in rows if row["label"])
    print(f"Wrote {count} reviewed calls to {args.output}")
    if args.ground_truth:
        print(f"Pre-labelled from ground truth: {labelled}; needs a human: {count - labelled}")


if __name__ == "__main__":
    main()
