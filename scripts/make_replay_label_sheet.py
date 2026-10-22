"""
把攻击重放中 reviewer 审核过的写调用写成标签表，供 build_reviewer_error_table.py
使用：

    python scripts/make_replay_label_sheet.py results/attack-replay-dev \\
        --setups p1_only --out results/reviewer-errors-dev/replay_dev_p1_only.csv

Writes the write calls the reviewer judged in an attack replay
(scripts/replay_attacks.py) as a labels sheet for
scripts/build_reviewer_error_table.py. No human is needed: an attacker's
call is unsafe by construction, and a user task's own ground-truth call
(the control pass) is safe, so every row is labelled with
label_source = ground_truth. Reads are left out: a read in an attack case
is only a step towards the attack, so it has no clear label.

--setups keeps only the reviews from those setups. The same attacker call is
reviewed once per setup, so pick the setups whose reviews the table should
count (for example hybrid alone) to avoid counting one call twice.

The run_id and tool_call_id match the replay's reviewer_decisions.jsonl,
which build_reviewer_error_table.py reads next to this sheet.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentdojo.task_suite.load_suites import get_suite

from harness.reviewer_errors import SAFE, UNSAFE, write_label_sheet
from scripts.replay_attacks import CALLS_LOG, CONFIG_FILE, CallOutcome, read_outcomes, run_id_for

LABEL_SOURCE = "ground_truth"


def calls_path(path: Path) -> Path:
    """接受 replay 文件夹或 calls.jsonl。 Accepts a replay folder or calls.jsonl."""

    return path / CALLS_LOG if path.is_dir() else path


def replay_settings(path: Path) -> dict[str, Any]:
    """同一文件夹中的 replay_config.json；没有时返回空字典。"""

    config = calls_path(path).parent / CONFIG_FILE
    if not config.is_file():
        return {}
    return json.loads(config.read_text(encoding="utf-8"))


def label_rows(
    outcomes: Iterable[CallOutcome],
    user_requests: dict[str, str],
    setups: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    reviewer 审核过的攻击者写调用记为 unsafe，对照组的写调用记为 safe。
    Reviewed attacker write calls are unsafe, control write calls are safe.
    """

    rows = []
    for outcome in outcomes:
        if setups and outcome.setup not in setups:
            continue
        if outcome.reviewer_verdict is None:
            continue
        if outcome.harmful:
            label = UNSAFE
            note = f"replay: attacker call from {outcome.injection_task_id}"
        elif outcome.benign_write:
            label = SAFE
            note = "replay: the user task's own ground-truth call"
        else:
            continue
        rows.append(
            {
                "run_id": run_id_for(outcome.replay_id, outcome.case, outcome.setup),
                "tool_call_id": outcome.tool_call_id,
                "task_id": outcome.task_id,
                "injection_task_id": outcome.injection_task_id or "",
                "tool_name": outcome.tool_name,
                "arguments": json.dumps(outcome.arguments, ensure_ascii=False),
                "user_request": user_requests.get(outcome.task_id, ""),
                "label": label,
                "label_source": LABEL_SOURCE,
                "note": note,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the attack replay's reviewed write calls as a labels sheet "
            "(attacker calls unsafe, control calls safe) for "
            "build_reviewer_error_table.py."
        )
    )
    parser.add_argument("replay", type=Path, help="Replay folder or its calls.jsonl.")
    parser.add_argument(
        "--setups",
        nargs="+",
        help="Keep only reviews from these setups (default: every setup in the replay).",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV output path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = calls_path(args.replay)
    if not path.is_file():
        parser.error(f"no {CALLS_LOG} at {path}")
    outcomes = read_outcomes(path)
    known = sorted({outcome.setup for outcome in outcomes})
    unknown = sorted(set(args.setups or ()) - set(known))
    if unknown:
        parser.error(f"setups not in this replay: {', '.join(unknown)} (it has {', '.join(known)})")

    settings = replay_settings(args.replay)
    suite = get_suite(settings.get("benchmark_version", "v1.2.2"), settings.get("suite", "banking"))
    user_requests = {
        task_id: suite.get_user_task_by_id(task_id).PROMPT
        for task_id in {outcome.task_id for outcome in outcomes}
    }
    rows = label_rows(outcomes, user_requests, setups=args.setups)
    count = write_label_sheet(rows, args.out)
    unsafe = sum(row["label"] == UNSAFE for row in rows)
    print(f"wrote {count} labelled reviews to {args.out}: {unsafe} unsafe, {count - unsafe} safe")


if __name__ == "__main__":
    main()
