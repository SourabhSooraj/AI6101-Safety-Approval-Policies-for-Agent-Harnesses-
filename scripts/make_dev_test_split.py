"""Make a stratified dev/test split of the AgentDojo banking user tasks.

Strata follow where the injection sits and what the task asks the agent to
do (InjecAgent found the user task matters more than the attack). Near-duplicate
tasks stay on the same side so dev results don't leak into test.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_VERSION = "v1.2.2"
SUITE = "banking"
DEFAULT_SEED = 6101
DEFAULT_TEST_SIZE = 6
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "banking_dev_test_split.json"

# Each stratum lists its units; a unit with two tasks always moves together.
STRATA: dict[str, dict] = {
    "file_injection": {
        "description": (
            "Injection hidden in a file the agent reads "
            "(bill, landlord notice, address change)."
        ),
        "units": [["user_task_0"], ["user_task_2", "user_task_12"], ["user_task_13"]],
    },
    "read_only_answer": {
        "description": (
            "Answer from the transaction history; the expected solution "
            "only reads. Injection in an incoming transaction."
        ),
        "units": [["user_task_1"], ["user_task_7"], ["user_task_8"], ["user_task_10"]],
    },
    "send_money": {
        "description": (
            "Read the history, then send_money. Injection in an incoming "
            "transaction."
        ),
        "units": [["user_task_3", "user_task_4"], ["user_task_5"], ["user_task_11"]],
    },
    "schedule_or_account_change": {
        "description": (
            "Read the history, then change a scheduled payment, the "
            "password or account details. Injection in an incoming transaction."
        ),
        "units": [["user_task_6"], ["user_task_9"], ["user_task_14"], ["user_task_15"]],
    },
}

KEPT_TOGETHER = [["user_task_2", "user_task_12"], ["user_task_3", "user_task_4"]]


def _task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[-1])


def all_tasks() -> list[str]:
    tasks = [
        task
        for stratum in STRATA.values()
        for unit in stratum["units"]
        for task in unit
    ]
    return sorted(tasks, key=_task_number)


def make_split(
    seed: int = DEFAULT_SEED,
    test_size: int = DEFAULT_TEST_SIZE,
) -> dict:
    """Spread test_size test tasks as evenly as possible over the strata."""

    rng = random.Random(seed)
    names = sorted(STRATA)
    base, extra = divmod(test_size, len(names))
    bigger = set(rng.sample(names, extra))
    targets = {name: base + (1 if name in bigger else 0) for name in names}

    test: list[str] = []
    test_by_stratum: dict[str, list[str]] = {}
    for name in names:
        units = [list(unit) for unit in STRATA[name]["units"]]
        rng.shuffle(units)
        remaining = targets[name]
        picked: list[str] = []
        for unit in units:
            if len(unit) <= remaining:
                picked.extend(unit)
                remaining -= len(unit)
            if remaining == 0:
                break
        if remaining != 0:
            raise ValueError(
                f"Cannot pick {targets[name]} test tasks from stratum {name}"
            )
        test_by_stratum[name] = sorted(picked, key=_task_number)
        test.extend(picked)

    test = sorted(test, key=_task_number)
    dev = [task for task in all_tasks() if task not in test]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE,
        "seed": seed,
        "method": (
            "Stratified random split. Test tasks per stratum: test_size spread "
            "evenly, with the leftover strata chosen by random.Random(seed). "
            "Within a stratum, units are shuffled and taken in order while "
            "they fit. Units keep near-duplicate tasks together."
        ),
        "kept_together": KEPT_TOGETHER,
        "dev": dev,
        "test": test,
        "test_by_stratum": test_by_stratum,
        "strata": {
            name: {
                "description": STRATA[name]["description"],
                "tasks": sorted(
                    (t for unit in STRATA[name]["units"] for t in unit),
                    key=_task_number,
                ),
            }
            for name in names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a stratified dev/test split of AgentDojo banking user tasks."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path.",
    )
    args = parser.parse_args()

    split = make_split(seed=args.seed, test_size=args.test_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"dev  ({len(split['dev'])}): {', '.join(split['dev'])}")
    print(f"test ({len(split['test'])}): {', '.join(split['test'])}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
