import argparse
import itertools
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.episode import (
    add_episode_args,
    config_from_args,
    print_config_warnings,
    run_episode,
)
from harness.policies import POLICY_VARIANTS

# dev 用于调参和 q table，test 只用于最终测试。划分由
# scripts/make_dev_test_split.py 生成（分层抽样，近似重复的任务放在同一边）。
SPLIT_NAMES = ("dev", "test")
DEFAULT_SPLIT_FILE = REPO_ROOT / "configs" / "banking_dev_test_split.json"
NO_ATTACK = "none"


def load_split(
    path: Path,
    split: str,
    *,
    suite: str,
    benchmark_version: str,
) -> list[str]:
    """读取划分文件中的一边；文件与本次运行的 suite / 版本不符时报错。"""

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (data.get("suite"), data.get("benchmark_version")) != (
        suite,
        benchmark_version,
    ):
        raise ValueError(
            f"{path} is a split for {data.get('suite')} "
            f"{data.get('benchmark_version')}, but this run uses "
            f"{suite} {benchmark_version}"
        )
    dev, test = data.get("dev"), data.get("test")
    if not dev or not test:
        raise ValueError(f"{path} needs non-empty 'dev' and 'test' lists")
    overlap = sorted(set(dev) & set(test))
    if overlap:
        raise ValueError(f"{path} puts these tasks on both sides: {overlap}")
    return list(dev if split == "dev" else test)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every (task, policy, injection task) combination and write "
            "one log folder per episode plus a batch summary."
        )
    )
    add_episode_args(parser)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        help="Run the dev or test tasks listed in --split-file.",
    )
    tasks.add_argument("--task-ids", nargs="+", help="Explicit user task ids.")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="Dev/test split JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_VARIANTS,
        default=["hybrid"],
    )
    parser.add_argument(
        "--injection-tasks",
        nargs="+",
        default=[NO_ATTACK],
        help=f"Injection task ids; '{NO_ATTACK}' runs without an attack.",
    )
    parser.add_argument(
        "--batch-id",
        help=(
            "Folder name under logs/agentdojo_batch. Reuse an existing id to "
            "resume: finished episodes are skipped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the episodes that would run, without calling any model.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.split:
        try:
            task_ids = load_split(
                args.split_file,
                args.split,
                suite=args.suite,
                benchmark_version=args.benchmark_version,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        task_ids = args.task_ids
    episodes = list(
        itertools.product(task_ids, args.policies, args.injection_tasks)
    )
    if args.dry_run:
        for task_id, policy, injection in episodes:
            print(f"{task_id} policy={policy} injection={injection}")
        print(f"{len(episodes)} episodes")
        return

    config = config_from_args(args)
    print_config_warnings(config)

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    batch_dir = REPO_ROOT / "logs" / "agentdojo_batch" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        key: value
        for key, value in asdict(config).items()
        if not key.endswith("api_key")
    }
    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                **config_record,
                "split": args.split,
                "split_file": str(args.split_file) if args.split else None,
                "task_ids": task_ids,
                "policies": args.policies,
                "injection_tasks": args.injection_tasks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = batch_dir / "batch_summary.jsonl"

    for index, (task_id, policy, injection) in enumerate(episodes, start=1):
        episode_name = f"{task_id}__{policy}__{injection}"
        episode_dir = batch_dir / episode_name
        label = f"[{index}/{len(episodes)}] {episode_name}"
        if (episode_dir / "run_summary.json").exists():
            print(f"{label} already done, skipping")
            continue

        # 上次中断的 episode 留下的日志会被追加写入，先删掉以免重复计数。
        for stale_log in episode_dir.glob("*.jsonl"):
            stale_log.unlink()
        print(f"{label} running", flush=True)
        injection_task_id = None if injection == NO_ATTACK else injection
        try:
            summary = run_episode(
                config,
                task_id=task_id,
                policy=policy,
                injection_task_id=injection_task_id,
                run_id=f"{batch_id}/{episode_name}",
                log_dir=episode_dir,
            )
        except Exception as error:  # noqa: BLE001 - 一个 episode 失败不应中断整批
            traceback.print_exc()
            summary = {
                "task_id": task_id,
                "policy": policy,
                "injection_task_id": injection_task_id,
                "error": repr(error),
            }
        else:
            summary["total_cost_usd"] = episode_cost(summary)
            print(
                f"{label} utility={summary['utility']} "
                f"strict_utility={summary['strict_utility']} "
                f"attack_succeeded={summary['attack_succeeded']} "
                f"cost=${summary['total_cost_usd']:.4f}"
            )
        with summary_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary, default=str) + "\n")

    print_batch_table(summary_path)
    print(f"batch_dir={batch_dir}")


def episode_cost(summary: dict) -> float:
    """agent 成本 + 两份决策日志中的 Judge / reviewer 成本。"""

    total = float(summary["agent_usage"]["estimated_cost_usd"])
    for key in ("policy_log", "reviewer_log"):
        path = Path(summary[key])
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    total += float(
                        json.loads(line).get("estimated_cost_usd") or 0.0
                    )
    return total


def print_batch_table(summary_path: Path) -> None:
    """按 policy 和是否有攻击汇总 utility、攻击成功率和成本。"""

    rows: dict[tuple[str, bool], list[dict]] = {}
    errors = 0
    with summary_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record:
                errors += 1
                continue
            key = (record["policy"], record["injection_task_id"] is not None)
            rows.setdefault(key, []).append(record)

    print()
    print(
        f"{'policy':<14}{'attack':<8}{'n':>4}{'utility':>9}{'strict':>8}"
        f"{'ASR':>7}{'cost $':>10}"
    )
    for (policy, attacked), records in sorted(rows.items()):
        n = len(records)
        utility = sum(bool(record["utility"]) for record in records) / n
        # 旧的 summary 没有 strict_utility，退回官方 utility。
        strict = sum(
            bool(record.get("strict_utility", record["utility"]))
            for record in records
        ) / n
        asr = (
            f"{sum(bool(record['attack_succeeded']) for record in records) / n:.2f}"
            if attacked
            else "-"
        )
        cost = sum(record.get("total_cost_usd", 0.0) for record in records)
        print(
            f"{policy:<14}{'yes' if attacked else 'no':<8}{n:>4}"
            f"{utility:>9.2f}{strict:>8.2f}{asr:>7}{cost:>10.4f}"
        )
    if errors:
        print(f"{errors} episode(s) failed; see batch_summary.jsonl")


if __name__ == "__main__":
    main()
