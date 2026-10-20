import argparse
import itertools
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.episode import (
    add_episode_args,
    config_from_args,
    print_config_warnings,
)
from harness.policies import POLICY_VARIANTS
from scripts.run_agentdojo_e2e import build_parser as build_run_parser

RUNNER = REPO_ROOT / "scripts" / "run_agentdojo_e2e.py"
# dev 用于调参和 q table，test 只用于最终测试。划分由
# scripts/make_dev_test_split.py 生成（分层抽样，近似重复的任务放在同一边）。
SPLIT_NAMES = ("dev", "test")
DEFAULT_SPLIT_FILE = REPO_ROOT / "configs" / "banking_dev_test_split.json"
NO_ATTACK = "none"
RUN_LOG_NAME = "run.log"
# setup 只能设置 policy 一侧的参数；模型、服务器和攻击由整批共用，避免一批里混用模型。
SETUP_KEYS = frozenset(
    {
        "policy",
        "k",
        "expected_remaining_risky_calls",
        "t_low",
        "t_high",
        "allocator",
        "tau0",
        "q_table",
        "utility_cost",
        "auto_rule",
        "reviewer_error_table",
        "sample_count",
    }
)
# 续跑同一批时必须一致的设置：q table 和 TCR 都依赖这些模型和 agent 的 system prompt。
MODEL_CONFIG_KEYS = (
    "auto_complete",
    "benchmark_version",
    "suite",
    "provider",
    "local_server",
    "agent_model",
    "agent_base_url",
    "agent_thinking",
    "judge_model",
    "judge_base_url",
    "judge_temperature",
    "reviewer_model",
    "reviewer_base_url",
    "attack",
    "attack_model_name",
)
SETUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# 值为文件路径的参数：子进程在 REPO_ROOT 下运行，先转成绝对路径。
PATH_KEYS = frozenset({"q_table", "reviewer_error_table"})


@dataclass(frozen=True)
class Setup:
    """一组 policy 设置；episode 目录名为 <task>__<name>__<injection>。"""

    name: str
    policy: str
    options: dict[str, Any] = field(default_factory=dict)


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


def load_setups(path: Path) -> list[Setup]:
    """
    读取 setup 文件：{"setups": [{"name": ..., "policy": ..., "k": ...}, ...]}。
    键用 episode 参数名（如 t_low、tau0）；拼错或不允许的键直接报错。
    """

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    entries = data.get("setups") if isinstance(data, dict) else None
    if not entries or not isinstance(entries, list):
        raise ValueError(f"{path} needs a non-empty 'setups' list")

    setups = []
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"setup {number} in {path} must be an object")
        unknown = sorted(set(entry) - SETUP_KEYS - {"name"})
        if unknown:
            raise ValueError(
                f"setup {number} in {path} has keys a setup cannot set: "
                f"{', '.join(unknown)} (allowed: {', '.join(sorted(SETUP_KEYS))})"
            )
        policy = entry.get("policy")
        if policy not in POLICY_VARIANTS:
            raise ValueError(
                f"setup {number} in {path} needs a policy from "
                f"{', '.join(POLICY_VARIANTS)}"
            )
        name = str(entry.get("name", policy))
        if not SETUP_NAME.match(name) or "__" in name:
            raise ValueError(
                f"setup name {name!r} must use letters, digits, '.', '-' or "
                "single '_' (it becomes part of a folder name)"
            )
        options = {
            key: value
            for key, value in entry.items()
            if key not in {"name", "policy"}
        }
        setups.append(Setup(name=name, policy=policy, options=options))

    names = [setup.name for setup in setups]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path} uses these setup names twice: {duplicates}")
    return setups


def _episode_actions() -> dict[str, argparse.Action]:
    parser = argparse.ArgumentParser()
    add_episode_args(parser)
    return {action.dest: action for action in parser._actions if action.dest != "help"}


def _flag_tokens(action: argparse.Action, value: Any) -> list[str]:
    option = action.option_strings[0]
    if isinstance(action, argparse._StoreTrueAction):
        return [option] if value else []
    if action.dest in PATH_KEYS:
        # 相对路径按运行命令的目录解析，而不是子进程的 REPO_ROOT。
        value = Path(value).expanduser().resolve()
    return [option, str(value)]


def given_episode_flags(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    """
    命令行上实际给出的 episode 参数，转成子进程的参数。参数列表来自
    add_episode_args，以后新增的模型参数也会自动传下去。
    """

    actions = _episode_actions()
    # 默认值全部换成 SUPPRESS，再解析一次，得到的只有用户写出的参数。
    saved = {action: action.default for action in parser._actions}
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    try:
        given = vars(parser.parse_args(argv))
    finally:
        for action, default in saved.items():
            action.default = default

    tokens = []
    for dest, action in actions.items():
        if dest in given:
            tokens += _flag_tokens(action, given[dest])
    return tokens


def setup_flags(setup: Setup) -> list[str]:
    actions = _episode_actions()
    tokens = ["--policy", setup.policy]
    for key, value in setup.options.items():
        tokens += _flag_tokens(actions[key], value)
    return tokens


def episode_name(task_id: str, setup: Setup, injection: str) -> str:
    return f"{task_id}__{setup.name}__{injection}"


def run_command(
    model_flags: list[str],
    setup: Setup,
    task_id: str,
    injection: str,
    log_dir: Path,
    run_id: str,
) -> list[str]:
    """一个 episode 的子进程命令；setup 的参数放在后面，覆盖同名的全局参数。"""

    command = [
        sys.executable,
        str(RUNNER),
        *model_flags,
        *setup_flags(setup),
        "--task-id",
        task_id,
        "--log-dir",
        str(log_dir),
        "--run-id",
        run_id,
    ]
    if injection != NO_ATTACK:
        command += ["--injection-task", injection]
    return command


def check_setups(model_flags: list[str], setups: list[Setup], *, resolve: bool) -> None:
    """
    用单次运行脚本的解析器检查每个 setup（类型、取值范围）；resolve=True 时还
    检查 config_from_args 中的互斥选项，避免跑到一半才失败。
    """

    run_parser = build_run_parser()
    for setup in setups:
        argv = [*model_flags, *setup_flags(setup)]
        try:
            # argparse 出错时调用 sys.exit；改为说明是哪个 setup。
            args = run_parser.parse_args(argv)
        except SystemExit as error:
            raise ValueError(
                f"setup {setup.name!r} has an invalid value: {shlex.join(argv)}"
            ) from error
        if resolve:
            try:
                config_from_args(args)
            except SystemExit as error:
                raise ValueError(f"setup {setup.name!r}: {error}") from error


def check_same_settings(path: Path, config_record: dict[str, Any]) -> None:
    """
    续跑时模型、传给每次运行的参数和 setup 都必须与上次相同，否则同一批的
    文件夹会混用不同的模型或 gate 设置（例如一半 k=3、一半 k=1）。
    旧的 batch_config.json 没有 flags / setups 字段时只检查模型。
    """

    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    changed = [
        f"{key}: {previous.get(key)!r} -> {config_record.get(key)!r}"
        for key in MODEL_CONFIG_KEYS
        if previous.get(key) != config_record.get(key)
    ]
    for key in ("flags", "setups"):
        if key in previous and previous[key] != config_record[key]:
            changed.append(f"{key}: {previous[key]!r} -> {config_record[key]!r}")
    if changed:
        raise ValueError(
            f"{path.parent.name} was started with other settings; "
            "use a new --batch-id. Changed: " + "; ".join(changed)
        )


def last_error_line(log_path: Path, returncode: int) -> str:
    """子进程日志的最后一行非空输出（通常是异常），读不到时给出退出码。"""

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return f"exit code {returncode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every (task, setup, injection task) combination, each in its "
            "own process, and write one log folder per episode plus a batch "
            "summary. Model, server and attack flags apply to every run."
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
    setups = parser.add_mutually_exclusive_group()
    setups.add_argument(
        "--setups",
        type=Path,
        help=(
            "JSON file of named setups, e.g. configs/setups/policy_comparison.json. "
            "Each setup picks a policy and may set "
            + ", ".join(sorted(SETUP_KEYS - {"policy"}))
            + "."
        ),
    )
    setups.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_VARIANTS,
        help="Shortcut for one setup per policy, named after it (default: hybrid).",
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
        help="Print the episodes and their commands, without calling any model.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    try:
        if args.split:
            task_ids = load_split(
                args.split_file,
                args.split,
                suite=args.suite,
                benchmark_version=args.benchmark_version,
            )
        else:
            task_ids = args.task_ids
        if args.setups:
            setups = load_setups(args.setups)
        else:
            setups = [
                Setup(name=policy, policy=policy)
                for policy in args.policies or ["hybrid"]
            ]
        model_flags = given_episode_flags(parser, argv)
        check_setups(model_flags, setups, resolve=not args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    batch_dir = REPO_ROOT / "logs" / "agentdojo_batch" / batch_id
    episodes = list(itertools.product(task_ids, setups, args.injection_tasks))

    if args.dry_run:
        for task_id, setup, injection in episodes:
            name = episode_name(task_id, setup, injection)
            print(f"{task_id} setup={setup.name} injection={injection}")
            command = run_command(
                model_flags,
                setup,
                task_id,
                injection,
                batch_dir / name,
                f"{batch_id}/{name}",
            )
            print(f"    {shlex.join(command)}")
        print(f"{len(episodes)} episodes")
        return

    config = config_from_args(args)
    print_config_warnings(config)
    config_record = {
        **{
            key: value
            for key, value in asdict(config).items()
            if not key.endswith("api_key")
        },
        # 每次运行实际收到的参数；续跑时必须相同。
        "flags": model_flags,
        "setups": [asdict(setup) for setup in setups],
    }
    config_path = batch_dir / "batch_config.json"
    try:
        check_same_settings(config_path, config_record)
    except ValueError as error:
        parser.error(str(error))

    batch_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                **config_record,
                "split": args.split,
                "split_file": str(args.split_file) if args.split else None,
                "task_ids": task_ids,
                "policies": sorted({setup.policy for setup in setups}),
                "injection_tasks": args.injection_tasks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = batch_dir / "batch_summary.jsonl"

    for index, (task_id, setup, injection) in enumerate(episodes, start=1):
        name = episode_name(task_id, setup, injection)
        episode_dir = batch_dir / name
        label = f"[{index}/{len(episodes)}] {name}"
        if (episode_dir / "run_summary.json").exists():
            print(f"{label} already done, skipping")
            continue

        # 上次中断的 episode 留下的日志会被追加写入，先删掉以免重复计数。
        for stale_log in [*episode_dir.glob("*.jsonl"), episode_dir / RUN_LOG_NAME]:
            stale_log.unlink(missing_ok=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        print(f"{label} running", flush=True)
        summary = run_one(
            model_flags,
            setup,
            task_id,
            injection,
            episode_dir,
            f"{batch_id}/{name}",
        )
        if "error" in summary:
            print(f"{label} failed: {summary['error']}")
        else:
            print(
                f"{label} utility={summary['utility']} "
                f"strict_utility={summary['strict_utility']} "
                f"attack_succeeded={summary['attack_succeeded']} "
                f"cost=${summary['total_cost_usd']:.4f}"
            )
        write_summary_line(summary_path, summary)

    print_batch_table(summary_path)
    print(f"batch_dir={batch_dir}")


def run_one(
    model_flags: list[str],
    setup: Setup,
    task_id: str,
    injection: str,
    episode_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """
    在单独的进程中运行一个 episode，输出写入 run.log。返回 batch_summary 的
    一行：成功时为 run_summary 加 setup 和 total_cost_usd，失败时为错误记录。
    """

    command = run_command(model_flags, setup, task_id, injection, episode_dir, run_id)
    log_path = episode_dir / RUN_LOG_NAME
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    summary_file = episode_dir / "run_summary.json"
    if result.returncode == 0 and summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        return {
            **summary,
            "setup": setup.name,
            "total_cost_usd": episode_cost(summary),
        }
    return {
        "task_id": task_id,
        "policy": setup.policy,
        "injection_task_id": None if injection == NO_ATTACK else injection,
        "error": last_error_line(log_path, result.returncode),
        "setup": setup.name,
    }


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


def summary_key(record: dict[str, Any]) -> tuple[str, str, str | None]:
    """batch_summary.jsonl 中一个 episode 的标识；旧记录没有 setup 时用 policy。"""

    return (
        record["task_id"],
        record.get("setup", record["policy"]),
        record["injection_task_id"],
    )


def read_summary_lines(summary_path: Path) -> list[dict[str, Any]]:
    """每个 episode 只保留最后一行：重跑成功后，旧的错误记录不再计入。"""

    latest: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    with summary_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                latest.pop(summary_key(record), None)
                latest[summary_key(record)] = record
    return list(latest.values())


def write_summary_line(summary_path: Path, record: dict[str, Any]) -> None:
    """写入一个 episode 的结果，并删除同一 episode 之前的记录。"""

    records = read_summary_lines(summary_path) if summary_path.exists() else []
    records = [old for old in records if summary_key(old) != summary_key(record)]
    records.append(record)
    summary_path.write_text(
        "".join(json.dumps(item, default=str) + "\n" for item in records),
        encoding="utf-8",
    )


def print_batch_table(summary_path: Path) -> None:
    """按 setup（旧记录没有 setup 时用 policy）和是否有攻击汇总 utility、攻击成功率和成本。"""

    rows: dict[tuple[str, bool], list[dict]] = {}
    errors = 0
    for record in read_summary_lines(summary_path):
        if "error" in record:
            errors += 1
            continue
        setup = record.get("setup", record["policy"])
        key = (setup, record["injection_task_id"] is not None)
        rows.setdefault(key, []).append(record)

    print()
    print(
        f"{'setup':<20}{'attack':<8}{'n':>4}{'utility':>9}{'strict':>8}"
        f"{'ASR':>7}{'cost $':>10}"
    )
    for (setup, attacked), records in sorted(rows.items()):
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
            f"{setup:<20}{'yes' if attacked else 'no':<8}{n:>4}"
            f"{utility:>9.2f}{strict:>8.2f}{asr:>7}{cost:>10.4f}"
        )
    if errors:
        print(f"{errors} episode(s) failed; see batch_summary.jsonl")


if __name__ == "__main__":
    main()
