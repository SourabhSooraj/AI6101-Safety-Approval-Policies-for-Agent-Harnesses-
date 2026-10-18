import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_agentdojo_e2e.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.policies import POLICY_VARIANTS

# Only these variants use the P6 allocator, so only they are swept over
# --allocators and --tau0-values; the others run once per k/task/attack.
P6_POLICIES = frozenset({"hybrid", "no_budget"})

# Options passed unchanged to run_agentdojo_e2e.py when given.
PASSTHROUGH_OPTIONS = (
    ("--sample-count", "Judge samples per call."),
    ("--model", "Agent model."),
    ("--agent-base-url", "OpenAI-compatible base URL for the agent."),
    ("--agent-api-key-env", "Environment variable holding the agent API key."),
    ("--judge-model", "Judge model."),
    ("--judge-base-url", "OpenAI-compatible base URL for the Judge."),
    ("--judge-api-key-env", "Environment variable holding the Judge API key."),
    ("--reviewer-model", "Simulated reviewer model."),
    ("--reviewer-base-url", "OpenAI-compatible base URL for the reviewer."),
    ("--reviewer-api-key-env", "Environment variable holding the reviewer API key."),
    ("--attack", "AgentDojo attack name used with injection tasks."),
    ("--attack-model-name", "Model name the attack text addresses."),
    ("--utility-cost", "Cost of denying a safe call (cost allocator)."),
    ("--auto-rule", "Auto decision rule for the cost allocator."),
    ("--reviewer-error-table", "Reviewer error-rate JSON (cost allocator)."),
)


def _dest(option: str) -> str:
    return option.lstrip("-").replace("-", "_")


def build_command(
    args: argparse.Namespace,
    policy: str,
    k_value: int,
    task_id: str,
    injection_task: str,
    allocator: str | None = None,
    tau0: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--policy",
        policy,
        "--k",
        str(k_value),
        "--task-id",
        task_id,
    ]
    for option, _ in PASSTHROUGH_OPTIONS:
        value = getattr(args, _dest(option))
        if value is not None:
            command += [option, str(value)]
    if allocator is not None:
        command += ["--allocator", allocator]
    if tau0 is not None:
        command += ["--tau0", str(tau0)]
    if args.agent_thinking:
        command.append("--agent-thinking")
    if injection_task != "none":
        command += ["--injection-task", injection_task]
    return command


def _parse_output(output: str, key: str) -> str:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare policy variants. Each run is a separate process, so no "
            "task state is shared between runs."
        )
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["p1_only", "p4_only", "hybrid"],
        choices=POLICY_VARIANTS,
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--task-ids", nargs="+", default=["user_task_0"])
    parser.add_argument(
        "--allocators",
        nargs="+",
        choices=("value", "cost"),
        help="P6 allocators to compare (hybrid and no_budget only).",
    )
    parser.add_argument(
        "--tau0-values",
        nargs="+",
        type=float,
        help=(
            "tau0 values to sweep (hybrid and no_budget only). Each allocator "
            "uses its own default when omitted."
        ),
    )
    parser.add_argument(
        "--injection-tasks",
        nargs="+",
        default=["none"],
        help="Injection task ids; 'none' runs without an attack.",
    )
    for option, help_text in PASSTHROUGH_OPTIONS:
        parser.add_argument(
            option,
            type=int if option == "--sample-count" else str,
            help=f"{help_text} Defaults to run_agentdojo_e2e.py's default.",
        )
    parser.add_argument(
        "--agent-thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode for the agent (DeepSeek API only).",
    )
    return parser


def allocator_settings(
    args: argparse.Namespace,
    policy: str,
) -> list[tuple[str | None, float | None]]:
    """(allocator, tau0) pairs to run for one policy; None keeps the runner default."""

    if policy not in P6_POLICIES:
        return [(None, None)]
    return [
        (allocator, tau0)
        for allocator in (args.allocators or [None])
        for tau0 in (args.tau0_values or [None])
    ]


def main() -> None:
    args = build_parser().parse_args()

    failures = 0
    for policy in args.policies:
        for allocator, tau0 in allocator_settings(args, policy):
            for k_value in args.k_values:
                for task_id in args.task_ids:
                    for injection_task in args.injection_tasks:
                        failures += run_one(
                            args,
                            policy,
                            k_value,
                            task_id,
                            injection_task,
                            allocator,
                            tau0,
                        )

    if failures:
        raise SystemExit(f"{failures} comparison runs failed.")


def run_one(
    args: argparse.Namespace,
    policy: str,
    k_value: int,
    task_id: str,
    injection_task: str,
    allocator: str | None,
    tau0: float | None,
) -> int:
    """Run one setting in its own process; returns 1 on failure."""

    command = build_command(
        args, policy, k_value, task_id, injection_task, allocator, tau0
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    run_id = _parse_output(result.stdout, "run_id")
    utility = _parse_output(result.stdout, "utility")
    attack_succeeded = _parse_output(result.stdout, "attack_succeeded")
    status = "ok" if result.returncode == 0 else "error"
    setting = ""
    if policy in P6_POLICIES:
        setting = (
            f"allocator={allocator or 'default':7} "
            f"tau0={'default' if tau0 is None else tau0:<7} "
        )
    print(
        f"{policy:12} {setting}k={k_value:<3} task={task_id:12} "
        f"attack={injection_task:17} utility={utility:5} "
        f"attack_succeeded={attack_succeeded:5} "
        f"run_id={run_id} {status}"
    )
    if result.returncode == 0:
        return 0
    if result.stderr:
        print(result.stderr.strip())
    return 1


if __name__ == "__main__":
    main()
