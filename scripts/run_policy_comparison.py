import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_agentdojo_e2e.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.policies import POLICY_VARIANTS

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
)


def _dest(option: str) -> str:
    return option.lstrip("-").replace("-", "_")


def build_command(
    args: argparse.Namespace,
    policy: str,
    k_value: int,
    task_id: str,
    injection_task: str,
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


def main() -> None:
    args = build_parser().parse_args()

    failures = 0
    for policy in args.policies:
        for k_value in args.k_values:
            for task_id in args.task_ids:
                for injection_task in args.injection_tasks:
                    command = build_command(
                        args, policy, k_value, task_id, injection_task
                    )

                    result = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                    )
                    run_id = _parse_output(result.stdout, "run_id")
                    utility = _parse_output(result.stdout, "utility")
                    attack_succeeded = _parse_output(
                        result.stdout,
                        "attack_succeeded",
                    )
                    status = "ok" if result.returncode == 0 else "error"
                    print(
                        f"{policy:12} k={k_value:<3} task={task_id:12} "
                        f"attack={injection_task:17} utility={utility:5} "
                        f"attack_succeeded={attack_succeeded:5} "
                        f"run_id={run_id} {status}"
                    )
                    if result.returncode != 0:
                        failures += 1
                        if result.stderr:
                            print(result.stderr.strip())

    if failures:
        raise SystemExit(f"{failures} comparison runs failed.")


if __name__ == "__main__":
    main()
