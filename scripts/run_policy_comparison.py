import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_agentdojo_e2e.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.policies import POLICY_VARIANTS


def _parse_output(output: str, key: str) -> str:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return "unknown"


def main() -> None:
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
    parser.add_argument(
        "--sample-count",
        type=int,
        help="Judge samples per call. Defaults to run_agentdojo_e2e.py's default.",
    )
    args = parser.parse_args()

    failures = 0
    for policy in args.policies:
        for k_value in args.k_values:
            for task_id in args.task_ids:
                for injection_task in args.injection_tasks:
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
                    if args.sample_count is not None:
                        command += ["--sample-count", str(args.sample_count)]
                    if injection_task != "none":
                        command += ["--injection-task", injection_task]

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
