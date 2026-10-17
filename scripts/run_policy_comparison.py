import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_agentdojo_e2e.py"


def _parse_output(output: str, key: str) -> str:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare P1-only, P4-only, and Hybrid policies."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["p1_only", "p4_only", "hybrid"],
        choices=["p1_only", "p4_only", "hybrid"],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--task-ids", nargs="+", default=["user_task_0"])
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()

    failures = 0
    for policy in args.policies:
        for k_value in args.k_values:
            for task_id in args.task_ids:
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--policy",
                    policy,
                    "--k",
                    str(k_value),
                    "--task-id",
                    task_id,
                    "--sample-count",
                    str(args.sample_count),
                ]
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                utility = _parse_output(result.stdout, "utility")
                security = _parse_output(result.stdout, "security")
                status = "ok" if result.returncode == 0 else "error"
                print(
                    f"{policy:8} k={k_value:<3} task={task_id:12} "
                    f"utility={utility:5} security={security:5} {status}"
                )
                if result.returncode != 0:
                    failures += 1
                    if result.stderr:
                        print(result.stderr.strip())

    if failures:
        raise SystemExit(f"{failures} comparison runs failed.")


if __name__ == "__main__":
    main()
