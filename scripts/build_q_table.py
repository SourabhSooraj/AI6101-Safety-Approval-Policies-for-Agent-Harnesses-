import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.q_estimator import estimate_q_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate q_table from dev JSONL logs."
    )
    parser.add_argument("log_path", help="Path to dev decisions JSONL.")
    parser.add_argument(
        "--group-by",
        default="tool_name",
        help="Record field used to group similar calls.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum reviewer samples required for a group.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    q_table = estimate_q_table(
        args.log_path,
        group_by=args.group_by,
        min_samples=args.min_samples,
    )
    serialized = json.dumps(q_table, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {len(q_table)} q values to {output_path}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
