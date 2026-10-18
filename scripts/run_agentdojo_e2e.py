import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.baselines import POLICIES
from harness.episode import (
    add_episode_args,
    config_from_args,
    print_config_warnings,
    run_episode,
    system_message as _system_message,  # noqa: F401 - 测试从这里导入
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real AgentDojo banking task through the policy gate."
    )
    add_episode_args(parser)
    parser.add_argument("--task-id", default="user_task_0")
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="full",
        help=(
            "full/hybrid: P1 -> Judge -> P6. p4_only: Judge thresholds only. "
            "p1_only: P1 only. allow_all, escalate_all, and no_budget are baselines."
        ),
    )
    parser.add_argument(
        "--injection-task",
        help="AgentDojo injection task id, e.g. injection_task_0. Omit for no attack.",
    )
    args = parser.parse_args()

    config = config_from_args(args)
    print_config_warnings(config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = run_episode(
        config,
        task_id=args.task_id,
        policy=args.policy,
        injection_task_id=args.injection_task,
        run_id=run_id,
        log_dir=REPO_ROOT / "logs" / "agentdojo_e2e" / run_id,
    )

    agent_usage = summary["agent_usage"]
    print(f"run_id={run_id}")
    print(f"task_id={args.task_id}")
    print(f"policy={args.policy}")
    print(f"policy_variant={summary['policy_variant']}")
    print(f"provider={config.provider}")
    print(f"agent_model={config.agent_model}")
    print(f"judge_model={config.judge_model}")
    print(f"reviewer_model={config.reviewer_model}")
    print(f"attack_mode={summary['attack_mode']}")
    print(f"injection_task={args.injection_task}")
    print(f"utility={summary['utility']}")
    print(f"attack_succeeded={summary['attack_succeeded']}")
    print(f"agent_input_tokens={agent_usage['input_tokens']}")
    print(f"agent_output_tokens={agent_usage['output_tokens']}")
    print(f"agent_estimated_cost_usd={agent_usage['estimated_cost_usd']:.8f}")
    print(f"policy_log={summary['policy_log']}")
    print(f"reviewer_log={summary['reviewer_log']}")
    print(f"agentdojo_trace_dir={summary['agentdojo_trace_dir']}")
    print(f"run_summary={Path(summary['policy_log']).parent / 'run_summary.json'}")


if __name__ == "__main__":
    main()
