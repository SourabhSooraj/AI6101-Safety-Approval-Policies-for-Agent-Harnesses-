import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.task_suite.load_suites import get_suite

from harness.budget_allocator import BudgetAllocator
from harness.decision_logger import DecisionLogger
from harness.deepseek_agent_llm import DeepSeekAgentLLM
from harness.deepseek_judge_client import DeepSeekJudgeClient
from harness.llm_judge import LLMJudge
from harness.policy_context import PolicyContextInitializer
from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.simulated_reviewer import SimulatedReviewer
from harness.tool_metadata import BANKING_TOOL_METADATA


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real AgentDojo banking task through the policy gate."
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--task-id", default="user_task_0")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--expected-remaining-risky-calls",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = REPO_ROOT / "logs" / "agentdojo_e2e" / run_id
    policy_log_path = log_dir / "policy_decisions.jsonl"
    reviewer_log_path = log_dir / "reviewer_decisions.jsonl"

    judge = LLMJudge(
        client=DeepSeekJudgeClient(model=args.model),
        sample_count=args.sample_count,
        model_name=args.model,
    )
    reviewer = SimulatedReviewer(
        client=DeepSeekJudgeClient(model=args.model),
        model_name=f"{args.model}-reviewer",
    )
    policy_engine = PolicyEngine(
        judge=judge,
        allocator=BudgetAllocator(tau0=1.0),
        decision_logger=DecisionLogger(
            policy_log_path,
            run_id=run_id,
            extra_context={
                "suite": args.suite,
                "attack_mode": "none",
                "k": args.k,
            },
        ),
    )
    gate = PolicyGateToolsExecutor(
        policy_engine=policy_engine,
        tool_metadata=BANKING_TOOL_METADATA,
        task_id=args.task_id,
        escalation_budget=args.k,
        reviewer=reviewer,
        review_logger=DecisionLogger(
            reviewer_log_path,
            run_id=run_id,
            extra_context={
                "suite": args.suite,
                "attack_mode": "none",
                "k": args.k,
            },
        ),
    )

    agent_client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=120.0,
    )
    agent_llm = DeepSeekAgentLLM(
        agent_client,
        args.model,
        temperature=0.0,
    )
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(None)),
            PolicyContextInitializer(
                task_id=args.task_id,
                escalation_budget=args.k,
                expected_remaining_risky_calls=(
                    args.expected_remaining_risky_calls
                ),
            ),
            InitQuery(),
            agent_llm,
            ToolsExecutionLoop([gate, agent_llm], max_iters=10),
        ]
    )
    pipeline.name = f"{args.model}-gated-policy"

    suite = get_suite(args.benchmark_version, args.suite)
    task = suite.get_user_task_by_id(args.task_id)
    utility, security = suite.run_task_with_pipeline(
        pipeline,
        task,
        injection_task=None,
        injections={},
    )

    print(f"run_id={run_id}")
    print(f"task_id={args.task_id}")
    print(f"utility={utility}")
    print(f"security={security}")
    print(f"policy_log={policy_log_path}")
    print(f"reviewer_log={reviewer_log_path}")


if __name__ == "__main__":
    main()
