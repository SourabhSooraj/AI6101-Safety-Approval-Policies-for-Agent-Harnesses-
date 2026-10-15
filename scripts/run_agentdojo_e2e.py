import argparse
import json
import os
import sys
from dataclasses import asdict
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
from agentdojo.attacks import load_attack
from agentdojo.task_suite.load_suites import get_suite

import harness.attacks  # noqa: F401 - 注册 important_instructions_deepseek
from harness.baselines import (
    POLICIES,
    build_policy_engine,
    escalation_budget_for,
)
from harness.budget_allocator import BudgetAllocator
from harness.decision_logger import DecisionLogger
from harness.deepseek_agent_llm import DeepSeekAgentLLM
from harness.deepseek_judge_client import DeepSeekJudgeClient
from harness.llm_judge import LLMJudge
from harness.policy_context import PolicyContextInitializer
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.q_estimator import load_q_table
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
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Judge samples per call. Disagreement detection needs at least 2.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="full",
        help=(
            "full: P1 -> Judge -> P6. allow_all: no gate. escalate_all: every "
            "non-read call goes to the reviewer, unlimited budget. p1_only: P1, "
            "with undecided calls sent to the reviewer. no_budget: full with "
            "unlimited budget."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Escalation budget per task. Ignored by escalate_all and no_budget.",
    )
    parser.add_argument(
        "--expected-remaining-risky-calls",
        type=int,
        default=3,
    )
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument(
        "--q-table",
        help="Optional q_table.json from build_q_table.py. Missing tools use q=1.0.",
    )
    parser.add_argument(
        "--injection-task",
        help="AgentDojo injection task id, e.g. injection_task_0. Omit for no attack.",
    )
    parser.add_argument(
        "--attack",
        default="important_instructions_deepseek",
        help="AgentDojo attack name used with --injection-task.",
    )
    parser.add_argument(
        "--reviewer-model",
        help="Simulated reviewer model. Defaults to --model.",
    )
    parser.add_argument(
        "--reviewer-base-url",
        default="https://api.deepseek.com",
        help="OpenAI-compatible base URL for the reviewer model.",
    )
    parser.add_argument(
        "--reviewer-api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable holding the reviewer API key.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")
    reviewer_api_key = os.environ.get(args.reviewer_api_key_env)
    if not reviewer_api_key:
        raise SystemExit(f"{args.reviewer_api_key_env} is not set.")

    reviewer_model = args.reviewer_model or args.model
    if args.sample_count < 2:
        print(
            "WARNING: --sample-count < 2 disables Judge disagreement detection.",
            file=sys.stderr,
        )
    if reviewer_model == args.model:
        print(
            "WARNING: the reviewer uses the same model as the agent and Judge; "
            "use --reviewer-model with a different model family for final experiments.",
            file=sys.stderr,
        )

    q_table = load_q_table(args.q_table) if args.q_table else {}
    attack_mode = args.attack if args.injection_task else "none"
    escalation_budget = escalation_budget_for(args.policy, args.k)
    suite = get_suite(args.benchmark_version, args.suite)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = REPO_ROOT / "logs" / "agentdojo_e2e" / run_id
    policy_log_path = log_dir / "policy_decisions.jsonl"
    reviewer_log_path = log_dir / "reviewer_decisions.jsonl"
    summary_path = log_dir / "run_summary.json"
    log_context = {
        "suite": args.suite,
        "policy": args.policy,
        "attack_mode": attack_mode,
        "injection_task_id": args.injection_task,
        "k": escalation_budget,
        "sample_count": args.sample_count,
        "tau0": args.tau0,
        "reviewer_model": reviewer_model,
    }

    judge = LLMJudge(
        client=DeepSeekJudgeClient(model=args.model),
        sample_count=args.sample_count,
        model_name=args.model,
    )
    reviewer = SimulatedReviewer(
        client=DeepSeekJudgeClient(
            model=reviewer_model,
            base_url=args.reviewer_base_url,
            api_key=reviewer_api_key,
        ),
        model_name=f"{reviewer_model}-reviewer",
    )
    policy_engine = build_policy_engine(
        args.policy,
        judge=judge,
        allocator=BudgetAllocator(tau0=args.tau0),
        tool_names=[function.name for function in suite.tools],
        q_table=q_table,
        decision_logger=DecisionLogger(
            policy_log_path,
            run_id=run_id,
            extra_context=log_context,
        ),
    )
    gate = PolicyGateToolsExecutor(
        policy_engine=policy_engine,
        tool_metadata=BANKING_TOOL_METADATA,
        task_id=args.task_id,
        escalation_budget=escalation_budget,
        reviewer=reviewer,
        review_logger=DecisionLogger(
            reviewer_log_path,
            run_id=run_id,
            extra_context=log_context,
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
                escalation_budget=escalation_budget,
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

    task = suite.get_user_task_by_id(args.task_id)
    injection_task = None
    injections: dict[str, str] = {}
    if args.injection_task:
        injection_task = suite.get_injection_task_by_id(args.injection_task)
        attack = load_attack(args.attack, suite, pipeline)
        injections = attack.attack(task, injection_task)

    utility, security = suite.run_task_with_pipeline(
        pipeline,
        task,
        injection_task=injection_task,
        injections=injections,
    )

    # AgentDojo 的 security=True 表示注入任务成功；无攻击时恒为 True，没有意义。
    attack_succeeded = security if injection_task is not None else None
    agent_usage = agent_llm.total_usage
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": args.task_id,
                **log_context,
                "q_table_path": args.q_table,
                "utility": utility,
                "attack_succeeded": attack_succeeded,
                # Judge 成本在 policy 日志，reviewer 成本在 reviewer 日志。
                "agent_usage": asdict(agent_usage),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"run_id={run_id}")
    print(f"task_id={args.task_id}")
    print(f"policy={args.policy}")
    print(f"attack_mode={attack_mode}")
    print(f"injection_task={args.injection_task}")
    print(f"utility={utility}")
    print(f"attack_succeeded={attack_succeeded}")
    print(f"agent_input_tokens={agent_usage.input_tokens}")
    print(f"agent_output_tokens={agent_usage.output_tokens}")
    print(f"agent_estimated_cost_usd={agent_usage.estimated_cost_usd:.8f}")
    print(f"policy_log={policy_log_path}")
    print(f"reviewer_log={reviewer_log_path}")
    print(f"run_summary={summary_path}")


if __name__ == "__main__":
    main()
