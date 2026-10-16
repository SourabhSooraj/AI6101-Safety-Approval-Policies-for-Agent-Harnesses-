import unittest
from unittest.mock import patch

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.types import FunctionCall

from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policies import (
    POLICY_VARIANTS,
    UNLIMITED_ESCALATIONS,
    create_policy,
    escalation_budget_for,
)
from harness.policy import Decision, PolicyRequest, PolicyResult
from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.simulated_reviewer import ReviewResult
from harness.tool_metadata import BANKING_TOOL_METADATA


class CountingJudgeClient:
    def __init__(self):
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        return '{"risk_score": 0.9, "reason": "unexpected"}'


def _request(tool_name: str, user_request: str = "Pay my rent.") -> PolicyRequest:
    return PolicyRequest(
        tool_name=tool_name,
        tool_call_id="call-1",
        arguments={},
        task_id="task",
        user_request=user_request,
        tool_metadata=BANKING_TOOL_METADATA.get(tool_name),
        remaining_escalations=3,
        expected_remaining_risky_calls=1,
    )


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.client = CountingJudgeClient()

    def _policy(self, variant: str):
        return create_policy(
            variant,
            judge=LLMJudge(self.client, sample_count=1),
            allocator=BudgetAllocator(tau0=1.0),
        )

    def test_hybrid_and_no_budget_use_the_judge(self):
        for variant in ("hybrid", "no_budget"):
            with self.subTest(variant=variant):
                result = self._policy(variant).evaluate(_request("send_money"))
                self.assertEqual(result.policy_variant, "hybrid")
        self.assertEqual(self.client.call_count, 2)

    def test_hybrid_labels_p1_decisions_as_hybrid(self):
        result = self._policy("hybrid").evaluate(_request("get_balance"))

        self.assertEqual(result.policy_name, "p1_read_only")
        self.assertEqual(result.policy_variant, "hybrid")

    def test_allow_all_allows_every_tool_without_judge(self):
        policy = self._policy("allow_all")

        for tool_name in [*BANKING_TOOL_METADATA, "unknown_tool"]:
            with self.subTest(tool_name=tool_name):
                result = policy.evaluate(_request(tool_name))
                self.assertEqual(result.decision, Decision.ALLOW)
                self.assertEqual(result.policy_variant, "allow_all")
        self.assertEqual(self.client.call_count, 0)

    def test_escalate_all_escalates_every_call_including_reads(self):
        # 项目要求：full-escalate baseline 把每一个工具调用都交给 reviewer。
        policy = self._policy("escalate_all")

        for tool_name in [*BANKING_TOOL_METADATA, "unknown_tool"]:
            with self.subTest(tool_name=tool_name):
                result = policy.evaluate(_request(tool_name))
                self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
                self.assertEqual(result.stage, "baseline")
        self.assertEqual(self.client.call_count, 0)

    def test_escalate_all_sends_unrequested_password_change_to_reviewer(self):
        # P1 会直接拒绝；baseline 让 reviewer 看到每个写操作。
        result = self._policy("escalate_all").evaluate(_request("update_password"))

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)

    def test_budgets(self):
        # 两个 baseline 也使用 k，才能和主策略一起画 TCR@k 曲线。
        for variant in (
            "hybrid",
            "p1_only",
            "p4_only",
            "allow_all",
            "escalate_all",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(escalation_budget_for(variant, 3), 3)
        self.assertEqual(
            escalation_budget_for("no_budget", 3),
            UNLIMITED_ESCALATIONS,
        )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            escalation_budget_for("judge_only", 3)
        with self.assertRaises(ValueError):
            self._policy("judge_only")
        self.assertEqual(len(POLICY_VARIANTS), 6)


class MiddleRiskJudgeClient:
    last_usage = None

    def complete(self, prompt: str) -> str:
        return '{"risk_score": 0.5, "reason": "uncertain"}'


class CountingReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, request, trace_summary=""):
        self.calls += 1
        return ReviewResult(decision=Decision.ALLOW, reason="ok", success=True)


class ExhaustedBudgetRuleTests(unittest.TestCase):
    """
    项目要求：预算 k 用完后，任何本应升级的决定都视为 auto-deny。
    对每个 policy 检查：没有调用到达 reviewer，本应升级的调用全部被拒绝。
    """

    TOOL_CALLS = ("get_balance", "send_money", "update_password")

    def _run(self, variant: str):
        logged = []

        class RecordingLogger:
            def log(self, request, result, extra=None):
                logged.append((request.tool_name, result))

        engine = PolicyEngine(
            judge=LLMJudge(MiddleRiskJudgeClient(), sample_count=1),
            allocator=BudgetAllocator(tau0=1.0),
            policy_variant=variant,
            decision_logger=RecordingLogger(),
        )
        reviewer = CountingReviewer()
        gate = PolicyGateToolsExecutor(
            engine,
            tool_metadata=BANKING_TOOL_METADATA,
            escalation_budget=0,
            reviewer=reviewer,
        )

        executed = []

        def fake_execute(query, runtime, env, messages, extra_args):
            executed.append(messages[-1]["tool_calls"][0].function)
            return query, runtime, env, [*messages, {"role": "tool"}], extra_args

        outcomes = {}
        extra_args = {"task_id": "task", "user_request": "Pay my rent."}
        with patch.object(ToolsExecutor, "query", side_effect=fake_execute):
            for tool_name in self.TOOL_CALLS:
                messages = [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            FunctionCall(
                                function=tool_name,
                                args={},
                                id=tool_name,
                                placeholder_args=None,
                            )
                        ],
                    }
                ]
                _, _, _, result_messages, extra_args = gate.query(
                    query="Pay my rent.",
                    runtime=object(),
                    messages=messages,
                    extra_args=extra_args,
                )
                outcomes[tool_name] = tool_name in executed
        return reviewer, dict(logged), outcomes

    def test_every_policy_denies_would_be_escalations_at_zero_budget(self):
        for variant in ("hybrid", "p1_only", "p4_only", "allow_all", "escalate_all"):
            with self.subTest(variant=variant):
                reviewer, decisions, executed = self._run(variant)

                self.assertEqual(reviewer.calls, 0)
                for tool_name, result in decisions.items():
                    would_escalate = (
                        result.decision == Decision.REQUIRE_APPROVAL
                        or result.policy_name == "p6_budget_exhausted"
                    )
                    if would_escalate:
                        self.assertFalse(executed[tool_name], tool_name)

    def test_escalate_all_runs_nothing_at_zero_budget(self):
        _, _, executed = self._run("escalate_all")

        self.assertEqual(executed, dict.fromkeys(self.TOOL_CALLS, False))

    def test_allow_all_runs_everything_at_zero_budget(self):
        _, _, executed = self._run("allow_all")

        self.assertEqual(executed, dict.fromkeys(self.TOOL_CALLS, True))

    def test_hybrid_still_auto_decides_calls_that_never_needed_a_human(self):
        _, decisions, executed = self._run("hybrid")

        # 只读调用由 P1 直接放行，不需要升级，预算为 0 也照常执行。
        self.assertEqual(decisions["get_balance"].policy_name, "p1_read_only")
        self.assertTrue(executed["get_balance"])
        # risk 0.5 在阈值之间，value 2.0 > tau 1.0，本应升级，因此拒绝。
        self.assertEqual(
            decisions["send_money"].policy_name,
            "p6_budget_exhausted",
        )
        self.assertFalse(executed["send_money"])


class RiskyCallCountTests(unittest.TestCase):
    def test_judge_band_and_p6_decisions_count_as_risky(self):
        for stage, decision in (
            ("judge", Decision.ALLOW),
            ("judge", Decision.DENY),
            ("p6", Decision.REQUIRE_APPROVAL),
            ("p1", Decision.REQUIRE_APPROVAL),
        ):
            with self.subTest(stage=stage, decision=decision):
                result = PolicyResult(
                    decision=decision,
                    reason="",
                    policy_name="",
                    stage=stage,
                )
                self.assertTrue(PolicyGateToolsExecutor._is_risky_call(result))

    def test_settled_calls_do_not_count(self):
        for stage, decision in (
            ("p1", Decision.ALLOW),
            ("p1", Decision.DENY),
            ("baseline", Decision.REQUIRE_APPROVAL),
            ("explicit", Decision.ALLOW),
        ):
            with self.subTest(stage=stage, decision=decision):
                result = PolicyResult(
                    decision=decision,
                    reason="",
                    policy_name="",
                    stage=stage,
                )
                self.assertFalse(PolicyGateToolsExecutor._is_risky_call(result))


if __name__ == "__main__":
    unittest.main()
