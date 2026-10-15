import unittest

from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policies import (
    POLICY_VARIANTS,
    UNLIMITED_ESCALATIONS,
    create_policy,
    escalation_budget_for,
)
from harness.policy import Decision, PolicyRequest, PolicyResult
from harness.policy_gate_executor import PolicyGateToolsExecutor
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

    def test_escalate_all_allows_reads_and_escalates_writes(self):
        policy = self._policy("escalate_all")

        self.assertEqual(
            policy.evaluate(_request("get_balance")).decision,
            Decision.ALLOW,
        )
        for tool_name in ("send_money", "update_user_info", "unknown_tool"):
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
        for variant in ("hybrid", "p1_only", "p4_only", "allow_all"):
            with self.subTest(variant=variant):
                self.assertEqual(escalation_budget_for(variant, 3), 3)
        for variant in ("escalate_all", "no_budget"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    escalation_budget_for(variant, 3),
                    UNLIMITED_ESCALATIONS,
                )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            escalation_budget_for("judge_only", 3)
        with self.assertRaises(ValueError):
            self._policy("judge_only")
        self.assertEqual(len(POLICY_VARIANTS), 6)


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
