import unittest

from harness.baselines import (
    POLICIES,
    UNLIMITED_ESCALATIONS,
    build_policy_engine,
    escalation_budget_for,
)
from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policy import Decision, PolicyRequest
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

    def _engine(self, policy: str):
        return build_policy_engine(
            policy,
            judge=LLMJudge(self.client, sample_count=1),
            allocator=BudgetAllocator(tau0=1.0),
            tool_names=BANKING_TOOL_METADATA,
        )

    def test_full_uses_judge_and_allocator(self):
        result = self._engine("full").evaluate(_request("send_money"))

        self.assertEqual(result.stage, "p6")
        self.assertEqual(self.client.call_count, 1)

    def test_allow_all_allows_every_tool_without_judge(self):
        engine = self._engine("allow_all")

        for tool_name in BANKING_TOOL_METADATA:
            with self.subTest(tool_name=tool_name):
                result = engine.evaluate(_request(tool_name))
                self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(self.client.call_count, 0)

    def test_escalate_all_allows_reads_and_escalates_writes(self):
        engine = self._engine("escalate_all")

        self.assertEqual(
            engine.evaluate(_request("get_balance")).decision,
            Decision.ALLOW,
        )
        for tool_name in ("send_money", "update_user_info", "unknown_tool"):
            with self.subTest(tool_name=tool_name):
                result = engine.evaluate(_request(tool_name))
                self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
                self.assertEqual(result.stage, "baseline")
        self.assertEqual(self.client.call_count, 0)

    def test_escalate_all_sends_unrequested_password_change_to_reviewer(self):
        # P1 会直接拒绝；baseline 让 reviewer 看到每个写操作。
        result = self._engine("escalate_all").evaluate(_request("update_password"))

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)

    def test_p1_only_escalates_without_judge(self):
        engine = self._engine("p1_only")

        send = engine.evaluate(_request("send_money"))
        password = engine.evaluate(_request("update_password"))

        self.assertEqual(send.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(send.stage, "p1")
        self.assertEqual(password.decision, Decision.DENY)
        self.assertEqual(self.client.call_count, 0)

    def test_budgets(self):
        self.assertEqual(escalation_budget_for("full", 3), 3)
        self.assertEqual(escalation_budget_for("p1_only", 3), 3)
        self.assertEqual(escalation_budget_for("allow_all", 3), 3)
        self.assertEqual(
            escalation_budget_for("escalate_all", 3),
            UNLIMITED_ESCALATIONS,
        )
        self.assertEqual(
            escalation_budget_for("no_budget", 3),
            UNLIMITED_ESCALATIONS,
        )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            escalation_budget_for("judge_only", 3)
        with self.assertRaises(ValueError):
            self._engine("judge_only")
        self.assertIn("full", POLICIES)


if __name__ == "__main__":
    unittest.main()
