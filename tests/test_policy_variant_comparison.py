import unittest

from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policies.factory import create_policy
from harness.policy import Decision, PolicyRequest, ToolMetadata


class HighRiskJudgeClient:
    last_usage = None

    def complete(self, prompt: str) -> str:
        return '{"risk_score": 0.9, "reason": "high risk"}'


def request_for() -> PolicyRequest:
    return PolicyRequest(
        tool_name="send_money",
        tool_call_id="call-1",
        arguments={"recipient": "US99", "amount": 500},
        task_id="task-1",
        user_request="Pay my rent.",
        tool_metadata=ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        ),
        remaining_escalations=3,
        expected_remaining_risky_calls=1,
    )


class PolicyVariantComparisonTests(unittest.TestCase):
    def test_variants_produce_expected_policy_paths(self):
        judge = LLMJudge(
            HighRiskJudgeClient(),
            sample_count=1,
        )
        p1_only = create_policy("p1_only")
        p4_only = create_policy("p4_only", judge=judge)
        hybrid = create_policy(
            "hybrid",
            judge=judge,
            allocator=BudgetAllocator(tau0=1.0),
        )
        request = request_for()

        p1_result = p1_only.evaluate(request)
        p4_result = p4_only.evaluate(request)
        hybrid_result = hybrid.evaluate(request)

        self.assertEqual(p1_result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(p1_result.policy_variant, "p1_only")
        self.assertEqual(p4_result.decision, Decision.DENY)
        self.assertEqual(p4_result.policy_variant, "p4_only")
        self.assertEqual(hybrid_result.decision, Decision.DENY)
        self.assertEqual(hybrid_result.policy_variant, "hybrid")


if __name__ == "__main__":
    unittest.main()
