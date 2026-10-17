import unittest

from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policies.factory import create_policy


class FakeJudgeClient:
    last_usage = None

    def complete(self, prompt: str) -> str:
        return '{"risk_score": 0.1, "reason": "safe"}'


class PolicyFactoryTests(unittest.TestCase):
    def test_creates_p1_only_without_judge(self):
        policy = create_policy("p1")

        self.assertEqual(policy.name, "p1_only")

    def test_creates_p4_only(self):
        policy = create_policy(
            "p4",
            judge=LLMJudge(FakeJudgeClient(), sample_count=1),
            t_low=0.2,
            t_high=0.8,
        )

        self.assertEqual(policy.name, "p4_only")

    def test_creates_hybrid_alias(self):
        policy = create_policy(
            "p6",
            judge=LLMJudge(FakeJudgeClient(), sample_count=1),
            allocator=BudgetAllocator(),
        )

        self.assertEqual(policy.name, "hybrid")

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            create_policy("unknown")


if __name__ == "__main__":
    unittest.main()
