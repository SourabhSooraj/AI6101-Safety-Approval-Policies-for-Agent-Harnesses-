import unittest

from harness.budget_allocator import BudgetAllocator
from harness.policy import Decision


class BudgetAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.allocator = BudgetAllocator(tau0=1.0, allow_threshold=0.5)

    def test_budget_exhaustion_denies(self):
        result = self.allocator.allocate(
            risk_score=0.9,
            impact=4,
            q=1.0,
            remaining_escalations=0,
            expected_remaining_risky_calls=2,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_budget_exhausted")

    def test_exhausted_budget_still_allows_low_value_low_risk_call(self):
        # value 0.4 < tau 3.0：本来就不会升级，预算为 0 时照常放行。
        result = self.allocator.allocate(
            risk_score=0.1,
            impact=4,
            q=1.0,
            remaining_escalations=0,
            expected_remaining_risky_calls=3,
        )

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p6_allow_below_threshold")

    def test_exhausted_budget_still_denies_low_value_high_risk_call(self):
        result = self.allocator.allocate(
            risk_score=0.6,
            impact=1,
            q=1.0,
            remaining_escalations=0,
            expected_remaining_risky_calls=3,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_deny_above_threshold")

    def test_exhausted_budget_denies_low_risk_call_that_would_escalate(self):
        # risk 0.3 低于放行阈值，但 value 1.2 > tau 1.0，本应升级，因此拒绝。
        result = self.allocator.allocate(
            risk_score=0.3,
            impact=4,
            q=1.0,
            remaining_escalations=0,
            expected_remaining_risky_calls=1,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_budget_exhausted")

    def test_exhausted_budget_denies_forced_escalation(self):
        # Judge 失败时即使风险很低也必须 fail closed。
        result = self.allocator.allocate(
            risk_score=0.1,
            impact=1,
            q=0.1,
            remaining_escalations=0,
            expected_remaining_risky_calls=3,
            force_escalation=True,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_budget_exhausted")

    def test_value_above_tau_escalates(self):
        result = self.allocator.allocate(
            risk_score=0.8,
            impact=4,
            q=1.0,
            remaining_escalations=3,
            expected_remaining_risky_calls=3,
        )

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p6_value_above_tau")
        self.assertAlmostEqual(result.value, 3.2)
        self.assertAlmostEqual(result.tau, 1.0)

    def test_low_value_and_low_risk_allows(self):
        result = self.allocator.allocate(
            risk_score=0.2,
            impact=1,
            q=0.1,
            remaining_escalations=10,
            expected_remaining_risky_calls=1,
        )

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p6_allow_below_threshold")
        self.assertLess(result.value, result.tau)

    def test_low_value_and_high_risk_denies(self):
        result = self.allocator.allocate(
            risk_score=0.8,
            impact=1,
            q=0.1,
            remaining_escalations=10,
            expected_remaining_risky_calls=1,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_deny_above_threshold")

    def test_scarce_budget_raises_tau(self):
        result = self.allocator.allocate(
            risk_score=0.8,
            impact=4,
            q=1.0,
            remaining_escalations=1,
            expected_remaining_risky_calls=5,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertAlmostEqual(result.tau, 5.0)

    def test_force_escalation_requires_approval(self):
        result = self.allocator.allocate(
            risk_score=0.1,
            impact=1,
            q=0.1,
            remaining_escalations=3,
            expected_remaining_risky_calls=1,
            force_escalation=True,
        )

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p6_force_escalation")


if __name__ == "__main__":
    unittest.main()
