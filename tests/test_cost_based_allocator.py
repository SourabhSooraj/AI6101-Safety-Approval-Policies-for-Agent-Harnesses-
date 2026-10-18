import json
import tempfile
import unittest
from pathlib import Path

from harness.budget_allocator import (
    BudgetAllocator,
    CostBasedAllocator,
    ReviewerErrorRates,
    load_reviewer_error_table,
)
from harness.llm_judge import LLMJudge
from harness.policy import Decision, PolicyRequest, ToolMetadata
from harness.policy_engine import PolicyEngine

PERFECT_REVIEWER = ReviewerErrorRates(false_approve=0.0, false_deny=0.0)
COIN_FLIP_REVIEWER = ReviewerErrorRates(false_approve=0.5, false_deny=0.5)


class CostBasedAllocatorTests(unittest.TestCase):
    def setUp(self):
        # 默认值：tau0 0.05，utility_cost 1，放行阈值 0.5，reviewer 错误率 0.1/0.1。
        self.allocator = CostBasedAllocator()

    def allocate(self, allocator=None, **overrides):
        arguments = {
            "risk_score": 0.4,
            "impact": 4,
            "q": 1.0,
            "remaining_escalations": 1,
            "expected_remaining_risky_calls": 3,
        }
        arguments.update(overrides)
        return (allocator or self.allocator).allocate(**arguments)

    def test_tight_budget_reviews_auto_allowed_transfer(self):
        # k=1、预计 3 个风险调用时，旧 allocator 的 tau=3 > value 1.6，
        # 风险 0.4 的转账会被自动放行；新 allocator 把这次审核留给它。
        old = self.allocate(BudgetAllocator(tau0=1.0))
        new = self.allocate()

        self.assertEqual(old.decision, Decision.ALLOW)
        self.assertEqual(new.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(new.policy_name, "p6_gain_above_tau")
        # auto cost 1.6 - review cost (1.6*0.1 + 0.6*0.1) = 1.38
        self.assertAlmostEqual(new.value, 1.38)
        self.assertAlmostEqual(new.tau, 0.15)
        self.assertEqual(new.stage, "p6")

    def test_auto_denied_high_risk_call_is_not_reviewed(self):
        # 自动拒绝已经安全；审核只可能救回一个合法调用，收益 0.055 < tau 0.15。
        result = self.allocate(risk_score=0.65)

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_auto_deny")
        self.assertAlmostEqual(result.value, 0.055)

    def test_priority_peaks_at_the_auto_decision_boundary(self):
        # 旧 value 随风险单调上升；新 gain 在自动决定会翻转的附近最大。
        old = BudgetAllocator(tau0=1.0)
        old_low = self.allocate(old, risk_score=0.45).value
        old_high = self.allocate(old, risk_score=0.65).value
        new_low = self.allocate(risk_score=0.45).value
        new_high = self.allocate(risk_score=0.65).value

        self.assertLess(old_low, old_high)
        self.assertGreater(new_low, new_high)

    def test_perfect_reviewer_gain_equals_auto_cost(self):
        allocator = CostBasedAllocator(default_reviewer_errors=PERFECT_REVIEWER)

        result = self.allocate(allocator)

        self.assertAlmostEqual(result.value, 0.4 * 4)

    def test_reviewer_no_better_than_auto_rule_is_never_used(self):
        allocator = CostBasedAllocator(
            allow_threshold=None,
            default_reviewer_errors=COIN_FLIP_REVIEWER,
        )

        # 预算充足（tau 很小）也不升级：gain = 0.6 - 0.5*(1.6+0.6) < 0。
        result = self.allocate(
            allocator,
            remaining_escalations=10,
            expected_remaining_risky_calls=1,
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_auto_deny")
        self.assertLess(result.value, 0)

    def test_cost_auto_rule_moves_allow_threshold_with_impact(self):
        # impact 4、utility_cost 1：只有 4p < 1-p，即 p < 0.2 时才放行。
        allocator = CostBasedAllocator(tau0=100.0, allow_threshold=None)

        low = self.allocate(allocator, risk_score=0.15)
        high = self.allocate(allocator, risk_score=0.25)

        self.assertEqual(low.decision, Decision.ALLOW)
        self.assertEqual(low.policy_name, "p6_auto_allow")
        self.assertEqual(high.decision, Decision.DENY)
        self.assertEqual(high.policy_name, "p6_auto_deny")

    def test_exhausted_budget_denies_call_that_would_escalate(self):
        result = self.allocate(remaining_escalations=0)

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_budget_exhausted")

    def test_exhausted_budget_keeps_normal_decision_for_low_gain_call(self):
        # gain = 0.05 - (0.005 + 0.095) < 0：本来就不会升级，照常放行。
        result = self.allocate(
            risk_score=0.05,
            impact=1,
            remaining_escalations=0,
        )

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p6_auto_allow")

    def test_force_escalation_requires_approval_or_denies_without_budget(self):
        allocator = CostBasedAllocator(default_reviewer_errors=COIN_FLIP_REVIEWER)

        with_budget = self.allocate(
            allocator, risk_score=0.05, impact=1, force_escalation=True
        )
        without_budget = self.allocate(
            allocator,
            risk_score=0.05,
            impact=1,
            remaining_escalations=0,
            force_escalation=True,
        )

        self.assertEqual(with_budget.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(with_budget.policy_name, "p6_force_escalation")
        self.assertEqual(without_budget.decision, Decision.DENY)
        self.assertEqual(without_budget.policy_name, "p6_budget_exhausted")

    def test_reviewer_error_rates_are_looked_up_by_tool(self):
        allocator = CostBasedAllocator(
            reviewer_errors={"send_money": PERFECT_REVIEWER},
            default_reviewer_errors=COIN_FLIP_REVIEWER,
        )

        known = self.allocate(allocator, tool_name="send_money")
        unknown = self.allocate(allocator, tool_name="update_user_info")
        unnamed = self.allocate(allocator)

        self.assertAlmostEqual(known.value, 1.6)
        # 1.6 - 0.5*(1.6+0.6)
        self.assertAlmostEqual(unknown.value, 0.5)
        self.assertAlmostEqual(unnamed.value, 0.5)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            CostBasedAllocator(tau0=-0.1)
        with self.assertRaises(ValueError):
            CostBasedAllocator(utility_cost=0)
        with self.assertRaises(ValueError):
            CostBasedAllocator(allow_threshold=1.5)
        with self.assertRaises(ValueError):
            ReviewerErrorRates(false_approve=1.2, false_deny=0.0)
        with self.assertRaises(ValueError):
            ReviewerErrorRates(false_approve=True, false_deny=0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.allocate(risk_score=1.5)
        with self.assertRaises(ValueError):
            self.allocate(impact=5)


class ReviewerErrorTableTests(unittest.TestCase):
    def write(self, text: str, encoding: str = "utf-8") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "reviewer_errors.json"
        path.write_text(text, encoding=encoding)
        return path

    def test_loads_table_with_bom(self):
        path = self.write(
            json.dumps(
                {
                    "send_money": {"false_approve": 0.05, "false_deny": 0.2},
                    "default": {"false_approve": 0.1, "false_deny": 0.1},
                }
            ),
            encoding="utf-8-sig",
        )

        table = load_reviewer_error_table(path)

        self.assertEqual(
            table["send_money"],
            ReviewerErrorRates(false_approve=0.05, false_deny=0.2),
        )
        self.assertIn("default", table)

    def test_rejects_bad_tables(self):
        bad_payloads = [
            [],
            {"send_money": 0.1},
            {"send_money": {"false_approve": 0.1}},
            {"send_money": {"false_approve": 0.1, "false_deny": 2}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                path = self.write(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_reviewer_error_table(path)


class SingleJudgeClient:
    def __init__(self, response: str):
        self.response = response

    def complete(self, prompt: str) -> str:
        return self.response


class HybridWithCostAllocatorTests(unittest.TestCase):
    def test_hybrid_passes_tool_name_to_cost_allocator(self):
        engine = PolicyEngine(
            judge=LLMJudge(
                SingleJudgeClient('{"risk_score": 0.4, "reason": "unclear"}'),
                sample_count=1,
            ),
            allocator=CostBasedAllocator(
                reviewer_errors={"send_money": PERFECT_REVIEWER},
            ),
        )
        request = PolicyRequest(
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
            remaining_escalations=1,
            expected_remaining_risky_calls=3,
        )

        result = engine.evaluate(request)

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p6_gain_above_tau")
        self.assertEqual(result.policy_variant, "hybrid")
        # 用到了 send_money 的完美 reviewer 错误率，而不是默认 0.1/0.1。
        self.assertAlmostEqual(result.value, 1.6)


if __name__ == "__main__":
    unittest.main()
