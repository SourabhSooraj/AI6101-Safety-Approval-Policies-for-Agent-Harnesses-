import json
import tempfile
import unittest
from pathlib import Path

from harness.budget_allocator import BudgetAllocator
from harness.decision_logger import DecisionLogger
from harness.llm_judge import LLMJudge
from harness.policy import Decision, PolicyRequest, ToolMetadata
from harness.policy_engine import PolicyEngine


class SequenceJudgeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        return next(self.responses)


def request_for(
    tool_name: str,
    metadata: ToolMetadata,
    remaining_escalations: int = 3,
    expected_remaining_risky_calls: int = 1,
) -> PolicyRequest:
    return PolicyRequest(
        tool_name=tool_name,
        tool_call_id="call-1",
        arguments={"recipient": "US99", "amount": 500},
        task_id="task-1",
        user_request="Pay my rent.",
        tool_metadata=metadata,
        remaining_escalations=remaining_escalations,
        expected_remaining_risky_calls=expected_remaining_risky_calls,
    )


def external_metadata() -> ToolMetadata:
    return ToolMetadata(
        name="send_money",
        side_effect="write",
        scope="external",
        reversibility="irreversible",
        sensitive_args=("recipient", "amount"),
        impact=4,
    )


class PolicyEngineHybridTests(unittest.TestCase):
    def test_read_only_bypasses_judge(self):
        client = SequenceJudgeClient([])
        engine = PolicyEngine(
            judge=LLMJudge(client, sample_count=1),
            allocator=BudgetAllocator(),
        )
        metadata = ToolMetadata(
            name="get_balance",
            side_effect="read",
            scope="local",
            reversibility="reversible",
            sensitive_args=(),
            impact=1,
        )

        result = engine.evaluate(request_for("get_balance", metadata))

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p1_read_only")
        self.assertEqual(client.call_count, 0)

    def test_high_risk_value_escalates(self):
        client = SequenceJudgeClient(
            ['{"risk_score": 0.9, "reason": "unexpected payment"}']
        )
        engine = PolicyEngine(
            judge=LLMJudge(client, sample_count=1),
            allocator=BudgetAllocator(tau0=1.0),
        )

        result = engine.evaluate(
            request_for("send_money", external_metadata())
        )

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p6_value_above_tau")
        self.assertEqual(result.stage, "p6")
        self.assertAlmostEqual(result.value, 3.6)

    def test_low_risk_value_allows(self):
        client = SequenceJudgeClient(
            ['{"risk_score": 0.1, "reason": "expected payment"}']
        )
        engine = PolicyEngine(
            judge=LLMJudge(client, sample_count=1),
            allocator=BudgetAllocator(tau0=10.0),
        )

        result = engine.evaluate(
            request_for("send_money", external_metadata())
        )

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p6_allow_below_threshold")

    def test_judge_disagreement_forces_escalation(self):
        client = SequenceJudgeClient(
            [
                '{"risk_score": 0.1, "reason": "safe"}',
                '{"risk_score": 0.9, "reason": "unsafe"}',
                '{"risk_score": 0.5, "reason": "uncertain"}',
            ]
        )
        engine = PolicyEngine(
            judge=LLMJudge(client),
            allocator=BudgetAllocator(tau0=10.0),
        )

        result = engine.evaluate(
            request_for("send_money", external_metadata())
        )

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p6_force_escalation")

    def test_exhausted_budget_denies(self):
        client = SequenceJudgeClient(
            ['{"risk_score": 0.9, "reason": "unsafe"}']
        )
        engine = PolicyEngine(
            judge=LLMJudge(client, sample_count=1),
            allocator=BudgetAllocator(tau0=10.0),
        )

        result = engine.evaluate(
            request_for(
                "send_money",
                external_metadata(),
                remaining_escalations=0,
            )
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p6_budget_exhausted")

    def test_policy_engine_writes_jsonl_log(self):
        client = SequenceJudgeClient(
            ['{"risk_score": 0.9, "reason": "unexpected payment"}']
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "policy.jsonl"
            engine = PolicyEngine(
                judge=LLMJudge(client, sample_count=1),
                allocator=BudgetAllocator(tau0=1.0),
                decision_logger=DecisionLogger(log_path, run_id="test-run"),
            )

            engine.evaluate(request_for("send_money", external_metadata()))

            record = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["run_id"], "test-run")
            self.assertEqual(record["tool_name"], "send_money")
            self.assertEqual(record["decision"], "REQUIRE_APPROVAL")
            self.assertEqual(record["stage"], "p6")
            self.assertTrue(record["judge_success"])


if __name__ == "__main__":
    unittest.main()
