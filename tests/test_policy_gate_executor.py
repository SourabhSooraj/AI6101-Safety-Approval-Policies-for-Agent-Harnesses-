import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.types import FunctionCall

from harness.policy import Decision, ToolMetadata
from harness.decision_logger import DecisionLogger
from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.simulated_reviewer import ReviewResult


class FakeReviewer:
    def __init__(self, decision: Decision, success: bool = True) -> None:
        self.decision = decision
        self.success = success
        self.calls = 0

    def review(self, request, trace_summary=""):
        self.calls += 1
        return ReviewResult(
            decision=self.decision,
            reason="review reason",
            success=self.success,
        )


class PolicyGateExecutorTests(unittest.TestCase):
    def test_denied_tool_is_blocked(self):
        policy_engine = PolicyEngine(denied_tools={"delete_file"})
        executor = PolicyGateToolsExecutor(policy_engine)

        tool_call = FunctionCall(
            function="delete_file",
            args={},
            id="call-1",
            placeholder_args=None,
        )

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]

        class FakeRuntime:
            pass

        _, _, _, result_messages, _ = executor.query(
            query="测试拒绝规则",
            runtime=FakeRuntime(),
            messages=messages,
        )

        self.assertEqual(len(result_messages), 2)
        self.assertIn("DENY", str(result_messages[-1]))

    def test_escalation_budget_is_consumed_and_persisted(self):
        tool_call = FunctionCall(
            function="send_money",
            args={"recipient": "US99", "amount": 500},
            id="call-2",
            placeholder_args=None,
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        )
        executor = PolicyGateToolsExecutor(
            PolicyEngine(),
            tool_metadata={"send_money": metadata},
            escalation_budget=1,
        )

        class FakeRuntime:
            pass

        extra_args = {"task_id": "budget-task"}
        _, _, _, first_messages, extra_args = executor.query(
            query="Pay rent.",
            runtime=FakeRuntime(),
            messages=messages,
            extra_args=extra_args,
        )

        self.assertIn("REQUIRE_APPROVAL", str(first_messages[-1]))
        self.assertEqual(extra_args["remaining_escalations"], 0)
        self.assertEqual(extra_args["escalations_used"], 1)

        _, _, _, second_messages, extra_args = executor.query(
            query="Pay rent again.",
            runtime=FakeRuntime(),
            messages=messages,
            extra_args=extra_args,
        )

        self.assertIn("DENY", str(second_messages[-1]))
        self.assertIn("budget exhausted", str(second_messages[-1]))
        self.assertEqual(extra_args["remaining_escalations"], 0)
        self.assertEqual(extra_args["escalations_used"], 1)

    def test_zero_budget_converts_required_approval_to_deny(self):
        tool_call = FunctionCall(
            function="send_money",
            args={},
            id="call-3",
            placeholder_args=None,
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=(),
            impact=4,
        )
        executor = PolicyGateToolsExecutor(
            PolicyEngine(),
            tool_metadata={"send_money": metadata},
            escalation_budget=0,
        )

        class FakeRuntime:
            pass

        _, _, _, result_messages, extra_args = executor.query(
            query="Pay rent.",
            runtime=FakeRuntime(),
            messages=messages,
        )

        self.assertIn("DENY", str(result_messages[-1]))
        self.assertEqual(extra_args["remaining_escalations"], 0)
        self.assertEqual(extra_args["escalations_used"], 0)

    def test_reviewer_approval_executes_tool(self):
        tool_call = FunctionCall(
            function="send_money",
            args={"recipient": "US99", "amount": 500},
            id="call-4",
            placeholder_args=None,
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        )
        reviewer = FakeReviewer(Decision.ALLOW)
        executor = PolicyGateToolsExecutor(
            PolicyEngine(),
            tool_metadata={"send_money": metadata},
            escalation_budget=1,
            reviewer=reviewer,
        )

        class FakeRuntime:
            pass

        executed_message = {"role": "tool", "content": "executed"}
        with patch.object(
            ToolsExecutor,
            "query",
            return_value=(
                "query",
                object(),
                object(),
                [executed_message],
                {
                    "remaining_escalations": 0,
                    "escalations_used": 1,
                },
            ),
        ):
            _, _, _, result_messages, _ = executor.query(
                query="Pay rent.",
                runtime=FakeRuntime(),
                messages=messages,
                extra_args={"task_id": "review-task"},
            )

        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(result_messages[-1], executed_message)

    def test_reviewer_denial_blocks_tool(self):
        tool_call = FunctionCall(
            function="send_money",
            args={},
            id="call-5",
            placeholder_args=None,
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=(),
            impact=4,
        )
        reviewer = FakeReviewer(Decision.DENY)
        executor = PolicyGateToolsExecutor(
            PolicyEngine(),
            tool_metadata={"send_money": metadata},
            escalation_budget=1,
            reviewer=reviewer,
        )

        class FakeRuntime:
            pass

        extra_args = {"task_id": "review-task"}
        with patch.object(ToolsExecutor, "query") as super_query:
            _, _, _, result_messages, extra_args = executor.query(
                query="Pay rent.",
                runtime=FakeRuntime(),
                messages=messages,
                extra_args=extra_args,
            )

        super_query.assert_not_called()
        self.assertEqual(reviewer.calls, 1)
        self.assertIn("Reviewer decision: DENY", str(result_messages[-1]))
        self.assertEqual(extra_args["reviewer_verdict"], "DENY")
        self.assertEqual(extra_args["remaining_escalations"], 0)

    def test_reviewer_verdict_is_written_to_jsonl(self):
        tool_call = FunctionCall(
            function="send_money",
            args={},
            id="call-6",
            placeholder_args=None,
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=(),
            impact=4,
        )

        class FakeRuntime:
            pass

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "reviewer.jsonl"
            executor = PolicyGateToolsExecutor(
                PolicyEngine(),
                tool_metadata={"send_money": metadata},
                escalation_budget=1,
                reviewer=FakeReviewer(Decision.DENY),
                review_logger=DecisionLogger(log_path, run_id="review-run"),
            )

            executor.query(
                query="Pay rent.",
                runtime=FakeRuntime(),
                messages=messages,
                extra_args={"task_id": "review-task"},
            )

            record = json.loads(
                log_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(record["event_type"], "reviewer_decision")
            self.assertEqual(record["reviewer_verdict"], "DENY")
            self.assertEqual(record["decision"], "REQUIRE_APPROVAL")


if __name__ == "__main__":
    unittest.main()
