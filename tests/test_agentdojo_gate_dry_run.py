import unittest

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policy import Decision
from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.simulated_reviewer import ReviewResult
from harness.tool_metadata import BANKING_TOOL_METADATA


class RiskyJudgeClient:
    last_usage = None

    def complete(self, prompt: str) -> str:
        return '{"risk_score": 0.8, "reason": "requires review"}'


class AllowReviewer:
    def review(self, request, trace_summary=""):
        return ReviewResult(
            decision=Decision.ALLOW,
            reason="justified",
            success=True,
        )


class AgentDojoGateDryRunTests(unittest.TestCase):
    def test_banking_tools_execute_through_gate(self):
        suite = get_suite("v1.2.2", "banking")
        task = suite.get_user_task_by_id("user_task_0")
        environment = suite.load_and_inject_default_environment({})
        task_environment = task.init_environment(environment)
        runtime = FunctionsRuntime(suite.tools)

        gate = PolicyGateToolsExecutor(
            policy_engine=PolicyEngine(
                judge=LLMJudge(RiskyJudgeClient(), sample_count=1),
                allocator=BudgetAllocator(tau0=1.0),
            ),
            tool_metadata=BANKING_TOOL_METADATA,
            task_id="user_task_0",
            escalation_budget=3,
            reviewer=AllowReviewer(),
        )
        extra_args = {
            "task_id": "user_task_0",
            "user_request": task.PROMPT,
            "remaining_escalations": 3,
            "expected_remaining_risky_calls": 2,
        }

        read_call = FunctionCall(
            function="read_file",
            args={"file_path": "bill-december-2023.txt"},
            id="read-bill",
            placeholder_args=None,
        )
        read_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [read_call],
            }
        ]
        _, runtime, task_environment, messages, extra_args = gate.query(
            query=task.PROMPT,
            runtime=runtime,
            env=task_environment,
            messages=read_messages,
            extra_args=extra_args,
        )
        read_result = messages[-1]

        self.assertEqual(read_result["role"], "tool")
        self.assertIsNone(read_result["error"])

        send_money_ground_truth = next(
            call
            for call in task.ground_truth(task_environment)
            if call.function == "send_money"
        )
        send_call = FunctionCall(
            function=send_money_ground_truth.function,
            args=send_money_ground_truth.args,
            id="send-money",
            placeholder_args=None,
        )
        send_messages = [
            *messages,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [send_call],
            },
        ]
        _, _, _, messages, extra_args = gate.query(
            query=task.PROMPT,
            runtime=runtime,
            env=task_environment,
            messages=send_messages,
            extra_args=extra_args,
        )
        send_result = messages[-1]

        self.assertEqual(send_result["role"], "tool")
        self.assertIsNone(send_result["error"])
        self.assertEqual(extra_args["escalations_used"], 1)
        self.assertEqual(extra_args["remaining_escalations"], 2)


if __name__ == "__main__":
    unittest.main()
