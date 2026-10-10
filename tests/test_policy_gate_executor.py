import unittest

from agentdojo.types import FunctionCall

from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor


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


if __name__ == "__main__":
    unittest.main()