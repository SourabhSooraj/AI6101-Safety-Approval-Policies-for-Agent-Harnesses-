import unittest
from types import SimpleNamespace

from agentdojo.functions_runtime import FunctionsRuntime

from harness.deepseek_agent_llm import (
    DeepSeekAgentLLM,
    _message_to_deepseek,
)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="final answer",
                        tool_calls=None,
                        reasoning_content="internal reasoning",
                    )
                )
            ]
        )


class FakeDeepSeekClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class DeepSeekAgentLLMTests(unittest.TestCase):
    def test_system_message_uses_system_role(self):
        message = {
            "role": "system",
            "content": [{"type": "text", "content": "You are helpful."}],
        }

        converted = _message_to_deepseek(message, "deepseek-flash")

        self.assertEqual(converted["role"], "system")
        self.assertEqual(converted["content"][0]["text"], "You are helpful.")

    def test_user_message_is_preserved(self):
        message = {
            "role": "user",
            "content": [{"type": "text", "content": "Pay the bill."}],
        }

        converted = _message_to_deepseek(message, "deepseek-flash")

        self.assertEqual(converted["role"], "user")
        self.assertEqual(converted["content"][0]["text"], "Pay the bill.")

    def test_disables_thinking_by_default(self):
        client = FakeDeepSeekClient()
        llm = DeepSeekAgentLLM(client, "deepseek-flash", temperature=0.0)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "content": "Pay the bill."}],
            }
        ]

        _, _, _, output_messages, _ = llm.query(
            query="Pay the bill.",
            runtime=FunctionsRuntime({}),
            messages=messages,
        )

        request = client.completions.calls[0]
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(
            output_messages[-1]["reasoning_content"],
            "internal reasoning",
        )

    def test_preserves_reasoning_content_when_thinking_enabled(self):
        client = FakeDeepSeekClient()
        llm = DeepSeekAgentLLM(
            client,
            "deepseek-flash",
            thinking=True,
        )
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "content": ""}],
                "reasoning_content": "previous reasoning",
                "tool_calls": None,
            }
        ]

        llm.query(
            query="continue",
            runtime=FunctionsRuntime({}),
            messages=messages,
        )

        request_message = client.completions.calls[0]["messages"][0]
        self.assertEqual(
            request_message["reasoning_content"],
            "previous reasoning",
        )


if __name__ == "__main__":
    unittest.main()
