import unittest
from types import SimpleNamespace

from agentdojo.functions_runtime import FunctionsRuntime

from harness.deepseek_agent_llm import (
    DeepSeekAgentLLM,
    _message_to_deepseek,
)


def _completion(
    content: str = "final answer",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_content: str | None = None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                    reasoning_content=reasoning_content,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


class FakeCompletions:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses or [_completion()])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeDeepSeekClient:
    def __init__(self, responses=None):
        self.completions = FakeCompletions(responses)
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
        client = FakeDeepSeekClient(
            [
                _completion(
                    reasoning_content="internal reasoning",
                )
            ]
        )
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

    def test_qwen3_system_message_disables_thinking(self):
        message = {
            "role": "system",
            "content": [{"type": "text", "content": "You are helpful."}],
        }

        converted = _message_to_deepseek(message, "qwen3:8b")
        with_thinking = _message_to_deepseek(
            message, "qwen3:8b", disable_thinking=False
        )

        self.assertEqual(
            converted["content"][0]["text"],
            "You are helpful. /no_think",
        )
        self.assertEqual(with_thinking["content"][0]["text"], "You are helpful.")

    def test_think_block_is_not_kept_in_history(self):
        client = FakeDeepSeekClient(
            [_completion(content="<think>plan {x}</think>All paid.")]
        )
        llm = DeepSeekAgentLLM(client, "qwen3:8b")

        _, _, _, output_messages, _ = llm.query(
            "Pay the bill.", FunctionsRuntime({})
        )

        self.assertEqual(
            output_messages[-1]["content"][0]["content"],
            "All paid.",
        )

    def test_agent_usage_is_accumulated(self):
        client = FakeDeepSeekClient(
            [
                _completion(prompt_tokens=1_000_000),
                _completion(completion_tokens=1_000_000),
            ]
        )
        llm = DeepSeekAgentLLM(
            client=client,
            model="deepseek-flash",
            cost_period="off_peak",
        )

        for _ in range(2):
            llm.query("Pay the bill.", FunctionsRuntime({}))

        self.assertEqual(llm.total_usage.input_tokens, 1_000_000)
        self.assertEqual(llm.total_usage.output_tokens, 1_000_000)
        self.assertAlmostEqual(llm.total_usage.estimated_cost_usd, 0.75)


if __name__ == "__main__":
    unittest.main()
