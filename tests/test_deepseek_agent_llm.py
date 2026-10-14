import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentdojo.functions_runtime import FunctionsRuntime
from openai.types.chat import ChatCompletionMessage

from harness.deepseek_agent_llm import DeepSeekAgentLLM, _message_to_deepseek


def _fake_completion(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage(
                    role="assistant",
                    content="done",
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


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

    def test_agent_usage_is_accumulated(self):
        llm = DeepSeekAgentLLM(
            client=object(),
            model="deepseek-flash",
            cost_period="off_peak",
        )

        with patch(
            "harness.deepseek_agent_llm.chat_completion_request",
            side_effect=[
                _fake_completion(1_000_000, 0),
                _fake_completion(0, 1_000_000),
            ],
        ):
            for _ in range(2):
                llm.query("Pay the bill.", FunctionsRuntime([]))

        self.assertEqual(llm.total_usage.input_tokens, 1_000_000)
        self.assertEqual(llm.total_usage.output_tokens, 1_000_000)
        self.assertAlmostEqual(llm.total_usage.estimated_cost_usd, 0.75)


if __name__ == "__main__":
    unittest.main()
