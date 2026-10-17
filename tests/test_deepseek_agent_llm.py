import unittest
from types import SimpleNamespace

import httpx
import openai
from agentdojo.functions_runtime import FunctionsRuntime
from tenacity import wait_none

from harness.deepseek_agent_llm import (
    DeepSeekAgentLLM,
    _create_completion,
    _message_to_deepseek,
)


def _usage(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


class FakeCompletions:
    def __init__(self, usages=None):
        self.calls = []
        self.usages = list(usages or [])

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
            ],
            usage=self.usages.pop(0) if self.usages else None,
        )


class FakeDeepSeekClient:
    def __init__(self, usages=None):
        self.completions = FakeCompletions(usages)
        self.chat = SimpleNamespace(completions=self.completions)


class FlakyCompletions:
    """第一次请求抛出连接错误，之后成功。"""

    def __init__(self):
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.deepseek.com")
            )
        return "ok"


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

    def test_non_deepseek_api_gets_no_thinking_field(self):
        client = FakeDeepSeekClient()
        llm = DeepSeekAgentLLM(
            client,
            "qwen2.5:7b",
            temperature=0.0,
            send_thinking_param=False,
        )

        llm.query("Pay the bill.", FunctionsRuntime({}))

        request = client.completions.calls[0]
        self.assertNotIn("extra_body", request)
        self.assertEqual(request["temperature"], 0.0)

    def test_thinking_without_deepseek_api_is_rejected(self):
        with self.assertRaises(ValueError):
            DeepSeekAgentLLM(
                FakeDeepSeekClient(),
                "qwen2.5:7b",
                thinking=True,
                send_thinking_param=False,
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

    def test_agent_usage_is_accumulated(self):
        client = FakeDeepSeekClient(
            usages=[_usage(1_000_000, 0), _usage(0, 1_000_000)]
        )
        llm = DeepSeekAgentLLM(
            client,
            "deepseek-flash",
            cost_period="off_peak",
        )

        for _ in range(2):
            llm.query("Pay the bill.", FunctionsRuntime({}))

        self.assertEqual(llm.total_usage.input_tokens, 1_000_000)
        self.assertEqual(llm.total_usage.output_tokens, 1_000_000)
        self.assertAlmostEqual(llm.total_usage.estimated_cost_usd, 0.75)

    def test_transient_api_error_is_retried(self):
        completions = FlakyCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        # 去掉等待时间，只验证重试次数。
        result = _create_completion.retry_with(wait=wait_none())(
            client,
            model="deepseek-flash",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(completions.attempts, 2)


if __name__ == "__main__":
    unittest.main()
