import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness.deepseek_judge_client import DeepSeekJudgeClient


class FakeCompletions:
    def __init__(self, content: str, usage=None) -> None:
        self.content = content
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ],
            usage=self.usage,
        )


class FakeOpenAIClient:
    def __init__(self, content: str, usage=None) -> None:
        self.completions = FakeCompletions(content, usage=usage)
        self.chat = SimpleNamespace(completions=self.completions)


class DeepSeekJudgeClientTests(unittest.TestCase):
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DeepSeekJudgeClient()

    def test_complete_uses_deepseek_configuration(self):
        sdk_client = FakeOpenAIClient(
            '{"risk_score": 0.7, "reason": "test"}'
        )
        client = DeepSeekJudgeClient(
            model="deepseek-flash",
            openai_client=sdk_client,
        )

        content = client.complete("judge this call")

        self.assertEqual(content, '{"risk_score": 0.7, "reason": "test"}')
        call = sdk_client.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-flash")
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(
            call["response_format"],
            {"type": "json_object"},
        )

    def test_local_base_url_needs_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            client = DeepSeekJudgeClient(
                model="llama3.1:8b",
                base_url="http://localhost:11434/v1",
            )
        self.assertEqual(client.client.api_key, "ollama")

    def test_qwen3_prompt_disables_thinking_and_strips_think_block(self):
        sdk_client = FakeOpenAIClient(
            '<think>{"risk_score": 0.1}</think>{"risk_score": 0.7}'
        )
        client = DeepSeekJudgeClient(
            model="qwen3:8b",
            openai_client=sdk_client,
        )

        content = client.complete("judge this call")

        self.assertEqual(content, '{"risk_score": 0.7}')
        prompt = sdk_client.completions.calls[0]["messages"][0]["content"]
        self.assertTrue(prompt.endswith("/no_think"))

    def test_local_server_turns_thinking_off(self):
        local = FakeOpenAIClient('{"risk_score": 0.7}')
        DeepSeekJudgeClient(
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
            openai_client=local,
        ).complete("judge this call")
        remote = FakeOpenAIClient('{"risk_score": 0.7}')
        DeepSeekJudgeClient(openai_client=remote).complete("judge this call")

        self.assertEqual(local.completions.calls[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", remote.completions.calls[0])

    def test_think_only_response_raises(self):
        client = DeepSeekJudgeClient(
            model="qwen3:8b",
            openai_client=FakeOpenAIClient("<think>hmm</think>"),
        )
        with self.assertRaises(RuntimeError):
            client.complete("judge this call")

    def test_complete_tracks_token_usage_and_cost(self):
        usage = SimpleNamespace(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        sdk_client = FakeOpenAIClient(
            '{"risk_score": 0.7, "reason": "test"}',
            usage=usage,
        )
        client = DeepSeekJudgeClient(
            model="deepseek-flash",
            cost_period="off_peak",
            openai_client=sdk_client,
        )

        client.complete("judge this call")

        self.assertIsNotNone(client.last_usage)
        self.assertEqual(client.last_usage.input_tokens, 1_000_000)
        self.assertEqual(client.last_usage.output_tokens, 1_000_000)
        self.assertAlmostEqual(
            client.last_usage.estimated_cost_usd,
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
