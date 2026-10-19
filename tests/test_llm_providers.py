import os
import unittest
from unittest.mock import patch

from harness.llm_providers import (
    LOCAL_PLACEHOLDER_API_KEY,
    PROVIDERS,
    QWEN_BASE_URL,
    is_dashscope_api,
    is_local_base_url,
    resolve_api_key,
    strip_think,
    thinking_request_args,
    thinking_switch,
)


class LLMProvidersTests(unittest.TestCase):
    def test_local_base_urls(self):
        self.assertTrue(is_local_base_url("http://localhost:11434/v1"))
        self.assertTrue(is_local_base_url("http://127.0.0.1:11434/v1"))
        self.assertFalse(is_local_base_url("https://api.deepseek.com"))

    def test_dashscope_base_urls(self):
        self.assertTrue(is_dashscope_api(QWEN_BASE_URL))
        self.assertTrue(
            is_dashscope_api("https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.assertTrue(
            is_dashscope_api(
                "https://ws-123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
            )
        )
        self.assertFalse(is_dashscope_api("https://api.deepseek.com"))
        self.assertFalse(is_dashscope_api("http://localhost:11434/v1"))
        self.assertFalse(is_dashscope_api("https://evil-dashscope.aliyuncs.com.example"))

    def test_local_url_without_key_uses_placeholder(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_api_key("http://localhost:11434/v1"),
                LOCAL_PLACEHOLDER_API_KEY,
            )

    def test_remote_url_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_api_key("https://api.deepseek.com")

    def test_key_from_environment(self):
        with patch.dict(os.environ, {"MY_KEY": "secret"}, clear=True):
            self.assertEqual(
                resolve_api_key("https://api.deepseek.com", None, "MY_KEY"),
                "secret",
            )

    def test_strip_think_removes_reasoning_with_braces(self):
        raw = '<think>maybe {"risk_score": 0.1}?</think>\n{"risk_score": 0.9}'
        self.assertEqual(strip_think(raw), '{"risk_score": 0.9}')

    def test_strip_think_removes_unclosed_block(self):
        self.assertEqual(strip_think("<think>cut off"), "")

    def test_strip_think_keeps_plain_text(self):
        self.assertEqual(strip_think(' {"a": 1} '), '{"a": 1}')
        self.assertIsNone(strip_think(None))

    def test_thinking_switch_only_for_qwen3(self):
        self.assertEqual(thinking_switch("qwen3:8b"), " /no_think")
        self.assertEqual(thinking_switch("llama3.1:8b"), "")
        self.assertEqual(thinking_switch("deepseek-flash"), "")


    def test_thinking_request_args_for_ollama(self):
        url = "http://localhost:11434/v1"
        self.assertEqual(
            thinking_request_args(url, thinking=False),
            {"reasoning_effort": "none"},
        )
        self.assertEqual(thinking_request_args(url, thinking=True), {})
        self.assertEqual(
            thinking_request_args(url, thinking=True, reasoning_effort="low"),
            {"reasoning_effort": "low"},
        )

    def test_thinking_request_args_for_vllm(self):
        url = "http://127.0.0.1:11535/v1"
        self.assertEqual(
            thinking_request_args(url, thinking=False, local_server="vllm"),
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        self.assertEqual(
            thinking_request_args(
                url, thinking=True, reasoning_effort="low", local_server="vllm"
            ),
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        )
        # vLLM 只影响本地地址；DeepSeek 照旧。
        self.assertEqual(
            thinking_request_args(
                "https://api.deepseek.com", thinking=False, local_server="vllm"
            ),
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_vllm_provider_defaults(self):
        provider = PROVIDERS["vllm"]
        self.assertEqual(provider.local_server, "vllm")
        self.assertEqual(provider.agent_model, "Qwen/Qwen3-8B")
        self.assertTrue(is_local_base_url(provider.base_url))
        self.assertEqual(PROVIDERS["ollama"].local_server, "ollama")

    def test_thinking_request_args_for_deepseek(self):
        url = "https://api.deepseek.com"
        self.assertEqual(
            thinking_request_args(url, thinking=False, reasoning_effort="high"),
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )
        self.assertEqual(
            thinking_request_args(url, thinking=True, reasoning_effort="high"),
            {
                "extra_body": {"thinking": {"type": "enabled"}},
                "reasoning_effort": "high",
            },
        )


if __name__ == "__main__":
    unittest.main()
