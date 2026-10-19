import argparse
import os
import unittest
from unittest.mock import patch

from harness.episode import add_episode_args, config_from_args
from harness.llm_providers import DEEPSEEK_BASE_URL, OLLAMA_BASE_URL


def _parse(*argv: str):
    parser = argparse.ArgumentParser()
    add_episode_args(parser)
    return config_from_args(parser.parse_args(list(argv)))


class EpisodeConfigTests(unittest.TestCase):
    def test_ollama_defaults_need_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _parse("--provider", "ollama")

        self.assertEqual(config.agent_model, "qwen3:8b")
        self.assertEqual(config.judge_model, "qwen3:8b")
        self.assertEqual(config.reviewer_model, "llama3.1:8b")
        self.assertEqual(config.agent_base_url, OLLAMA_BASE_URL)
        self.assertEqual(config.reviewer_base_url, OLLAMA_BASE_URL)
        self.assertEqual(config.request_timeout, 300.0)
        self.assertTrue(config.uses_local_model)

    def test_deepseek_defaults_match_previous_behaviour(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}, clear=True):
            config = _parse("--model", "deepseek-v4-pro")

        self.assertEqual(config.agent_base_url, DEEPSEEK_BASE_URL)
        self.assertEqual(config.judge_model, "deepseek-v4-pro")
        self.assertEqual(config.reviewer_model, "deepseek-v4-pro")
        self.assertFalse(config.uses_local_model)

    def test_deepseek_without_key_exits(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                _parse()

    def test_mixed_local_agent_and_remote_reviewer(self):
        env = {"REVIEWER_KEY": "r"}
        with patch.dict(os.environ, env, clear=True):
            config = _parse(
                "--provider", "ollama",
                "--reviewer-model", "deepseek-flash",
                "--reviewer-base-url", DEEPSEEK_BASE_URL,
                "--reviewer-api-key-env", "REVIEWER_KEY",
            )

        self.assertEqual(config.reviewer_api_key, "r")
        self.assertEqual(config.model_context()["judge_model"], "qwen3:8b")

    def test_vllm_provider_sets_the_local_server(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _parse(
                "--provider", "vllm",
                "--agent-base-url", "http://127.0.0.1:11535/v1",
            )

        self.assertEqual(config.local_server, "vllm")
        self.assertEqual(config.agent_model, "Qwen/Qwen3-8B")
        self.assertEqual(config.reviewer_model, "Qwen/Qwen3-8B")
        self.assertEqual(config.judge_base_url, "http://127.0.0.1:11535/v1")
        self.assertTrue(config.send_thinking_param)
        self.assertEqual(config.model_context()["local_server"], "vllm")

    def test_local_server_flag_and_default(self):
        with patch.dict(os.environ, {}, clear=True):
            ollama = _parse("--provider", "ollama")
            forced = _parse("--provider", "ollama", "--local-server", "vllm")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}, clear=True):
            remote = _parse()

        self.assertEqual(ollama.local_server, "ollama")
        self.assertEqual(forced.local_server, "vllm")
        # 没有本地模型时日志里不写 local_server。
        self.assertIsNone(remote.model_context()["local_server"])


if __name__ == "__main__":
    unittest.main()
