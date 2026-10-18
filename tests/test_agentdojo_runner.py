import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.budget_allocator import (
    DEFAULT_COST_TAU0,
    DEFAULT_REVIEWER_ERRORS,
    BudgetAllocator,
    CostBasedAllocator,
    ReviewerErrorRates,
)
from harness.episode import build_allocator, config_from_args
from harness.llm_providers import DEEPSEEK_BASE_URL, LOCAL_PLACEHOLDER_API_KEY
from scripts.run_agentdojo_e2e import _system_message, build_parser

OLLAMA_URL = "http://localhost:11434/v1"
HOSTED_URL = "https://api.together.xyz/v1"


def _config(argv, environ):
    with patch.dict(os.environ, environ, clear=True):
        return config_from_args(build_parser().parse_args(argv))


class ModelSettingsTests(unittest.TestCase):
    def test_defaults_use_deepseek_for_every_role(self):
        config = _config([], {"DEEPSEEK_API_KEY": "key"})

        for model, base_url, api_key in (
            (config.agent_model, config.agent_base_url, config.agent_api_key),
            (config.judge_model, config.judge_base_url, config.judge_api_key),
            (
                config.reviewer_model,
                config.reviewer_base_url,
                config.reviewer_api_key,
            ),
        ):
            self.assertEqual(model, "deepseek-flash")
            self.assertEqual(base_url, DEEPSEEK_BASE_URL)
            self.assertEqual(api_key, "key")
        self.assertTrue(config.send_thinking_param)
        self.assertEqual(config.attack_model_name, "DeepSeek")

    def test_local_agent_needs_no_key_and_judge_reviewer_follow_it(self):
        config = _config(
            ["--model", "qwen2.5:7b", "--agent-base-url", OLLAMA_URL],
            {},
        )

        for model, base_url, api_key in (
            (config.agent_model, config.agent_base_url, config.agent_api_key),
            (config.judge_model, config.judge_base_url, config.judge_api_key),
            (
                config.reviewer_model,
                config.reviewer_base_url,
                config.reviewer_api_key,
            ),
        ):
            self.assertEqual(model, "qwen2.5:7b")
            self.assertEqual(base_url, OLLAMA_URL)
            self.assertEqual(api_key, LOCAL_PLACEHOLDER_API_KEY)
        # Ollama 用 reasoning_effort 开关思考，所以也发送思考参数。
        self.assertTrue(config.send_thinking_param)
        self.assertEqual(config.attack_model_name, "Qwen")

    def test_judge_can_use_a_different_server_from_the_agent(self):
        config = _config(
            [
                "--model", "llama3.1:8b",
                "--agent-base-url", OLLAMA_URL,
                "--judge-model", "deepseek-flash",
                "--judge-base-url", DEEPSEEK_BASE_URL,
                "--judge-api-key-env", "DEEPSEEK_API_KEY",
            ],
            {"DEEPSEEK_API_KEY": "key"},
        )

        self.assertEqual(config.judge_model, "deepseek-flash")
        self.assertEqual(config.judge_api_key, "key")
        self.assertEqual(config.reviewer_model, "llama3.1:8b")
        self.assertEqual(config.reviewer_base_url, OLLAMA_URL)
        self.assertEqual(config.attack_model_name, "Llama")

    def test_missing_key_for_a_remote_server_stops_the_run(self):
        with self.assertRaises(SystemExit):
            _config([], {})

    def test_other_hosted_apis_get_no_thinking_param(self):
        config = _config(
            [
                "--model", "Qwen/Qwen2.5-72B-Instruct",
                "--agent-base-url", HOSTED_URL,
                "--agent-api-key-env", "HOSTED_KEY",
            ],
            {"HOSTED_KEY": "key"},
        )

        self.assertFalse(config.send_thinking_param)

    def test_thinking_needs_deepseek_or_a_local_server(self):
        with self.assertRaises(SystemExit):
            _config(
                [
                    "--agent-base-url", HOSTED_URL,
                    "--agent-api-key-env", "HOSTED_KEY",
                    "--agent-thinking",
                ],
                {"HOSTED_KEY": "key"},
            )
        config = _config(
            ["--provider", "ollama", "--agent-thinking"],
            {},
        )
        self.assertTrue(config.agent_thinking)

    def test_explicit_attack_model_name_wins(self):
        config = _config(
            ["--attack-model-name", "Assistant"],
            {"DEEPSEEK_API_KEY": "key"},
        )

        self.assertEqual(config.attack_model_name, "Assistant")

    def test_old_api_key_env_name_still_works(self):
        config = _config(
            ["--api-key-env", "MY_KEY"],
            {"MY_KEY": "key"},
        )

        self.assertEqual(config.agent_api_key, "key")
        self.assertEqual(config.judge_api_key, "key")


class AgentDojoRunnerTests(unittest.TestCase):
    def test_auto_complete_instruction_is_added(self):
        message = _system_message(auto_complete=True)

        self.assertIn("No human user is available", message)
        self.assertIn("Do not stop to ask for confirmation", message)

    def test_default_system_message_can_be_preserved(self):
        message = _system_message(auto_complete=False)

        self.assertNotIn("No human user is available", message)

    def test_default_policy_is_hybrid(self):
        self.assertEqual(build_parser().parse_args([]).policy, "hybrid")


def _allocator(argv):
    return build_allocator(_config(["--provider", "ollama", *argv], {}))


class AllocatorOptionTests(unittest.TestCase):
    def test_default_is_the_original_value_allocator(self):
        allocator = _allocator([])

        self.assertIsInstance(allocator, BudgetAllocator)
        self.assertEqual(allocator.tau0, 1.0)

    def test_cost_allocator_has_its_own_tau0_default(self):
        allocator = _allocator(["--allocator", "cost"])

        self.assertIsInstance(allocator, CostBasedAllocator)
        self.assertEqual(allocator.tau0, DEFAULT_COST_TAU0)
        self.assertEqual(allocator.allow_threshold, 0.5)
        self.assertEqual(allocator.utility_cost, 1.0)
        self.assertEqual(allocator.default_reviewer_errors, DEFAULT_REVIEWER_ERRORS)

    def test_cost_allocator_options(self):
        allocator = _allocator(
            [
                "--allocator", "cost",
                "--tau0", "0.2",
                "--utility-cost", "2",
                "--auto-rule", "cost",
            ]
        )

        self.assertEqual(allocator.tau0, 0.2)
        self.assertEqual(allocator.utility_cost, 2.0)
        self.assertIsNone(allocator.allow_threshold)

    def test_reviewer_error_table_and_its_default_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.json"
            path.write_text(
                json.dumps(
                    {
                        "send_money": {"false_approve": 0.05, "false_deny": 0.2},
                        "default": {"false_approve": 0.3, "false_deny": 0.3},
                    }
                ),
                encoding="utf-8",
            )

            allocator = _allocator(
                ["--allocator", "cost", "--reviewer-error-table", str(path)]
            )

        self.assertEqual(
            allocator.reviewer_errors,
            {"send_money": ReviewerErrorRates(0.05, 0.2)},
        )
        self.assertEqual(
            allocator.default_reviewer_errors, ReviewerErrorRates(0.3, 0.3)
        )

    def test_options_for_the_other_allocator_stop_the_run(self):
        with self.assertRaises(SystemExit):
            _allocator(["--allocator", "cost", "--q-table", "q.json"])
        with self.assertRaises(SystemExit):
            _allocator(["--reviewer-error-table", "errors.json"])

    def test_allocator_settings_are_logged(self):
        config = _config(
            ["--provider", "ollama", "--allocator", "cost"], {}
        )

        context = config.allocator_context()
        self.assertEqual(context["allocator"], "cost")
        self.assertEqual(context["tau0"], DEFAULT_COST_TAU0)
        self.assertEqual(context["utility_cost"], 1.0)
        self.assertEqual(context["auto_rule"], "threshold")


if __name__ == "__main__":
    unittest.main()
