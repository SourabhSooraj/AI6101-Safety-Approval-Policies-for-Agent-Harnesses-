import json
import tempfile
import unittest
from pathlib import Path

from harness.budget_allocator import (
    DEFAULT_COST_TAU0,
    DEFAULT_REVIEWER_ERRORS,
    BudgetAllocator,
    CostBasedAllocator,
    ReviewerErrorRates,
)
from scripts.run_agentdojo_e2e import (
    DEEPSEEK_BASE_URL,
    LOCAL_PLACEHOLDER_KEY,
    _system_message,
    build_allocator,
    build_parser,
    resolve_model_settings,
)

OLLAMA_URL = "http://localhost:11434/v1"


def _settings(argv, environ):
    return resolve_model_settings(build_parser().parse_args(argv), environ)


class ModelSettingsTests(unittest.TestCase):
    def test_defaults_use_deepseek_for_every_role(self):
        settings = _settings([], {"DEEPSEEK_API_KEY": "key"})

        for endpoint in (settings.agent, settings.judge, settings.reviewer):
            self.assertEqual(endpoint.model, "deepseek-flash")
            self.assertEqual(endpoint.base_url, DEEPSEEK_BASE_URL)
            self.assertEqual(endpoint.api_key, "key")
        self.assertTrue(settings.send_thinking_param)
        self.assertEqual(settings.attack_model_name, "DeepSeek")

    def test_local_agent_needs_no_key_and_judge_reviewer_follow_it(self):
        settings = _settings(
            ["--model", "qwen2.5:7b", "--agent-base-url", OLLAMA_URL],
            {},
        )

        for endpoint in (settings.agent, settings.judge, settings.reviewer):
            self.assertEqual(endpoint.model, "qwen2.5:7b")
            self.assertEqual(endpoint.base_url, OLLAMA_URL)
            self.assertEqual(endpoint.api_key, LOCAL_PLACEHOLDER_KEY)
        self.assertFalse(settings.send_thinking_param)
        self.assertEqual(settings.attack_model_name, "Qwen")

    def test_judge_can_use_a_different_server_from_the_agent(self):
        settings = _settings(
            [
                "--model", "llama3.1:8b",
                "--agent-base-url", OLLAMA_URL,
                "--judge-model", "deepseek-flash",
                "--judge-base-url", DEEPSEEK_BASE_URL,
                "--judge-api-key-env", "DEEPSEEK_API_KEY",
            ],
            {"DEEPSEEK_API_KEY": "key"},
        )

        self.assertEqual(settings.judge.model, "deepseek-flash")
        self.assertEqual(settings.judge.api_key, "key")
        self.assertEqual(settings.reviewer.model, "llama3.1:8b")
        self.assertEqual(settings.reviewer.base_url, OLLAMA_URL)
        self.assertEqual(settings.attack_model_name, "Llama")

    def test_missing_key_for_a_remote_server_stops_the_run(self):
        with self.assertRaises(SystemExit):
            _settings([], {})

    def test_thinking_needs_the_deepseek_api(self):
        with self.assertRaises(SystemExit):
            _settings(
                ["--agent-base-url", OLLAMA_URL, "--agent-thinking"],
                {},
            )

    def test_explicit_attack_model_name_wins(self):
        settings = _settings(
            ["--attack-model-name", "Assistant"],
            {"DEEPSEEK_API_KEY": "key"},
        )

        self.assertEqual(settings.attack_model_name, "Assistant")


class AgentDojoRunnerTests(unittest.TestCase):
    def test_auto_complete_instruction_is_added(self):
        message = _system_message(auto_complete=True)

        self.assertIn("No human user is available", message)
        self.assertIn("Do not stop to ask for confirmation", message)

    def test_default_system_message_can_be_preserved(self):
        message = _system_message(auto_complete=False)

        self.assertNotIn("No human user is available", message)


def _allocator(argv):
    return build_allocator(build_parser().parse_args(argv))


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


if __name__ == "__main__":
    unittest.main()
