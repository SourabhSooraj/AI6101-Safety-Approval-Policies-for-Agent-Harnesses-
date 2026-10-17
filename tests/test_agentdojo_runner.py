import unittest

from scripts.run_agentdojo_e2e import (
    DEEPSEEK_BASE_URL,
    LOCAL_PLACEHOLDER_KEY,
    _system_message,
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


if __name__ == "__main__":
    unittest.main()
