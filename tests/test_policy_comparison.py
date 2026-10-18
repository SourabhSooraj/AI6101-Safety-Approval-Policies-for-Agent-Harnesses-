import unittest

from scripts.run_policy_comparison import (
    allocator_settings,
    build_command,
    build_parser,
)


class PolicyComparisonCommandTests(unittest.TestCase):
    def test_unset_options_are_left_to_the_runner_defaults(self):
        args = build_parser().parse_args([])

        command = build_command(args, "hybrid", 3, "user_task_0", "none")

        self.assertEqual(
            command[2:],
            ["--policy", "hybrid", "--k", "3", "--task-id", "user_task_0"],
        )

    def test_model_options_are_passed_through(self):
        args = build_parser().parse_args(
            [
                "--model", "qwen2.5:7b",
                "--agent-base-url", "http://localhost:11434/v1",
                "--judge-model", "deepseek-flash",
                "--judge-base-url", "https://api.deepseek.com",
                "--judge-api-key-env", "DEEPSEEK_API_KEY",
                "--reviewer-model", "llama3.1:8b",
                "--attack-model-name", "Qwen",
                "--sample-count", "2",
            ]
        )

        command = build_command(
            args, "allow_all", 0, "user_task_3", "injection_task_1"
        )

        def value_of(option):
            return command[command.index(option) + 1]

        self.assertEqual(value_of("--model"), "qwen2.5:7b")
        self.assertEqual(value_of("--agent-base-url"), "http://localhost:11434/v1")
        self.assertEqual(value_of("--judge-model"), "deepseek-flash")
        self.assertEqual(value_of("--judge-base-url"), "https://api.deepseek.com")
        self.assertEqual(value_of("--judge-api-key-env"), "DEEPSEEK_API_KEY")
        self.assertEqual(value_of("--reviewer-model"), "llama3.1:8b")
        self.assertEqual(value_of("--attack-model-name"), "Qwen")
        self.assertEqual(value_of("--sample-count"), "2")
        self.assertEqual(value_of("--injection-task"), "injection_task_1")
        self.assertNotIn("--agent-thinking", command)

    def test_provider_options_are_passed_through(self):
        args = build_parser().parse_args(
            [
                "--provider", "ollama",
                "--request-timeout", "600",
                "--judge-temperature", "0.7",
            ]
        )

        command = build_command(args, "hybrid", 3, "user_task_0", "none")

        def value_of(option):
            return command[command.index(option) + 1]

        self.assertEqual(value_of("--provider"), "ollama")
        self.assertEqual(value_of("--request-timeout"), "600")
        self.assertEqual(value_of("--judge-temperature"), "0.7")

    def test_allocator_options_are_passed_through(self):
        args = build_parser().parse_args(
            [
                "--utility-cost", "2",
                "--auto-rule", "cost",
                "--reviewer-error-table", "errors.json",
            ]
        )

        command = build_command(
            args, "hybrid", 1, "user_task_0", "none", "cost", 0.1
        )

        def value_of(option):
            return command[command.index(option) + 1]

        self.assertEqual(value_of("--allocator"), "cost")
        self.assertEqual(value_of("--tau0"), "0.1")
        self.assertEqual(value_of("--utility-cost"), "2")
        self.assertEqual(value_of("--auto-rule"), "cost")
        self.assertEqual(value_of("--reviewer-error-table"), "errors.json")

    def test_sweep_covers_p6_policies_only(self):
        args = build_parser().parse_args(
            [
                "--allocators", "value", "cost",
                "--tau0-values", "0.05", "0.2",
            ]
        )

        self.assertEqual(
            allocator_settings(args, "hybrid"),
            [("value", 0.05), ("value", 0.2), ("cost", 0.05), ("cost", 0.2)],
        )
        self.assertEqual(allocator_settings(args, "p1_only"), [(None, None)])

    def test_no_sweep_keeps_runner_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(allocator_settings(args, "hybrid"), [(None, None)])
        command = build_command(args, "hybrid", 1, "user_task_0", "none")
        self.assertNotIn("--allocator", command)
        self.assertNotIn("--tau0", command)

if __name__ == "__main__":
    unittest.main()
