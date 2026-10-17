import unittest

from scripts.run_policy_comparison import build_command, build_parser


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


if __name__ == "__main__":
    unittest.main()
