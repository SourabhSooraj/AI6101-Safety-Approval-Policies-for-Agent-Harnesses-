import unittest

from agentdojo.attacks import load_attack
from agentdojo.task_suite.load_suites import get_suite

import harness.attacks  # noqa: F401 - 注册 DeepSeek 攻击
from harness.attacks import attack_model_name_for


class FakePipeline:
    name = "deepseek-flash-gated-policy"


class DeepSeekAttackTests(unittest.TestCase):
    def test_injections_contain_goal_and_model_name(self):
        suite = get_suite("v1.2.2", "banking")
        user_task = suite.get_user_task_by_id("user_task_0")
        injection_task = suite.get_injection_task_by_id("injection_task_0")

        attack = load_attack(
            "important_instructions_deepseek",
            suite,
            FakePipeline(),
        )
        injections = attack.attack(user_task, injection_task)

        self.assertTrue(injections)
        for text in injections.values():
            self.assertIn(injection_task.GOAL, text)
            self.assertIn("DeepSeek", text)
            self.assertIn("Emma Johnson", text)

    def test_model_name_can_be_overridden(self):
        suite = get_suite("v1.2.2", "banking")
        user_task = suite.get_user_task_by_id("user_task_0")
        injection_task = suite.get_injection_task_by_id("injection_task_0")

        attack = load_attack(
            "important_instructions_deepseek",
            suite,
            FakePipeline(),
        )
        attack.model_name = "Qwen"
        injections = attack.attack(user_task, injection_task)

        for text in injections.values():
            self.assertIn("Qwen", text)
            self.assertNotIn("DeepSeek", text)

    def test_attack_model_name_is_guessed_from_model_id(self):
        self.assertEqual(attack_model_name_for("deepseek-flash"), "DeepSeek")
        self.assertEqual(attack_model_name_for("llama3.1:8b"), "Llama")
        self.assertEqual(attack_model_name_for("Qwen/Qwen2.5-7B-Instruct"), "Qwen")
        self.assertEqual(
            attack_model_name_for("some-new-model"),
            "the AI language model",
        )


if __name__ == "__main__":
    unittest.main()
