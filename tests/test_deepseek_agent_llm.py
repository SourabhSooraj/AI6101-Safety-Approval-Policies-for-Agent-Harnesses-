import unittest

from harness.deepseek_agent_llm import _message_to_deepseek


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


if __name__ == "__main__":
    unittest.main()
