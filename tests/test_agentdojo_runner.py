import unittest

from scripts.run_agentdojo_e2e import _system_message


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
