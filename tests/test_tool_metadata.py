import unittest

from harness.tool_metadata import BANKING_TOOL_METADATA


class ToolMetadataTests(unittest.TestCase):
    def test_banking_tools_are_covered(self):
        expected = {
            "get_iban",
            "get_balance",
            "get_most_recent_transactions",
            "get_scheduled_transactions",
            "read_file",
            "get_user_info",
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
            "update_password",
            "update_user_info",
        }

        self.assertEqual(set(BANKING_TOOL_METADATA), expected)

    def test_sensitive_writes_have_high_impact(self):
        self.assertEqual(BANKING_TOOL_METADATA["send_money"].impact, 4)
        self.assertEqual(BANKING_TOOL_METADATA["update_password"].impact, 4)


if __name__ == "__main__":
    unittest.main()
