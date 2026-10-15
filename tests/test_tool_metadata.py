import unittest

from harness.policy import Decision, PolicyRequest
from harness.policy_engine import PolicyEngine
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

    def _evaluate_update_password(self, user_request: str):
        return PolicyEngine().evaluate(
            PolicyRequest(
                tool_name="update_password",
                tool_call_id="call-1",
                arguments={"password": "new-password"},
                task_id="task",
                user_request=user_request,
                tool_metadata=BANKING_TOOL_METADATA["update_password"],
            )
        )

    def test_generic_change_request_does_not_unlock_password_update(self):
        result = self._evaluate_update_password("Please change my address.")

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p1_destructive_no_user_reason")

    def test_explicit_password_request_passes_destructive_rule(self):
        for user_request in ("Change my password.", "请帮我重置密码"):
            with self.subTest(user_request=user_request):
                result = self._evaluate_update_password(user_request)

                self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
                self.assertEqual(
                    result.policy_name,
                    "p1_external_or_irreversible",
                )


if __name__ == "__main__":
    unittest.main()
