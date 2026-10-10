import unittest

from harness.policy import Decision, PolicyRequest, ToolMetadata
from harness.policy_engine import PolicyEngine


def request_for(
    tool_name: str,
    metadata: ToolMetadata | None,
    user_request: str = "",
) -> PolicyRequest:
    return PolicyRequest(
        tool_name=tool_name,
        tool_call_id="call-1",
        arguments={},
        task_id="test-task",
        user_request=user_request,
        tool_metadata=metadata,
    )


class PolicyEngineP1Tests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_read_only_is_allowed(self):
        metadata = ToolMetadata(
            name="get_balance",
            side_effect="read",
            scope="local",
            reversibility="reversible",
            sensitive_args=(),
            impact=1,
        )

        result = self.engine.evaluate(request_for("get_balance", metadata))

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p1_read_only")

    def test_local_reversible_write_is_allowed(self):
        metadata = ToolMetadata(
            name="create_calendar_event",
            side_effect="write",
            scope="local",
            reversibility="reversible",
            sensitive_args=("start_time",),
            impact=2,
        )

        result = self.engine.evaluate(
            request_for("create_calendar_event", metadata)
        )

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p1_local_reversible_write")

    def test_external_write_requires_approval(self):
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        )

        result = self.engine.evaluate(request_for("send_money", metadata))

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p1_external_or_irreversible")

    def test_destructive_without_reason_is_denied(self):
        metadata = ToolMetadata(
            name="delete_file",
            side_effect="write",
            scope="local",
            reversibility="irreversible",
            sensitive_args=("file_id",),
            impact=3,
            destructive=True,
            intent_keywords=("delete", "删除"),
        )

        result = self.engine.evaluate(
            request_for("delete_file", metadata, "Summarize this email.")
        )

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(
            result.policy_name,
            "p1_destructive_no_user_reason",
        )

    def test_destructive_with_reason_requires_approval(self):
        metadata = ToolMetadata(
            name="delete_file",
            side_effect="write",
            scope="local",
            reversibility="irreversible",
            sensitive_args=("file_id",),
            impact=3,
            destructive=True,
            intent_keywords=("delete", "删除"),
        )

        result = self.engine.evaluate(
            request_for("delete_file", metadata, "请删除旧报告。")
        )

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p1_external_or_irreversible")

    def test_missing_metadata_requires_approval(self):
        result = self.engine.evaluate(request_for("unknown_tool", None))

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p1_unknown_tool")


if __name__ == "__main__":
    unittest.main()
