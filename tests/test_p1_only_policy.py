import unittest

from harness.policies.p1_only import P1OnlyPolicy, evaluate_p1
from harness.policy import Decision, PolicyRequest, ToolMetadata


def request_for(tool_name: str, metadata: ToolMetadata, user_request=""):
    return PolicyRequest(
        tool_name=tool_name,
        tool_call_id="call-1",
        arguments={},
        task_id="task-1",
        user_request=user_request,
        tool_metadata=metadata,
    )


class P1OnlyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = P1OnlyPolicy()

    def test_read_only_is_allowed(self):
        metadata = ToolMetadata(
            name="get_balance",
            side_effect="read",
            scope="local",
            reversibility="reversible",
            sensitive_args=(),
            impact=1,
        )

        result = self.policy.evaluate(request_for("get_balance", metadata))

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_variant, "p1_only")

    def test_external_write_is_escalated(self):
        metadata = ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        )

        result = evaluate_p1(request_for("send_money", metadata))

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
            intent_keywords=("delete",),
        )

        result = self.policy.evaluate(
            request_for("delete_file", metadata, "Summarize my inbox.")
        )

        self.assertEqual(result.decision, Decision.DENY)


if __name__ == "__main__":
    unittest.main()
