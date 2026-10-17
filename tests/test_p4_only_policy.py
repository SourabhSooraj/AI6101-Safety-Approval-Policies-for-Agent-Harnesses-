import unittest

from harness.cost import TokenUsage
from harness.llm_judge import JudgeResult
from harness.policies.p4_only import P4OnlyPolicy
from harness.policy import Decision, PolicyRequest, ToolMetadata


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self, result: JudgeResult) -> None:
        self.result = result

    def score(self, request: PolicyRequest) -> JudgeResult:
        return self.result


def judge_result(risk_score: float, success: bool = True, disagreement: bool = False):
    return JudgeResult(
        risk_score=risk_score,
        reason="test",
        samples=(),
        raw_responses=(),
        disagreement=disagreement,
        success=success,
        usage=TokenUsage(
            model="fake-judge",
            input_tokens=10,
            output_tokens=5,
        ),
    )


def request_for() -> PolicyRequest:
    return PolicyRequest(
        tool_name="send_money",
        tool_call_id="call-1",
        arguments={},
        task_id="task-1",
        tool_metadata=ToolMetadata(
            name="send_money",
            side_effect="write",
            scope="external",
            reversibility="irreversible",
            sensitive_args=("recipient", "amount"),
            impact=4,
        ),
    )


class P4OnlyPolicyTests(unittest.TestCase):
    def test_low_risk_is_allowed(self):
        policy = P4OnlyPolicy(
            judge=FakeJudge(judge_result(0.1)),
            t_low=0.3,
            t_high=0.7,
        )

        result = policy.evaluate(request_for())

        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.policy_name, "p4_low_risk_allow")
        self.assertEqual(result.input_tokens, 10)

    def test_high_risk_is_denied(self):
        policy = P4OnlyPolicy(
            judge=FakeJudge(judge_result(0.9)),
            t_low=0.3,
            t_high=0.7,
        )

        result = policy.evaluate(request_for())

        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.policy_name, "p4_high_risk_deny")

    def test_middle_risk_is_escalated(self):
        policy = P4OnlyPolicy(
            judge=FakeJudge(judge_result(0.5)),
            t_low=0.3,
            t_high=0.7,
        )

        result = policy.evaluate(request_for())

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p4_middle_risk_escalate")

    def test_unreliable_judge_is_escalated(self):
        policy = P4OnlyPolicy(
            judge=FakeJudge(judge_result(1.0, success=False)),
        )

        result = policy.evaluate(request_for())

        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertEqual(result.policy_name, "p4_judge_unreliable")


if __name__ == "__main__":
    unittest.main()
