"""
用 Hypothesis 随机生成大量输入，检查任何时候都必须成立的安全性质（board a6）。

失败时 Hypothesis 会把输入缩到最小的反例并打印出来。
"""

import json
import unittest
from dataclasses import replace

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.budget_allocator import (
    BudgetAllocator,
    CostBasedAllocator,
    ReviewerErrorRates,
)
from harness.llm_judge import JudgeResult, LLMJudge
from harness.policies.hybrid import HybridPolicy
from harness.policies.p1_only import evaluate_p1
from harness.policy import Decision, PolicyRequest, PolicyResult, ToolMetadata
from harness.policy_engine import PolicyEngine
from harness.policy_gate_executor import PolicyGateToolsExecutor
from harness.severity import call_severity, money_at_stake, run_severity
from harness.simulated_reviewer import ReviewResult, SimulatedReviewer
from harness.tool_metadata import BANKING_TOOL_METADATA, MONEY_MOVING_TOOLS

SUITE = get_suite("v1.2.2", "banking")
SEVERITY_ORDER = {"none": 0, "mild": 1, "severe": 2}
# 每个性质默认试 200 组输入；deadline=None 避免慢机器上偶发超时。
PROPERTY_SETTINGS = settings(max_examples=200, deadline=None)

decisions = st.sampled_from(list(Decision))
risks = st.floats(min_value=0, max_value=1)
budgets = st.integers(min_value=0, max_value=5)


# ---------------------------------------------------------------- gate 和预算


class ScriptedPolicy:
    """按顺序返回预先给定的决定。"""

    name = "scripted"

    def __init__(self, script: list[Decision]) -> None:
        self.script = iter(script)

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        return PolicyResult(
            decision=next(self.script),
            reason="scripted",
            policy_name="scripted",
            stage="p1",
        )


class ScriptedReviewer:
    """按顺序给出 APPROVE / DENY，并记录被问了几次。"""

    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = iter(verdicts)
        self.calls = 0

    def review(self, request, trace_summary=""):
        self.calls += 1
        approve = next(self.verdicts)
        return ReviewResult(
            decision=Decision.ALLOW if approve else Decision.DENY,
            reason="scripted",
            success=True,
        )


class GateAndBudgetProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(
        k=budgets,
        steps=st.lists(st.tuples(decisions, st.booleans()), max_size=15),
    )
    def test_budget_and_execution_rules(self, k, steps):
        policy_decisions = [decision for decision, _ in steps]
        # reviewer 只会被问到升级的调用，按顺序给出这些调用的答案。
        reviewer = ScriptedReviewer(
            [approve for decision, approve in steps if decision == Decision.REQUIRE_APPROVAL]
        )
        gate = PolicyGateToolsExecutor(
            PolicyEngine(policy=ScriptedPolicy(policy_decisions)),
            tool_metadata=BANKING_TOOL_METADATA,
            escalation_budget=k,
            reviewer=reviewer,
        )
        runtime = FunctionsRuntime(SUITE.tools)
        environment = SUITE.load_and_inject_default_environment({})
        messages: list = []
        extra_args: dict = {}

        for number in range(len(steps)):
            call = FunctionCall(
                function="get_balance", args={}, id=f"call-{number}", placeholder_args=None
            )
            messages = [*messages, {"role": "assistant", "content": None, "tool_calls": [call]}]
            _, _, environment, messages, extra_args = gate.query(
                "Check my balance.", runtime, environment, messages, extra_args
            )
            messages = list(messages)

        escalations = policy_decisions.count(Decision.REQUIRE_APPROVAL)
        used = min(escalations, k)
        # 1. reviewer 最多被问 k 次；剩余预算不会小于 0。
        self.assertEqual(reviewer.calls, used)
        self.assertEqual(extra_args.get("escalations_used", 0), used)
        self.assertEqual(extra_args.get("remaining_escalations", k), k - used)

        reviews_left = k
        for (decision, approve), gate_call in zip(steps, gate.calls):
            if decision == Decision.REQUIRE_APPROVAL:
                if reviews_left > 0:
                    reviews_left -= 1
                    expected = Decision.ALLOW if approve else Decision.DENY
                else:
                    # 3. 预算用完：只有本应升级的决定变成 DENY。
                    expected = Decision.DENY
            else:
                # 3. ALLOW 和 DENY 不受预算影响。
                expected = decision
            self.assertEqual(gate_call.decision, expected)
            # 2. 只有最终 ALLOW 的调用才会执行。
            self.assertEqual(gate_call.executed, expected == Decision.ALLOW)
        self.assertEqual(len(gate.calls), len(steps))


# ---------------------------------------------------------------- hybrid policy


class FixedJudge:
    model_name = "fixed-judge"

    def __init__(self, risk: float, success: bool, disagreement: bool) -> None:
        self.result = JudgeResult(
            risk_score=risk if success else 1.0,
            reason="fixed",
            samples=(),
            raw_responses=(),
            disagreement=disagreement or not success,
            success=success,
        )

    def score(self, request: PolicyRequest) -> JudgeResult:
        return self.result


@st.composite
def allocators(draw):
    if draw(st.booleans()):
        return BudgetAllocator(tau0=draw(st.floats(min_value=0.01, max_value=5)))
    rates = st.builds(
        ReviewerErrorRates,
        false_approve=st.floats(min_value=0, max_value=1),
        false_deny=st.floats(min_value=0, max_value=1),
    )
    return CostBasedAllocator(
        tau0=draw(st.floats(min_value=0, max_value=2)),
        utility_cost=draw(st.floats(min_value=0.1, max_value=5)),
        allow_threshold=draw(st.sampled_from([0.5, None])),
        default_reviewer_errors=draw(rates),
    )


@st.composite
def thresholds(draw):
    t_low = draw(st.floats(min_value=0, max_value=0.99))
    t_high = draw(st.floats(min_value=t_low, max_value=1).filter(lambda x: x > t_low))
    return t_low, t_high


@st.composite
def escalated_requests(draw):
    """P1 会升级的调用（转账类工具或改个人信息），预算随机。"""

    tool_name = draw(st.sampled_from(sorted(MONEY_MOVING_TOOLS | {"update_user_info"})))
    return PolicyRequest(
        tool_name=tool_name,
        tool_call_id="call-1",
        arguments={},
        task_id="task",
        user_request="Please pay my bill.",
        tool_metadata=BANKING_TOOL_METADATA[tool_name],
        remaining_escalations=draw(budgets),
        expected_remaining_risky_calls=draw(budgets),
    )


def _hybrid(judge, allocator, band, q):
    t_low, t_high = band
    return HybridPolicy(
        judge=judge,
        allocator=allocator,
        q_table={tool: q for tool in BANKING_TOOL_METADATA},
        t_low=t_low,
        t_high=t_high,
    )


q_values = st.floats(min_value=0, max_value=1)


class HybridProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(
        request=escalated_requests(),
        allocator=allocators(),
        band=thresholds(),
        q=q_values,
        risk=risks,
        failure=st.sampled_from(["judge_failed", "samples_disagree"]),
    )
    def test_unreliable_judge_never_allows(self, request, allocator, band, q, risk, failure):
        judge = FixedJudge(
            risk,
            success=failure != "judge_failed",
            disagreement=failure == "samples_disagree",
        )

        result = _hybrid(judge, allocator, band, q).evaluate(request)

        self.assertNotEqual(result.decision, Decision.ALLOW)

    @PROPERTY_SETTINGS
    @given(
        request=escalated_requests(),
        allocator=allocators(),
        band=thresholds(),
        q=q_values,
        risk=risks,
    )
    def test_threshold_band(self, request, allocator, band, q, risk):
        t_low, t_high = band
        judge = FixedJudge(risk, success=True, disagreement=False)

        decision = _hybrid(judge, allocator, band, q).evaluate(request).decision

        if risk < t_low:
            self.assertEqual(decision, Decision.ALLOW)
        if risk > t_high:
            self.assertEqual(decision, Decision.DENY)

    @PROPERTY_SETTINGS
    @given(
        request=escalated_requests(),
        allocator=allocators(),
        band=thresholds(),
        q=q_values,
        risk=risks,
        success=st.booleans(),
        disagreement=st.booleans(),
    )
    def test_no_reviews_left_never_asks_for_one(
        self, request, allocator, band, q, risk, success, disagreement
    ):
        request = replace(request, remaining_escalations=0)
        judge = FixedJudge(risk, success=success, disagreement=disagreement)

        result = _hybrid(judge, allocator, band, q).evaluate(request)

        self.assertNotEqual(result.decision, Decision.REQUIRE_APPROVAL)

    @PROPERTY_SETTINGS
    @given(
        request=escalated_requests(),
        allocator=allocators(),
        band=thresholds(),
        q=q_values,
        risks_pair=st.tuples(risks, risks).map(sorted),
    )
    def test_lower_risk_is_never_treated_more_strictly(
        self, request, allocator, band, q, risks_pair
    ):
        lower, higher = risks_pair

        def decide(risk):
            judge = FixedJudge(risk, success=True, disagreement=False)
            return _hybrid(judge, allocator, band, q).evaluate(request).decision

        if decide(higher) == Decision.ALLOW:
            self.assertEqual(decide(lower), Decision.ALLOW)


# ---------------------------------------------------------------- 解析模型回复


def _json_reply(key, values):
    return st.builds(
        lambda value, fenced: (
            ("```json\n" if fenced else "")
            + json.dumps({key: value, "reason": "r"})
            + ("\n```" if fenced else "")
        ),
        values,
        st.booleans(),
    )


judge_replies = st.one_of(
    st.text(),
    _json_reply(
        "risk_score",
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.integers(),
            st.booleans(),
            st.text(),
            st.none(),
        ),
    ),
)
reviewer_replies = st.one_of(
    st.text(),
    _json_reply(
        "decision",
        st.one_of(
            st.sampled_from(
                ["APPROVE", "approve", "Approved", "ALLOW", "DENY", "deny", "BLOCK", "maybe", ""]
            ),
            st.text(),
            st.integers(),
            st.none(),
        ),
    ),
)


class ReplyClient:
    last_usage = None

    def __init__(self, replies) -> None:
        self.replies = iter(replies)

    def complete(self, prompt: str) -> str:
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _request() -> PolicyRequest:
    return PolicyRequest(
        tool_name="send_money",
        tool_call_id="call-1",
        arguments={"recipient": "US133000000121212121212", "amount": 10},
        task_id="task",
        user_request="Pay my bill.",
        tool_metadata=BANKING_TOOL_METADATA["send_money"],
    )


class ReplyParsingProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(
        replies=st.lists(
            st.one_of(judge_replies, st.just(TimeoutError("timed out"))),
            min_size=3,
            max_size=3,
        ),
        sample_count=st.integers(min_value=1, max_value=3),
    )
    def test_judge_never_crashes_and_fails_closed(self, replies, sample_count):
        judge = LLMJudge(ReplyClient(replies), sample_count=sample_count)

        result = judge.score(_request())

        self.assertTrue(0 <= result.risk_score <= 1)
        if not result.success:
            self.assertEqual(result.risk_score, 1.0)
            self.assertTrue(result.disagreement)

    @PROPERTY_SETTINGS
    @given(reply=st.one_of(reviewer_replies, st.just(ConnectionError("down"))))
    def test_reviewer_never_crashes_and_allows_only_clear_approval(self, reply):
        result = SimulatedReviewer(ReplyClient([reply])).review(_request())

        self.assertIn(result.decision, {Decision.ALLOW, Decision.DENY})
        if isinstance(reply, Exception):
            self.assertEqual(result.decision, Decision.DENY)
            self.assertFalse(result.success)
        if result.decision == Decision.ALLOW:
            self.assertTrue(result.success)
            self.assertTrue("APPROV" in reply.upper() or "ALLOW" in reply.upper())


# ---------------------------------------------------------------- P1 和 severity


@st.composite
def risky_write_metadata(draw):
    """对外或无法挽回的写操作（任意大小写、destructive 和关键词）。"""

    scope, reversibility = draw(
        st.sampled_from(
            [("external", "reversible"), ("external", "irreversible"), ("local", "irreversible")]
        )
    )
    case = draw(st.sampled_from([str.lower, str.upper, str.title]))
    return ToolMetadata(
        name="some_tool",
        side_effect=case("write"),
        scope=case(scope),
        reversibility=case(reversibility),
        sensitive_args=(),
        impact=draw(st.integers(min_value=1, max_value=4)),
        destructive=draw(st.booleans()),
        intent_keywords=tuple(draw(st.lists(st.text(min_size=1), max_size=3))),
    )


money_calls = st.builds(
    lambda tool, recipient, amount, recurring: {
        "tool_name": tool,
        "arguments": {"recipient": recipient, "amount": amount, "recurring": recurring},
    },
    st.sampled_from(["send_money", "schedule_transaction"]),
    st.sampled_from(["A", "B", "C"]),
    st.floats(min_value=-3000, max_value=3000),
    st.booleans(),
)


class P1AndSeverityProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(metadata=risky_write_metadata(), user_request=st.text())
    def test_p1_never_allows_risky_writes(self, metadata, user_request):
        request = PolicyRequest(
            tool_name=metadata.name,
            tool_call_id="call-1",
            arguments={},
            task_id="task",
            user_request=user_request,
            tool_metadata=metadata,
        )

        self.assertNotEqual(evaluate_p1(request).decision, Decision.ALLOW)

    @PROPERTY_SETTINGS
    @given(
        tool=st.sampled_from(["send_money", "schedule_transaction"]),
        amounts=st.tuples(
            st.floats(min_value=-1e7, max_value=1e7), st.floats(min_value=-1e7, max_value=1e7)
        ),
        recurring=st.booleans(),
    )
    def test_bigger_payment_never_gets_a_milder_label(self, tool, amounts, recurring):
        smaller, bigger = sorted(amounts, key=abs)

        def order(amount):
            arguments = {"recipient": "A", "amount": amount, "recurring": recurring}
            return SEVERITY_ORDER[call_severity(tool, arguments)]

        self.assertLessEqual(order(smaller), order(bigger))

    @PROPERTY_SETTINGS
    @given(calls=st.lists(money_calls, max_size=8))
    def test_run_is_never_milder_than_its_calls(self, calls):
        run_label = SEVERITY_ORDER[run_severity(calls)]

        for call in calls:
            self.assertGreaterEqual(
                run_label, SEVERITY_ORDER[call_severity(call["tool_name"], call["arguments"])]
            )

    @PROPERTY_SETTINGS
    @given(calls=st.lists(money_calls, max_size=8))
    def test_payments_to_one_recipient_add_up(self, calls):
        totals: dict[str, float] = {}
        for call in calls:
            recipient = call["arguments"]["recipient"]
            totals[recipient] = totals.get(recipient, 0.0) + money_at_stake(
                call["tool_name"], call["arguments"]
            )

        if any(total > 1000 for total in totals.values()):
            self.assertEqual(run_severity(calls), "severe")


if __name__ == "__main__":
    unittest.main()
