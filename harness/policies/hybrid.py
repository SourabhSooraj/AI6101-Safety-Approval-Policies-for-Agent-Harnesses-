from dataclasses import replace

from ..budget_allocator import BudgetAllocator
from ..llm_judge import LLMJudge
from ..policy import Decision, PolicyRequest, PolicyResult
from .base import attach_judge_metadata
from .p1_only import evaluate_p1


class HybridPolicy:
    """P1 rulebase + LLM Judge + P6 budget allocator 主策略。"""

    name = "hybrid"

    def __init__(
        self,
        judge: LLMJudge,
        allocator: BudgetAllocator,
        q_table: dict[str, float] | None = None,
        t_low: float = 0.3,
        t_high: float = 0.7,
    ) -> None:
        if not 0 <= t_low < t_high <= 1:
            raise ValueError("thresholds must satisfy 0 <= t_low < t_high <= 1")

        self.judge = judge
        self.allocator = allocator
        self.q_table = dict(q_table or {})
        self.t_low = t_low
        self.t_high = t_high

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        p1_result = evaluate_p1(request)
        if p1_result.decision != Decision.REQUIRE_APPROVAL:
            return p1_result

        judge_result = self.judge.score(request)
        impact = (
            request.tool_metadata.impact
            if request.tool_metadata is not None
            else 3
        )
        q = self.q_table.get(request.tool_name, 1.0)

        if (
            judge_result.success
            and not judge_result.disagreement
            and judge_result.risk_score < self.t_low
        ):
            return attach_judge_metadata(
                PolicyResult(
                    decision=Decision.ALLOW,
                    reason="Judge risk score is below t_low.",
                    policy_name="hybrid_low_risk_allow",
                    policy_variant=self.name,
                    risk_score=judge_result.risk_score,
                    impact=impact,
                    stage="judge",
                ),
                judge_result,
                fallback_model=self.judge.model_name,
            )

        if (
            judge_result.success
            and not judge_result.disagreement
            and judge_result.risk_score > self.t_high
        ):
            return attach_judge_metadata(
                PolicyResult(
                    decision=Decision.DENY,
                    reason="Judge risk score is above t_high.",
                    policy_name="hybrid_high_risk_deny",
                    policy_variant=self.name,
                    risk_score=judge_result.risk_score,
                    impact=impact,
                    stage="judge",
                ),
                judge_result,
                fallback_model=self.judge.model_name,
            )

        allocated_result = self.allocator.allocate(
            risk_score=judge_result.risk_score,
            impact=impact,
            q=q,
            remaining_escalations=request.remaining_escalations,
            expected_remaining_risky_calls=(
                request.expected_remaining_risky_calls
            ),
            force_escalation=(
                not judge_result.success or judge_result.disagreement
            ),
        )
        allocated_result = replace(
            allocated_result,
            policy_variant=self.name,
        )
        return attach_judge_metadata(
            allocated_result,
            judge_result,
            fallback_model=self.judge.model_name,
        )
