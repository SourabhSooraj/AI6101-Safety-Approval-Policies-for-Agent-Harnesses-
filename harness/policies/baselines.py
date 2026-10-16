from ..policy import Decision, PolicyRequest, PolicyResult


class AllowAllPolicy:
    """baseline：没有 gate，所有工具调用直接执行。"""

    name = "allow_all"

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        metadata = request.tool_metadata
        return PolicyResult(
            decision=Decision.ALLOW,
            reason="Baseline allows every call.",
            policy_name="baseline_allow_all",
            policy_variant=self.name,
            risk_score=0.0,
            impact=metadata.impact if metadata is not None else 3,
            stage="baseline",
        )


class EscalateAllPolicy:
    """
    项目要求中的 full-escalate baseline：每一个工具调用（包括只读调用）都交给
    reviewer。预算 k 与其他 policy 相同，用完后 gate 会把后续升级转为 DENY。
    """

    name = "escalate_all"

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        metadata = request.tool_metadata
        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="Baseline escalates every tool call to the reviewer.",
            policy_name="baseline_escalate_all",
            policy_variant=self.name,
            risk_score=1.0,
            impact=metadata.impact if metadata is not None else 3,
            stage="baseline",
        )
