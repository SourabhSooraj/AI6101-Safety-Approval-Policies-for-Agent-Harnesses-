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
    """baseline：只读调用放行，其余调用（含未知工具）全部交给 reviewer。"""

    name = "escalate_all"

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        metadata = request.tool_metadata
        if metadata is not None and metadata.side_effect.casefold() == "read":
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="Read-only tool call.",
                policy_name="baseline_read_only",
                policy_variant=self.name,
                risk_score=0.0,
                impact=metadata.impact,
                stage="baseline",
            )

        # 不经过 P1 的 destructive 规则，让 reviewer 看到每一个写操作。
        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="Baseline sends every non-read call to the reviewer.",
            policy_name="baseline_escalate_write",
            policy_variant=self.name,
            risk_score=1.0,
            impact=metadata.impact if metadata is not None else 3,
            stage="baseline",
        )
