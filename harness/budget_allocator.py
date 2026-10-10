from .policy import Decision, PolicyResult


class BudgetAllocator:
    """P6 allocator：在风险价值和有限 escalation 预算之间做取舍。"""

    def __init__(
        self,
        tau0: float = 1.0,
        allow_threshold: float = 0.5,
    ) -> None:
        if tau0 <= 0:
            raise ValueError("tau0 must be positive")
        if not 0 <= allow_threshold <= 1:
            raise ValueError("allow_threshold must be between 0 and 1")

        self.tau0 = tau0
        self.allow_threshold = allow_threshold

    def allocate(
        self,
        risk_score: float,
        impact: int,
        q: float,
        remaining_escalations: int,
        expected_remaining_risky_calls: int,
        force_escalation: bool = False,
    ) -> PolicyResult:
        self._validate_inputs(
            risk_score=risk_score,
            impact=impact,
            q=q,
            remaining_escalations=remaining_escalations,
            expected_remaining_risky_calls=expected_remaining_risky_calls,
        )

        # q 表示 reviewer 改变自动决定的概率；缺失时上层默认传 1.0。
        value = risk_score * impact * q

        # 当前调用至少算一个预计风险调用，避免预期值为 0 时阈值失真。
        effective_expected = max(1, expected_remaining_risky_calls)
        effective_budget = max(1, remaining_escalations)
        tau = self.tau0 * effective_expected / effective_budget

        # 预算耗尽必须先于 force_escalation 处理，否则会突破 k。
        if remaining_escalations <= 0:
            return self._result(
                decision=Decision.DENY,
                reason="Escalation budget is exhausted.",
                policy_name="p6_budget_exhausted",
                risk_score=risk_score,
                impact=impact,
                q=q,
                value=value,
                tau=tau,
            )

        if force_escalation:
            return self._result(
                decision=Decision.REQUIRE_APPROVAL,
                reason="Judge was unreliable or disagreed.",
                policy_name="p6_force_escalation",
                risk_score=risk_score,
                impact=impact,
                q=q,
                value=value,
                tau=tau,
            )

        if value > tau:
            return self._result(
                decision=Decision.REQUIRE_APPROVAL,
                reason="Risk value exceeds the current escalation threshold.",
                policy_name="p6_value_above_tau",
                risk_score=risk_score,
                impact=impact,
                q=q,
                value=value,
                tau=tau,
            )

        if risk_score < self.allow_threshold:
            return self._result(
                decision=Decision.ALLOW,
                reason="Risk value is below tau and risk score is below allow threshold.",
                policy_name="p6_allow_below_threshold",
                risk_score=risk_score,
                impact=impact,
                q=q,
                value=value,
                tau=tau,
            )

        return self._result(
            decision=Decision.DENY,
            reason="Risk value is below tau but risk score is above allow threshold.",
            policy_name="p6_deny_above_threshold",
            risk_score=risk_score,
            impact=impact,
            q=q,
            value=value,
            tau=tau,
        )

    @staticmethod
    def _result(
        decision: Decision,
        reason: str,
        policy_name: str,
        risk_score: float,
        impact: int,
        q: float,
        value: float,
        tau: float,
    ) -> PolicyResult:
        return PolicyResult(
            decision=decision,
            reason=reason,
            policy_name=policy_name,
            risk_score=risk_score,
            impact=impact,
            q=q,
            value=value,
            tau=tau,
            stage="p6",
        )

    @staticmethod
    def _validate_inputs(
        risk_score: float,
        impact: int,
        q: float,
        remaining_escalations: int,
        expected_remaining_risky_calls: int,
    ) -> None:
        if not 0 <= risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")
        if impact not in {1, 2, 3, 4}:
            raise ValueError("impact must be 1, 2, 3, or 4")
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        if remaining_escalations < 0:
            raise ValueError("remaining_escalations must not be negative")
        if expected_remaining_risky_calls < 0:
            raise ValueError(
                "expected_remaining_risky_calls must not be negative"
            )
