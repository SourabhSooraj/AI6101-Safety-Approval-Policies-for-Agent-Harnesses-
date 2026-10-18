import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .policy import Decision, PolicyResult

# allocator 自主决定时的放行阈值；q_estimator 用同一个值重建自动决定。
DEFAULT_ALLOW_THRESHOLD = 0.5

# CostBasedAllocator 的 tau0：一次人工审核的代价，与 utility_cost 同单位。
DEFAULT_COST_TAU0 = 0.05


class Allocator(Protocol):
    """HybridPolicy 使用的 P6 接口；BudgetAllocator 和 CostBasedAllocator 都实现它。"""

    def allocate(
        self,
        risk_score: float,
        impact: int,
        q: float,
        remaining_escalations: int,
        expected_remaining_risky_calls: int,
        force_escalation: bool = False,
        tool_name: str | None = None,
    ) -> PolicyResult:
        ...


class BudgetAllocator:
    """P6 allocator：在风险价值和有限 escalation 预算之间做取舍。"""

    def __init__(
        self,
        tau0: float = 1.0,
        allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
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
        tool_name: str | None = None,
    ) -> PolicyResult:
        # tool_name 只供 CostBasedAllocator 查 reviewer 错误率；这里不使用。
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

        would_escalate = force_escalation or value > tau

        # 项目要求：预算 k 用完后，任何本应升级的决定都视为 auto-deny。
        # 不需要升级的低 value 调用照常按风险阈值决定。
        # force_escalation 也属于本应升级，因此 Judge 失败时仍 fail closed。
        if would_escalate and remaining_escalations <= 0:
            return self._result(
                decision=Decision.DENY,
                reason="Call would be escalated but the escalation budget is exhausted.",
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


@dataclass(frozen=True)
class ReviewerErrorRates:
    """
    reviewer 的两类错误率，应在 dev 上用带标签的升级调用估计：
    - false_approve: P(reviewer APPROVE | 调用其实不安全)
    - false_deny: P(reviewer DENY | 调用其实安全)
    """

    false_approve: float
    false_deny: float

    def __post_init__(self) -> None:
        for field_name in ("false_approve", "false_deny"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")


# 没有 dev 估计时的占位值，不代表真实 reviewer。
DEFAULT_REVIEWER_ERRORS = ReviewerErrorRates(false_approve=0.1, false_deny=0.1)


class CostBasedAllocator:
    """
    按 Mozannar & Sontag (2020) 的 deferral 规则选择升级哪些调用：
    只有当 reviewer 决定的期望代价比自动决定低得多时才升级。

    代价（同一单位）：
    - 放行一个不安全调用：impact
    - 拒绝一个安全调用：utility_cost
    gain = 自动决定的期望代价 - reviewer 决定的期望代价；gain > tau 时升级。
    tau 与 BudgetAllocator 一样随预算紧张而升高。
    """

    def __init__(
        self,
        tau0: float = DEFAULT_COST_TAU0,
        utility_cost: float = 1.0,
        allow_threshold: float | None = DEFAULT_ALLOW_THRESHOLD,
        reviewer_errors: Mapping[str, ReviewerErrorRates] | None = None,
        default_reviewer_errors: ReviewerErrorRates = DEFAULT_REVIEWER_ERRORS,
    ) -> None:
        if tau0 < 0:
            raise ValueError("tau0 must not be negative")
        if utility_cost <= 0:
            raise ValueError("utility_cost must be positive")
        if allow_threshold is not None and not 0 <= allow_threshold <= 1:
            raise ValueError("allow_threshold must be between 0 and 1")

        self.tau0 = tau0
        self.utility_cost = utility_cost
        # None：自动决定选期望代价较小的一边；数字：沿用固定风险阈值。
        self.allow_threshold = allow_threshold
        self.reviewer_errors = dict(reviewer_errors or {})
        self.default_reviewer_errors = default_reviewer_errors

    def allocate(
        self,
        risk_score: float,
        impact: int,
        q: float,
        remaining_escalations: int,
        expected_remaining_risky_calls: int,
        force_escalation: bool = False,
        tool_name: str | None = None,
    ) -> PolicyResult:
        # q 只为接口兼容而接收；本 allocator 改用 reviewer 错误率。
        BudgetAllocator._validate_inputs(
            risk_score=risk_score,
            impact=impact,
            q=q,
            remaining_escalations=remaining_escalations,
            expected_remaining_risky_calls=expected_remaining_risky_calls,
        )

        allow_cost = risk_score * impact
        deny_cost = (1 - risk_score) * self.utility_cost
        auto_allows = self._auto_allows(risk_score, allow_cost, deny_cost)
        auto_cost = allow_cost if auto_allows else deny_cost

        errors = self.reviewer_errors.get(
            tool_name or "",
            self.default_reviewer_errors,
        )
        review_cost = (
            allow_cost * errors.false_approve
            + deny_cost * errors.false_deny
        )
        gain = auto_cost - review_cost

        effective_expected = max(1, expected_remaining_risky_calls)
        effective_budget = max(1, remaining_escalations)
        tau = self.tau0 * effective_expected / effective_budget

        costs = f"(auto cost {auto_cost:.3f}, review cost {review_cost:.3f})"
        would_escalate = force_escalation or gain > tau

        # 与 BudgetAllocator 相同：预算用完后，本应升级的调用一律拒绝。
        if would_escalate and remaining_escalations <= 0:
            return self._result(
                decision=Decision.DENY,
                reason=(
                    "Call would be escalated but the escalation budget is "
                    f"exhausted {costs}."
                ),
                policy_name="p6_budget_exhausted",
                risk_score=risk_score,
                impact=impact,
                q=q,
                gain=gain,
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
                gain=gain,
                tau=tau,
            )

        if gain > tau:
            return self._result(
                decision=Decision.REQUIRE_APPROVAL,
                reason=f"Review lowers expected cost by more than tau {costs}.",
                policy_name="p6_gain_above_tau",
                risk_score=risk_score,
                impact=impact,
                q=q,
                gain=gain,
                tau=tau,
            )

        if auto_allows:
            return self._result(
                decision=Decision.ALLOW,
                reason=f"Review is not worth its cost; auto-allowing {costs}.",
                policy_name="p6_auto_allow",
                risk_score=risk_score,
                impact=impact,
                q=q,
                gain=gain,
                tau=tau,
            )

        return self._result(
            decision=Decision.DENY,
            reason=f"Review is not worth its cost; auto-denying {costs}.",
            policy_name="p6_auto_deny",
            risk_score=risk_score,
            impact=impact,
            q=q,
            gain=gain,
            tau=tau,
        )

    def _auto_allows(
        self,
        risk_score: float,
        allow_cost: float,
        deny_cost: float,
    ) -> bool:
        if self.allow_threshold is None:
            return allow_cost < deny_cost
        return risk_score < self.allow_threshold

    @staticmethod
    def _result(
        decision: Decision,
        reason: str,
        policy_name: str,
        risk_score: float,
        impact: int,
        q: float,
        gain: float,
        tau: float,
    ) -> PolicyResult:
        # 日志中 value 字段记录 gain，便于与 BudgetAllocator 的 value 对照。
        return BudgetAllocator._result(
            decision=decision,
            reason=reason,
            policy_name=policy_name,
            risk_score=risk_score,
            impact=impact,
            q=q,
            value=gain,
            tau=tau,
        )


def load_reviewer_error_table(
    path: str | Path,
) -> dict[str, ReviewerErrorRates]:
    """
    读取 {tool_name: {"false_approve": x, "false_deny": y}} 格式的 JSON。
    键 "default" 表示表中没有的工具使用的错误率。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(
            "reviewer error table must be a JSON object of tool_name -> rates"
        )

    table: dict[str, ReviewerErrorRates] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"rates for {key!r} must be a JSON object")
        missing = {"false_approve", "false_deny"} - value.keys()
        if missing:
            raise ValueError(
                f"rates for {key!r} are missing {', '.join(sorted(missing))}"
            )
        try:
            table[str(key)] = ReviewerErrorRates(
                false_approve=value["false_approve"],
                false_deny=value["false_deny"],
            )
        except ValueError as error:
            raise ValueError(f"rates for {key!r}: {error}") from error
    return table
