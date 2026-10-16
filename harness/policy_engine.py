from .budget_allocator import BudgetAllocator
from .decision_logger import DecisionLogger
from .llm_judge import LLMJudge
from .policies import PolicyEvaluator, P1OnlyPolicy, create_policy
from .policy import Decision, PolicyRequest, PolicyResult


class PolicyEngine:
    """兼容入口：应用显式覆盖后，把请求委托给选定的 Policy 变体。"""

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        judge: LLMJudge | None = None,
        allocator: BudgetAllocator | None = None,
        q_table: dict[str, float] | None = None,
        decision_logger: DecisionLogger | None = None,
        escalate_writes: bool = False,
        policy: PolicyEvaluator | None = None,
        policy_variant: str | None = None,
        t_low: float = 0.3,
        t_high: float = 0.7,
    ) -> None:
        self.allowed_tools = set(allowed_tools or ())
        self.denied_tools = set(denied_tools or ())
        self.decision_logger = decision_logger
        self.escalate_writes = escalate_writes

        if policy is not None:
            self.policy = policy
        elif escalate_writes:
            # baseline 会绕过策略，evaluate() 中单独处理。
            self.policy = P1OnlyPolicy()
        elif policy_variant is not None:
            self.policy = create_policy(
                policy_variant,
                judge=judge,
                allocator=allocator,
                q_table=q_table,
                t_low=t_low,
                t_high=t_high,
            )
        elif judge is not None and allocator is not None:
            self.policy = create_policy(
                "hybrid",
                judge=judge,
                allocator=allocator,
                q_table=q_table,
                t_low=t_low,
                t_high=t_high,
            )
        else:
            # 保留旧测试和 P1-only 开发场景：未提供 Judge/allocator 时只运行 P1。
            self.policy = P1OnlyPolicy()

    @property
    def policy_variant(self) -> str:
        if self.escalate_writes:
            return "escalate_all"
        return self.policy.name

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        result = self._apply_explicit_overrides(request)
        if result is None and self.escalate_writes:
            result = self._evaluate_escalate_writes(request)
        if result is None:
            result = self.policy.evaluate(request)

        if self.decision_logger is not None:
            self.decision_logger.log(request, result)
        return result

    def _apply_explicit_overrides(
        self,
        request: PolicyRequest,
    ) -> PolicyResult | None:
        if request.tool_name in self.denied_tools:
            return PolicyResult(
                decision=Decision.DENY,
                reason="This tool is explicitly denied.",
                policy_name="explicit_tool_deny",
                policy_variant="explicit",
                risk_score=1.0,
                impact=3,
                stage="explicit",
            )

        if request.tool_name in self.allowed_tools:
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="This tool is explicitly allowed.",
                policy_name="explicit_tool_allow",
                policy_variant="explicit",
                risk_score=0.0,
                impact=1,
                stage="explicit",
            )

        return None

    @staticmethod
    def _evaluate_escalate_writes(
        request: PolicyRequest,
    ) -> PolicyResult:
        metadata = request.tool_metadata
        if metadata is not None and metadata.side_effect.casefold() == "read":
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="Read-only tool call.",
                policy_name="baseline_read_only",
                policy_variant="escalate_all",
                risk_score=0.0,
                impact=metadata.impact,
                stage="baseline",
            )

        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="Baseline sends every non-read call to the reviewer.",
            policy_name="baseline_escalate_write",
            policy_variant="escalate_all",
            risk_score=1.0,
            impact=metadata.impact if metadata is not None else 3,
            stage="baseline",
        )
