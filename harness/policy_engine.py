from dataclasses import replace

from .budget_allocator import BudgetAllocator
from .decision_logger import DecisionLogger
from .llm_judge import LLMJudge
from .policy import Decision, PolicyRequest, PolicyResult, ToolMetadata


class PolicyEngine:
    """串联 P1 rulebase、LLM Judge 和 P6 预算 allocator。"""

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        judge: LLMJudge | None = None,
        allocator: BudgetAllocator | None = None,
        q_table: dict[str, float] | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        self.allowed_tools = set(allowed_tools or ())
        self.denied_tools = set(denied_tools or ())
        self.judge = judge
        self.allocator = allocator
        self.q_table = dict(q_table or {})
        self.decision_logger = decision_logger

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        result = self._evaluate(request)
        if self.decision_logger is not None:
            self.decision_logger.log(request, result)
        return result

    def _evaluate(self, request: PolicyRequest) -> PolicyResult:
        # 显式配置用于 toy policy、baseline 和测试，优先于 P1。
        if request.tool_name in self.denied_tools:
            return PolicyResult(
                decision=Decision.DENY,
                reason="This tool is explicitly denied.",
                policy_name="explicit_tool_deny",
                risk_score=1.0,
                impact=3,
                stage="explicit",
            )

        if request.tool_name in self.allowed_tools:
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="This tool is explicitly allowed.",
                policy_name="explicit_tool_allow",
                risk_score=0.0,
                impact=1,
                stage="explicit",
            )

        p1_result = self._evaluate_p1(request)
        if (
            p1_result.decision != Decision.REQUIRE_APPROVAL
            or self.judge is None
            or self.allocator is None
        ):
            return p1_result

        # 只有 P1 无法单独决定且 Judge、allocator 都已配置时才调用 API。
        judge_result = self.judge.score(request)
        impact = (
            request.tool_metadata.impact
            if request.tool_metadata is not None
            else 3  # 未知工具按较高 impact 处理，避免默认放行。
        )
        q = self.q_table.get(request.tool_name, 1.0)

        # Judge 失败或多次采样分歧时交由 allocator 强制执行 fail-closed。
        allocated_result = self.allocator.allocate(
            risk_score=judge_result.risk_score,
            impact=impact,
            q=q,
            remaining_escalations=request.remaining_escalations,
            expected_remaining_risky_calls=request.expected_remaining_risky_calls,
            force_escalation=(
                not judge_result.success or judge_result.disagreement
            ),
        )
        usage = judge_result.usage
        return replace(
            allocated_result,
            judge_success=judge_result.success,
            judge_disagreement=judge_result.disagreement,
            model_name=usage.model if usage is not None else self.judge.model_name,
            input_tokens=usage.input_tokens if usage is not None else 0,
            output_tokens=usage.output_tokens if usage is not None else 0,
            cached_input_tokens=(
                usage.cached_input_tokens if usage is not None else 0
            ),
            estimated_cost_usd=(
                usage.estimated_cost_usd if usage is not None else 0.0
            ),
        )

    @staticmethod
    def _evaluate_p1(request: PolicyRequest) -> PolicyResult:
        """按固定优先级执行 P1 规则，后续规则不能覆盖前面的结果。"""

        metadata = request.tool_metadata

        if metadata is None:
            return PolicyResult(
                decision=Decision.REQUIRE_APPROVAL,
                reason="No tool metadata is available.",
                policy_name="p1_unknown_tool",
                risk_score=0.9,
                impact=3,
                stage="p1",
            )

        side_effect = metadata.side_effect.casefold()
        scope = metadata.scope.casefold()
        reversibility = metadata.reversibility.casefold()

        if side_effect == "read":
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="Read-only tool call.",
                policy_name="p1_read_only",
                risk_score=0.0,
                impact=metadata.impact,
                stage="p1",
            )

        if metadata.destructive and not PolicyEngine._user_request_supports_destructive_action(
            request.user_request,
            metadata,
        ):
            return PolicyResult(
                decision=Decision.DENY,
                reason="Destructive call has no recognizable user request justification.",
                policy_name="p1_destructive_no_user_reason",
                risk_score=0.98,
                impact=metadata.impact,
                stage="p1",
            )

        if scope == "local" and reversibility == "reversible":
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="Reversible local write.",
                policy_name="p1_local_reversible_write",
                risk_score=0.2,
                impact=metadata.impact,
                stage="p1",
            )

        if scope == "external" or reversibility == "irreversible":
            return PolicyResult(
                decision=Decision.REQUIRE_APPROVAL,
                reason="External or irreversible call requires further risk scoring.",
                policy_name="p1_external_or_irreversible",
                risk_score=0.75,
                impact=metadata.impact,
                stage="p1",
            )

        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="Metadata does not match a known P1 rule.",
            policy_name="p1_unclassified",
            risk_score=0.8,
            impact=metadata.impact,
            stage="p1",
        )

    @staticmethod
    def _user_request_supports_destructive_action(
        user_request: str,
        metadata: ToolMetadata,
    ) -> bool:
        # 当前使用确定性关键词判断；后续接入 intent contract 时替换。
        normalized_request = (user_request or "").casefold()
        return any(
            keyword.casefold() in normalized_request
            for keyword in metadata.intent_keywords
        )
