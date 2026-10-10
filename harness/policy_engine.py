from .policy import Decision, PolicyRequest, PolicyResult


class PolicyEngine:
    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
    ) -> None:
        self.allowed_tools = set(allowed_tools or ())
        self.denied_tools = set(denied_tools or ())

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        # 明确禁止的工具优先级最高
        if request.tool_name in self.denied_tools:
            return PolicyResult(
                decision=Decision.DENY,
                reason="This tool is explicitly denied.",
                policy_name="tool_deny",
            )

        # 只允许明确列入允许列表的工具
        if request.tool_name in self.allowed_tools:
            return PolicyResult(
                decision=Decision.ALLOW,
                reason="This tool is explicitly allowed.",
                policy_name="tool_allow",
            )

        # 没有匹配规则时，先要求人工审批
        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="No policy rule matches this tool.",
            policy_name="default_require_approval",
        )