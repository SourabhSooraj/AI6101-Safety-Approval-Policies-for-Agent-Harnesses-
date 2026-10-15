from ..policy import Decision, PolicyRequest, PolicyResult, ToolMetadata


def evaluate_p1(request: PolicyRequest) -> PolicyResult:
    """执行 P1 rulebase；该函数供 P1-only 和 Hybrid 共用。"""

    metadata = request.tool_metadata

    if metadata is None:
        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="No tool metadata is available.",
            policy_name="p1_unknown_tool",
            policy_variant="p1_only",
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
            policy_variant="p1_only",
            risk_score=0.0,
            impact=metadata.impact,
            stage="p1",
        )

    if metadata.destructive and not user_request_supports_destructive_action(
        request.user_request,
        metadata,
    ):
        return PolicyResult(
            decision=Decision.DENY,
            reason="Destructive call has no recognizable user request justification.",
            policy_name="p1_destructive_no_user_reason",
            policy_variant="p1_only",
            risk_score=0.98,
            impact=metadata.impact,
            stage="p1",
        )

    if scope == "local" and reversibility == "reversible":
        return PolicyResult(
            decision=Decision.ALLOW,
            reason="Reversible local write.",
            policy_name="p1_local_reversible_write",
            policy_variant="p1_only",
            risk_score=0.2,
            impact=metadata.impact,
            stage="p1",
        )

    if scope == "external" or reversibility == "irreversible":
        return PolicyResult(
            decision=Decision.REQUIRE_APPROVAL,
            reason="External or irreversible call requires further risk scoring.",
            policy_name="p1_external_or_irreversible",
            policy_variant="p1_only",
            risk_score=0.75,
            impact=metadata.impact,
            stage="p1",
        )

    return PolicyResult(
        decision=Decision.REQUIRE_APPROVAL,
        reason="Metadata does not match a known P1 rule.",
        policy_name="p1_unclassified",
        policy_variant="p1_only",
        risk_score=0.8,
        impact=metadata.impact,
        stage="p1",
    )


def user_request_supports_destructive_action(
    user_request: str,
    metadata: ToolMetadata,
) -> bool:
    normalized_request = (user_request or "").casefold()
    return any(
        keyword.casefold() in normalized_request
        for keyword in metadata.intent_keywords
    )


class P1OnlyPolicy:
    name = "p1_only"

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        return evaluate_p1(request)
