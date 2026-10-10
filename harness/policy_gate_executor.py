from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from .decision_logger import DecisionLogger
from .policy import (
    Decision,
    PolicyHistoryItem,
    PolicyRequest,
    ToolMetadata,
)
from .policy_engine import PolicyEngine
from .simulated_reviewer import ReviewResult, SimulatedReviewer


class PolicyGateToolsExecutor(ToolsExecutor):
    """在 AgentDojo 执行任一工具前插入 PolicyEngine 决策。"""

    def __init__(
        self,
        policy_engine: PolicyEngine,
        tool_metadata: dict[str, ToolMetadata] | None = None,
        task_id: str = "demo-task",
        escalation_budget: int = 0,
        reviewer: SimulatedReviewer | None = None,
        review_logger: DecisionLogger | None = None,
    ) -> None:
        if escalation_budget < 0:
            raise ValueError("escalation_budget must not be negative")

        super().__init__()
        self.policy_engine = policy_engine
        self.tool_metadata = dict(tool_metadata or {})
        self.task_id = task_id
        self.escalation_budget = escalation_budget
        self.reviewer = reviewer
        self.review_logger = review_logger

    @staticmethod
    def _history_from_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[PolicyHistoryItem, ...]:
        """提取已完成历史；调用方会排除当前批次 assistant 消息。"""

        history: list[PolicyHistoryItem] = []
        for message in messages:
            if message["role"] != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                history.append(
                    PolicyHistoryItem(
                        tool_name=tool_call.function,
                        tool_call_id=tool_call.id,
                        arguments=dict(tool_call.args),
                    )
                )
        return tuple(history)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        if extra_args is None:
            extra_args = {}

        self._initialize_escalation_state(extra_args)

        # 没有可检查的助手工具调用时，原样返回
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls = messages[-1].get("tool_calls") or []
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        tool_results = []
        # extra_args 若显式提供 history，则优先使用；否则从会话消息重建。
        history = tuple(
            extra_args.get(
                "history",
                PolicyGateToolsExecutor._history_from_messages(messages[:-1]),
            )
        )

        for tool_call in tool_calls:
            # 先检查策略，再决定是否运行工具
            policy_request = PolicyRequest(
                tool_name=tool_call.function,
                tool_call_id=tool_call.id,
                arguments=tool_call.args,
                task_id=str(extra_args.get("task_id", self.task_id)),
                user_request=str(extra_args.get("user_request", "")),
                history=history,
                # metadata 缺失时会在 P1 中 fail closed 为 REQUIRE_APPROVAL。
                tool_metadata=self.tool_metadata.get(tool_call.function),
                remaining_escalations=int(
                    extra_args.get("remaining_escalations", 0)
                ),
                expected_remaining_risky_calls=int(
                    extra_args.get("expected_remaining_risky_calls", 0)
                ),
            )
            policy_result = self.policy_engine.evaluate(policy_request)

            effective_decision = policy_result.decision
            review_result: ReviewResult | None = None
            if effective_decision == Decision.REQUIRE_APPROVAL:
                # 消费发生在真正交给 reviewer 之前；无论 reviewer 最终 approve
                # 还是 deny，这次人工审核都已经占用一次预算。
                if not self._consume_escalation(extra_args):
                    effective_decision = Decision.DENY
                elif self.reviewer is not None:
                    review_result = self.reviewer.review(
                        policy_request,
                        trace_summary=str(
                            extra_args.get("trace_summary", "")
                        ),
                    )
                    extra_args["reviewer_verdict"] = (
                        review_result.decision.value
                    )
                    extra_args["reviewer_success"] = review_result.success
                    extra_args["reviewer_reason"] = review_result.reason
                    if self.review_logger is not None:
                        self.review_logger.log(
                            policy_request,
                            policy_result,
                            extra={
                                "event_type": "reviewer_decision",
                                "reviewer_verdict": (
                                    review_result.decision.value
                                ),
                                "reviewer_success": review_result.success,
                                "reviewer_reason": review_result.reason,
                            },
                        )
                    effective_decision = review_result.decision

            if effective_decision == Decision.ALLOW:
                # 只把当前获准的工具调用交给 AgentDojo 执行
                assistant_message = dict(messages[-1])
                assistant_message["tool_calls"] = [tool_call]

                _, runtime, env, processed_messages, extra_args = super().query(
                    query,
                    runtime,
                    env,
                    [*messages[:-1], assistant_message],
                    extra_args,
                )
                tool_results.append(processed_messages[-1])

            else:
                # DENY 和 REQUIRE_APPROVAL 都不执行；必须返回匹配的 tool_call_id。
                if review_result is not None:
                    message = (
                        "Reviewer decision: DENY. Tool was not executed. "
                        f"Reason: {review_result.reason}"
                    )
                elif (
                    policy_result.decision == Decision.REQUIRE_APPROVAL
                    and effective_decision == Decision.DENY
                ):
                    message = (
                        "Policy decision: DENY. Escalation budget exhausted. "
                        f"Original policy reason: {policy_result.reason}"
                    )
                else:
                    message = (
                        f"Policy decision: {effective_decision.value}. "
                        f"Tool was not executed. Reason: {policy_result.reason}"
                    )
                tool_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(message)],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=message,
                    )
                )

        return query, runtime, env, [*messages, *tool_results], extra_args

    def _initialize_escalation_state(
        self,
        extra_args: dict[str, Any],
    ) -> None:
        remaining = extra_args.get(
            "remaining_escalations",
            extra_args.get("escalation_budget", self.escalation_budget),
        )
        remaining = self._non_negative_int(
            remaining,
            field_name="remaining_escalations",
        )
        used = self._non_negative_int(
            extra_args.get("escalations_used", 0),
            field_name="escalations_used",
        )

        extra_args["remaining_escalations"] = remaining
        extra_args["escalations_used"] = used
        extra_args.setdefault("initial_escalations", remaining)

    @staticmethod
    def _consume_escalation(extra_args: dict[str, Any]) -> bool:
        remaining = int(extra_args.get("remaining_escalations", 0))
        if remaining <= 0:
            return False

        extra_args["remaining_escalations"] = remaining - 1
        extra_args["escalations_used"] = (
            int(extra_args.get("escalations_used", 0)) + 1
        )
        return True

    @staticmethod
    def _non_negative_int(value: Any, field_name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must be an integer") from error
        if parsed < 0:
            raise ValueError(f"{field_name} must not be negative")
        return parsed
