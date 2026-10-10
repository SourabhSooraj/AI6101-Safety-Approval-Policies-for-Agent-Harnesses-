from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from .policy import Decision, PolicyRequest
from .policy_engine import PolicyEngine


class PolicyGateToolsExecutor(ToolsExecutor):
    def __init__(
        self,
        policy_engine: PolicyEngine,
        task_id: str = "demo-task",
    ) -> None:
        super().__init__()
        self.policy_engine = policy_engine
        self.task_id = task_id

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

        # 没有可检查的助手工具调用时，原样返回
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls = messages[-1].get("tool_calls") or []
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        tool_results = []

        for tool_call in tool_calls:
            # 先检查策略，再决定是否运行工具
            policy_result = self.policy_engine.evaluate(
                PolicyRequest(
                    tool_name=tool_call.function,
                    tool_call_id=tool_call.id,
                    arguments=tool_call.args,
                    task_id=str(extra_args.get("task_id", self.task_id)),
                )
            )

            if policy_result.decision == Decision.ALLOW:
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
                # DENY 和 REQUIRE_APPROVAL 都不会执行工具
                message = (
                    f"Policy decision: {policy_result.decision.value}. "
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