from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


class PolicyContextInitializer(BasePipelineElement):
    """把任务级 Policy 配置写入 AgentDojo 的 extra_args。"""

    def __init__(
        self,
        task_id: str,
        escalation_budget: int,
        expected_remaining_risky_calls: int = 0,
    ) -> None:
        if escalation_budget < 0:
            raise ValueError("escalation_budget must not be negative")
        if expected_remaining_risky_calls < 0:
            raise ValueError(
                "expected_remaining_risky_calls must not be negative"
            )

        self.task_id = task_id
        self.escalation_budget = escalation_budget
        self.expected_remaining_risky_calls = expected_remaining_risky_calls

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}

        extra_args.setdefault("task_id", self.task_id)
        extra_args.setdefault("user_request", query)
        extra_args.setdefault("escalation_budget", self.escalation_budget)
        extra_args.setdefault(
            "remaining_escalations",
            self.escalation_budget,
        )
        extra_args.setdefault(
            "expected_remaining_risky_calls",
            self.expected_remaining_risky_calls,
        )
        return query, runtime, env, messages, extra_args
