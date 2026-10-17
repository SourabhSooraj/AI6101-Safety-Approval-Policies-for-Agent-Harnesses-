from dataclasses import replace
from typing import Protocol

from ..llm_judge import JudgeResult
from ..policy import PolicyRequest, PolicyResult


class PolicyEvaluator(Protocol):
    """所有可比较 Policy 变体必须实现的接口。"""

    name: str

    def evaluate(self, request: PolicyRequest) -> PolicyResult:
        ...


def attach_judge_metadata(
    result: PolicyResult,
    judge_result: JudgeResult,
    fallback_model: str,
) -> PolicyResult:
    """把 Judge 状态、token 和成本附加到 PolicyResult。"""

    usage = judge_result.usage
    return replace(
        result,
        judge_success=judge_result.success,
        judge_disagreement=judge_result.disagreement,
        model_name=usage.model if usage is not None else fallback_model,
        input_tokens=usage.input_tokens if usage is not None else 0,
        output_tokens=usage.output_tokens if usage is not None else 0,
        cached_input_tokens=(
            usage.cached_input_tokens if usage is not None else 0
        ),
        estimated_cost_usd=(
            usage.estimated_cost_usd if usage is not None else 0.0
        ),
    )
