from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Gate 对外统一返回的三种决定。"""

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


@dataclass(frozen=True)
class ToolMetadata:
    """工具静态属性，不包含任务答案或攻击标签。"""

    name: str
    side_effect: str
    scope: str
    reversibility: str
    sensitive_args: tuple[str, ...]
    impact: int
    destructive: bool = False
    intent_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyHistoryItem:
    """Policy 可查看的历史调用，只保留工具名和参数，不包含工具输出。"""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PolicyRequest:
    """Harness 交给 PolicyEngine 的完整决策上下文。"""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    task_id: str
    user_request: str = ""
    history: Sequence[PolicyHistoryItem] = ()
    tool_metadata: ToolMetadata | None = None
    remaining_escalations: int = 0
    expected_remaining_risky_calls: int = 0


@dataclass(frozen=True)
class PolicyResult:
    """Policy 返回结果，同时携带 allocator 计算所需的中间量。"""

    decision: Decision
    reason: str
    policy_name: str
    risk_score: float = 0.0
    impact: int = 1
    q: float = 1.0
    value: float = 0.0
    tau: float = 0.0
    stage: str = "p1"
    judge_success: bool | None = None
    judge_disagreement: bool | None = None
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
