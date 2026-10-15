from collections.abc import Iterable

from .budget_allocator import BudgetAllocator
from .decision_logger import DecisionLogger
from .llm_judge import LLMJudge
from .policy_engine import PolicyEngine

# 实验中对比的 policy 设置；full 是完整系统，其余为 baseline。
POLICIES = ("full", "allow_all", "escalate_all", "p1_only", "no_budget")

# "无限" reviewer 预算：远大于一个任务可能产生的调用数。
UNLIMITED_ESCALATIONS = 1_000_000


def escalation_budget_for(policy: str, k: int) -> int:
    """escalate_all 和 no_budget 不限预算，其余 policy 使用 k。"""

    _check_policy(policy)
    if policy in {"escalate_all", "no_budget"}:
        return UNLIMITED_ESCALATIONS
    return k


def build_policy_engine(
    policy: str,
    *,
    judge: LLMJudge,
    allocator: BudgetAllocator,
    tool_names: Iterable[str],
    q_table: dict[str, float] | None = None,
    decision_logger: DecisionLogger | None = None,
) -> PolicyEngine:
    """
    - full / no_budget：P1 -> Judge -> P6（no_budget 只是预算不同）
    - allow_all：没有 gate，所有工具直接执行
    - escalate_all：只读调用放行，其余全部交给 reviewer
    - p1_only：只用 P1，P1 无法决定的调用直接交给 reviewer
    """

    _check_policy(policy)
    if policy in {"full", "no_budget"}:
        return PolicyEngine(
            judge=judge,
            allocator=allocator,
            q_table=q_table,
            decision_logger=decision_logger,
        )
    if policy == "allow_all":
        return PolicyEngine(
            allowed_tools=set(tool_names),
            decision_logger=decision_logger,
        )
    if policy == "escalate_all":
        return PolicyEngine(
            escalate_writes=True,
            decision_logger=decision_logger,
        )
    # p1_only：不配置 Judge/allocator，P1 的 REQUIRE_APPROVAL 直接交给 reviewer。
    return PolicyEngine(decision_logger=decision_logger)


def _check_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
