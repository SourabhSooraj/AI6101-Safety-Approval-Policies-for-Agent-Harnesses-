from ..budget_allocator import BudgetAllocator
from ..llm_judge import LLMJudge
from .base import PolicyEvaluator
from .baselines import AllowAllPolicy, EscalateAllPolicy
from .hybrid import HybridPolicy
from .p1_only import P1OnlyPolicy
from .p4_only import P4OnlyPolicy


# 命令行可选的 policy；no_budget 是预算无限的 hybrid。
POLICY_VARIANTS = (
    "hybrid",
    "p1_only",
    "p4_only",
    "allow_all",
    "escalate_all",
    "no_budget",
)

POLICY_ALIASES = {
    "p1": "p1_only",
    "p1_only": "p1_only",
    "p4": "p4_only",
    "p4_only": "p4_only",
    "hybrid": "hybrid",
    "p6": "hybrid",
    "full": "hybrid",
    "allow_all": "allow_all",
    "escalate_all": "escalate_all",
    "no_budget": "no_budget",
}

# "无限" reviewer 预算：远大于一个任务可能产生的调用数。
UNLIMITED_ESCALATIONS = 1_000_000


def normalize_variant(variant: str) -> str:
    normalized = POLICY_ALIASES.get(variant.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported policy variant: {variant}")
    return normalized


def escalation_budget_for(variant: str, k: int) -> int:
    """
    所有 policy（包括两个 baseline）都使用同一个 k，以便画 TCR@k 曲线；
    只有 no_budget 消融不限预算。
    """

    if normalize_variant(variant) == "no_budget":
        return UNLIMITED_ESCALATIONS
    return k


def create_policy(
    variant: str,
    judge: LLMJudge | None = None,
    allocator: BudgetAllocator | None = None,
    q_table: dict[str, float] | None = None,
    t_low: float = 0.3,
    t_high: float = 0.7,
) -> PolicyEvaluator:
    normalized = normalize_variant(variant)

    if normalized == "p1_only":
        return P1OnlyPolicy()
    if normalized == "allow_all":
        return AllowAllPolicy()
    if normalized == "escalate_all":
        return EscalateAllPolicy()

    if judge is None:
        raise ValueError(f"{normalized} requires an LLM Judge")

    if normalized == "p4_only":
        return P4OnlyPolicy(judge=judge, t_low=t_low, t_high=t_high)

    if allocator is None:
        raise ValueError(f"{normalized} policy requires a budget allocator")
    return HybridPolicy(
        judge=judge,
        allocator=allocator,
        q_table=q_table,
        t_low=t_low,
        t_high=t_high,
    )
