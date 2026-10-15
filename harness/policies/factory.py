from ..budget_allocator import BudgetAllocator
from ..llm_judge import LLMJudge
from .base import PolicyEvaluator
from .hybrid import HybridPolicy
from .p1_only import P1OnlyPolicy
from .p4_only import P4OnlyPolicy


POLICY_ALIASES = {
    "p1": "p1_only",
    "p1_only": "p1_only",
    "p4": "p4_only",
    "p4_only": "p4_only",
    "hybrid": "hybrid",
    "p6": "hybrid",
}


def create_policy(
    variant: str,
    judge: LLMJudge | None = None,
    allocator: BudgetAllocator | None = None,
    q_table: dict[str, float] | None = None,
    t_low: float = 0.3,
    t_high: float = 0.7,
) -> PolicyEvaluator:
    normalized = POLICY_ALIASES.get(variant.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported policy variant: {variant}")

    if normalized == "p1_only":
        return P1OnlyPolicy()

    if judge is None:
        raise ValueError(f"{normalized} requires an LLM Judge")

    if normalized == "p4_only":
        return P4OnlyPolicy(judge=judge, t_low=t_low, t_high=t_high)

    if allocator is None:
        raise ValueError("hybrid policy requires a budget allocator")
    return HybridPolicy(
        judge=judge,
        allocator=allocator,
        q_table=q_table,
        t_low=t_low,
        t_high=t_high,
    )
