from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TokenUsage:
    """一次或多次 LLM 调用的 token 和估算成本汇总。"""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @classmethod
    def zero(cls, model: str) -> "TokenUsage":
        return cls(model=model)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        model = self.model if self.model == other.model else "mixed"
        return TokenUsage(
            model=model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=(
                self.cached_input_tokens + other.cached_input_tokens
            ),
            estimated_cost_usd=(
                self.estimated_cost_usd + other.estimated_cost_usd
            ),
        )


@dataclass(frozen=True)
class ModelRates:
    """每百万 token 的美元价格，来源于 DeepSeek 官方价格页。"""

    cache_hit_input: float
    cache_miss_input: float
    output: float


# 官方价格可能调整，更新成本时需同步核对：
# https://api-docs.deepseek.com/quick_start/pricing/
DEEPSEEK_RATES: dict[str, dict[str, ModelRates]] = {
    "deepseek-flash": {
        "peak": ModelRates(0.006, 0.3, 1.2),
        "off_peak": ModelRates(0.003, 0.15, 0.6),
    },
    "deepseek-v4-pro": {
        "peak": ModelRates(0.044, 1.32, 3.96),
        "off_peak": ModelRates(0.022, 0.66, 1.98),
    },
}


def is_deepseek_peak(now: datetime | None = None) -> bool:
    """
    DeepSeek 高峰时段为 UTC 周一至周五 01:00-04:00 和 06:00-10:00。
    """

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if current.weekday() >= 5:
        return False
    return 1 <= current.hour < 4 or 6 <= current.hour < 10


def estimate_deepseek_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    period: str = "auto",
    now: datetime | None = None,
) -> TokenUsage:
    """根据 API usage 估算 DeepSeek 调用成本。"""

    pricing_period = period
    if pricing_period == "auto":
        pricing_period = "peak" if is_deepseek_peak(now) else "off_peak"
    if pricing_period not in {"peak", "off_peak"}:
        raise ValueError("period must be auto, peak, or off_peak")

    rates_by_period = DEEPSEEK_RATES.get(model)
    if rates_by_period is None:
        return TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    rates = rates_by_period[pricing_period]
    cache_hit_tokens = min(max(0, cached_input_tokens), max(0, input_tokens))
    cache_miss_tokens = max(0, input_tokens - cache_hit_tokens)
    estimated_cost = (
        cache_hit_tokens / 1_000_000 * rates.cache_hit_input
        + cache_miss_tokens / 1_000_000 * rates.cache_miss_input
        + output_tokens / 1_000_000 * rates.output
    )

    return TokenUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        estimated_cost_usd=estimated_cost,
    )
