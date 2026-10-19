from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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
    """每百万 token 的美元价格，来源于各 provider 的官方价格页。"""

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


@dataclass(frozen=True)
class RateTier:
    """单次请求的输入 token 不超过 max_input_tokens 时适用的价格。"""

    max_input_tokens: int
    rates: ModelRates


# 阿里云百炼（Model Studio / DashScope）国际站（新加坡）标准价，按模型名查找。
# 核对日期 2026-10-18（价格页标注更新于 2026-10-15）；Key 和价格都按地域区分：
# https://www.alibabacloud.com/help/en/model-studio/model-pricing
# 档位按单次请求的输入 token 总数划分，整个请求都按所在档位计费。
# 隐式缓存命中按输入单价的 20% 计费：
# https://www.alibabacloud.com/help/en/model-studio/context-cache
QWEN_RATES: dict[str, tuple[RateTier, ...]] = {
    # Judge：Qwen 家族里最便宜的 flash 档。
    "qwen3.7-flash": (
        RateTier(32_000, ModelRates(0.006, 0.03, 0.13)),
        RateTier(256_000, ModelRates(0.02, 0.1, 0.4)),
        RateTier(1_000_000, ModelRates(0.04, 0.2, 0.8)),
    ),
    # reviewer：智谱 GLM，与 Judge 不同家族，减少两者犯同样的错。
    "glm-5.1": (
        RateTier(200_000, ModelRates(0.28, 1.4, 4.4)),
    ),
}


def qwen_rates(model: str, input_tokens: int) -> ModelRates | None:
    """按单次请求的输入长度选档；超过最高档时按最高档估算。"""

    tiers = QWEN_RATES.get(model)
    if tiers is None:
        return None
    for tier in tiers:
        if input_tokens <= tier.max_input_tokens:
            return tier.rates
    return tiers[-1].rates


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


def usage_from_openai_response(
    response: Any,
    model: str,
    period: str = "auto",
) -> TokenUsage:
    """从 OpenAI-compatible 响应的 usage 字段提取 token 并估算成本。"""

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_input_tokens = int(
        getattr(prompt_details, "cached_tokens", 0) or 0
    )
    return estimate_usage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        period=period,
    )


def estimate_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    period: str = "auto",
    now: datetime | None = None,
) -> TokenUsage:
    """
    根据 API usage 估算调用成本。

    DeepSeek 按 period 区分高峰/低谷价；百炼模型按输入长度选档，忽略 period。
    价格表里没有的模型（如本地 Ollama）成本记为 0。
    """

    pricing_period = period
    if pricing_period == "auto":
        pricing_period = "peak" if is_deepseek_peak(now) else "off_peak"
    if pricing_period not in {"peak", "off_peak"}:
        raise ValueError("period must be auto, peak, or off_peak")

    rates_by_period = DEEPSEEK_RATES.get(model)
    if rates_by_period is not None:
        rates = rates_by_period[pricing_period]
    else:
        rates = qwen_rates(model, input_tokens)
    if rates is None:
        return TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

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


# 旧名字，保留给已有调用方。
estimate_deepseek_usage = estimate_usage
