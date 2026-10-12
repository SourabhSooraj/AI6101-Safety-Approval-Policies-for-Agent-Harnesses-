import os
from typing import Any

from openai import OpenAI

from .cost import TokenUsage, estimate_deepseek_usage


class DeepSeekJudgeClient:
    """通过 OpenAI-compatible API 调用 DeepSeek 的 JudgeClient 适配器。"""

    def __init__(
        self,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        cost_period: str = "auto",
        openai_client: Any | None = None,
    ) -> None:
        # 只在创建真实客户端时读取 Key，避免无 Key 的单元测试和 P1-only 运行失败。
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_api_key and openai_client is None:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Set it in the environment "
                "or pass api_key explicitly."
            )

        self.model = model
        self.temperature = temperature
        self.cost_period = cost_period
        self.last_usage: TokenUsage | None = None
        self.client = openai_client or OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def complete(self, prompt: str) -> str:
        # DeepSeek 支持 JSON Output，配合 LLMJudge 的严格解析器使用。
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned an empty response.")

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_input_tokens = int(
            getattr(prompt_details, "cached_tokens", 0) or 0
        )
        self.last_usage = estimate_deepseek_usage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            period=self.cost_period,
        )
        return content
