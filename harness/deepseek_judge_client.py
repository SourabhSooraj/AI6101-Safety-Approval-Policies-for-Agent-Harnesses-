from typing import Any

from openai import OpenAI

from .cost import TokenUsage, usage_from_openai_response
from .llm_providers import (
    DEEPSEEK_BASE_URL,
    resolve_api_key,
    strip_think,
    thinking_switch,
)


class DeepSeekJudgeClient:
    """
    通过 OpenAI-compatible API 调用 Judge / reviewer 模型的 JudgeClient 适配器。

    默认连接 DeepSeek；base_url 指向本地 Ollama 时无需 API Key。
    """

    def __init__(
        self,
        model: str = "deepseek-flash",
        base_url: str = DEEPSEEK_BASE_URL,
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.0,
        timeout: float = 60.0,
        cost_period: str = "auto",
        openai_client: Any | None = None,
    ) -> None:
        # 只在创建真实客户端时读取 Key，避免无 Key 的单元测试和 P1-only 运行失败。
        resolved_api_key = None
        if openai_client is None:
            resolved_api_key = resolve_api_key(base_url, api_key, api_key_env)

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.cost_period = cost_period
        self.last_usage: TokenUsage | None = None
        self.client = openai_client or OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def complete(self, prompt: str) -> str:
        # JSON Output 配合 LLMJudge 的严格解析器使用；Ollama 也支持 json_object。
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt + thinking_switch(self.model),
                }
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        self.last_usage = usage_from_openai_response(
            response,
            model=self.model,
            period=self.cost_period,
        )
        # <think> 段落里可能有花括号，会让 Judge 解析器抓错 JSON。
        content = strip_think(response.choices[0].message.content)
        if not content:
            raise RuntimeError(f"{self.model} returned an empty response.")
        return content
