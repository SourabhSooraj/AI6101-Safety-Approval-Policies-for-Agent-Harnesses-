from typing import Any

from openai import OpenAI

from .cost import TokenUsage, usage_from_openai_response
from .llm_providers import (
    DEEPSEEK_BASE_URL,
    is_dashscope_api,
    is_local_base_url,
    local_thinking_off_args,
    resolve_api_key,
    strip_think,
    thinking_switch,
)


class DeepSeekJudgeClient:
    """
    通过 OpenAI-compatible API 调用 Judge / reviewer 模型的 JudgeClient 适配器。

    默认连接 DeepSeek；base_url 指向本地 Ollama 时无需 API Key；
    指向阿里云百炼时用 enable_thinking 关闭思考模式。
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
        local_server: str = "ollama",
    ) -> None:
        # 只在创建真实客户端时读取 Key，避免无 Key 的单元测试和 P1-only 运行失败。
        resolved_api_key = None
        if openai_client is None:
            resolved_api_key = resolve_api_key(base_url, api_key, api_key_env)

        self.model = model
        self.base_url = base_url
        self.local_server = local_server
        self.temperature = temperature
        self.cost_period = cost_period
        self.last_usage: TokenUsage | None = None
        self.client = openai_client or OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _thinking_off_args(self) -> tuple[str, dict]:
        """返回 (提示末尾的软开关, 额外请求参数)，用于关闭思考模式。"""

        if is_dashscope_api(self.base_url):
            # 百炼上开源 Qwen3、glm-5.1 等默认开启思考，部分模型思考时只支持流式调用；
            # 显式关闭后可以非流式调用，json_object 输出也更稳定。
            return "", {"extra_body": {"enable_thinking": False}}
        if is_local_base_url(self.base_url):
            # Ollama 上 qwen3 一边思考一边受 JSON 约束时常返回空的 {}。
            return thinking_switch(self.model), local_thinking_off_args(
                self.local_server
            )
        return thinking_switch(self.model), {}

    def complete(self, prompt: str) -> str:
        # JSON Output 配合 LLMJudge 的严格解析器使用；Ollama 和百炼也支持 json_object。
        switch, thinking_args = self._thinking_off_args()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt + switch}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            **thinking_args,
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
