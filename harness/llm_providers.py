import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Ollama 不校验 Key，但 OpenAI SDK 要求非空字符串。
LOCAL_PLACEHOLDER_API_KEY = "ollama"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", flags=re.DOTALL)


@dataclass(frozen=True)
class ProviderDefaults:
    """
    一个 provider 的默认模型和连接设置。

    Judge 默认与 agent 使用同一模型；reviewer_model 为 None 时也与 agent 相同。
    """

    base_url: str
    agent_model: str
    reviewer_model: str | None
    api_key_env: str
    request_timeout: float


PROVIDERS: dict[str, ProviderDefaults] = {
    "deepseek": ProviderDefaults(
        base_url=DEEPSEEK_BASE_URL,
        agent_model="deepseek-flash",
        reviewer_model=None,
        api_key_env="DEEPSEEK_API_KEY",
        request_timeout=120.0,
    ),
    # 免费的本地开发配置；reviewer 使用不同模型家族，减少与 Judge 的相关性。
    "ollama": ProviderDefaults(
        base_url=OLLAMA_BASE_URL,
        agent_model="qwen3:8b",
        reviewer_model="llama3.1:8b",
        api_key_env="OLLAMA_API_KEY",
        request_timeout=300.0,
    ),
}


def is_local_base_url(base_url: str) -> bool:
    """本地 OpenAI-compatible 服务（如 Ollama）不需要真实 API Key。"""

    host = urlparse(base_url).hostname or ""
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def resolve_api_key(
    base_url: str,
    api_key: str | None = None,
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> str:
    """优先使用显式 Key，其次读环境变量；本地服务缺 Key 时使用占位值。"""

    resolved = api_key or os.environ.get(api_key_env)
    if resolved:
        return resolved
    if is_local_base_url(base_url):
        return LOCAL_PLACEHOLDER_API_KEY
    raise ValueError(
        f"{api_key_env} is not set. Set it in the environment "
        "or pass api_key explicitly."
    )


def strip_think(text: str | None) -> str | None:
    """去掉 qwen3 等推理模型输出的 <think>...</think> 段落。"""

    if text is None:
        return None
    return _THINK_BLOCK.sub("", text).strip()


def thinking_switch(model: str) -> str:
    """qwen3 的软开关：在提示末尾加 /no_think 关闭思考模式。"""

    return " /no_think" if model.lower().startswith("qwen3") else ""


def thinking_request_args(
    base_url: str,
    thinking: bool,
    reasoning_effort: str | None = None,
) -> dict:
    """
    按 provider 生成开关思考模式的请求参数。

    Ollama 会静默忽略 DeepSeek 的 extra_body.thinking，较新版本也不再理会
    /no_think；只有 reasoning_effort="none" 能真正关闭 qwen3 的思考。
    """

    if is_local_base_url(base_url):
        if not thinking:
            return {"reasoning_effort": "none"}
        return {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    args: dict = {
        "extra_body": {
            "thinking": {"type": "enabled" if thinking else "disabled"}
        }
    }
    if thinking and reasoning_effort:
        args["reasoning_effort"] = reasoning_effort
    return args
