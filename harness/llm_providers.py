import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# 阿里云百炼（Model Studio / DashScope）国际站（新加坡）的 OpenAI-compatible 地址。
# API Key 按地域区分：新加坡的 Key 只能用于这个地址。
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Ollama 不校验 Key，但 OpenAI SDK 要求非空字符串。
LOCAL_PLACEHOLDER_API_KEY = "ollama"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# 旧域名 dashscope(-intl).aliyuncs.com，以及新的 {WorkspaceId}.<region>.maas.aliyuncs.com。
_DASHSCOPE_HOSTS = (
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "maas.aliyuncs.com",
)
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
    # 本地服务的种类决定如何关闭思考模式（见 thinking_request_args）。
    local_server: str = "ollama"


VLLM_BASE_URL = "http://localhost:8000/v1"
LOCAL_SERVERS = ("ollama", "vllm")

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
    # 不能装 Ollama 的机器（如学校 GPU 服务器）用 pip 安装的 vLLM；模型名是
    # Hugging Face id，reviewer 默认与 agent 相同（脚本会警告）。
    "vllm": ProviderDefaults(
        base_url=VLLM_BASE_URL,
        agent_model="Qwen/Qwen3-8B",
        reviewer_model=None,
        api_key_env="VLLM_API_KEY",
        request_timeout=300.0,
        local_server="vllm",
    ),
}


def is_local_base_url(base_url: str) -> bool:
    """本地 OpenAI-compatible 服务（如 Ollama）不需要真实 API Key。"""

    host = urlparse(base_url).hostname or ""
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def is_deepseek_api(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host == "deepseek.com" or host.endswith(".deepseek.com")


def is_dashscope_api(base_url: str) -> bool:
    """阿里云百炼的 OpenAI-compatible 接口，用顶层 enable_thinking 开关思考模式。"""

    host = (urlparse(base_url).hostname or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in _DASHSCOPE_HOSTS
    )


def supports_thinking_switch(base_url: str) -> bool:
    """
    只有 DeepSeek API 和本地服务知道如何开关思考模式；其他托管服务
    可能拒绝 DeepSeek 的 thinking 字段，因此不发送。
    """

    return is_deepseek_api(base_url) or is_local_base_url(base_url)


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


def local_thinking_off_args(local_server: str) -> dict:
    """本地服务上关闭思考模式的请求参数。"""

    if local_server == "vllm":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {"reasoning_effort": "none"}


def thinking_request_args(
    base_url: str,
    thinking: bool,
    reasoning_effort: str | None = None,
    local_server: str = "ollama",
) -> dict:
    """
    按 provider 生成开关思考模式的请求参数。

    Ollama 会静默忽略 DeepSeek 的 extra_body.thinking，较新版本也不再理会
    /no_think；只有 reasoning_effort="none" 能真正关闭 qwen3 的思考。
    vLLM 不接受 reasoning_effort="none"，改用 Qwen3 聊天模板的 enable_thinking。
    """

    if is_local_base_url(base_url) and local_server == "vllm":
        return {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}
        }
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
