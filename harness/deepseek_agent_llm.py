from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline import OpenAILLM
from agentdojo.agent_pipeline.llms.openai_llm import (
    _content_blocks_to_openai_content_blocks,
    _function_to_openai,
    _openai_to_assistant_message,
    _tool_call_to_openai,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .cost import TokenUsage, usage_from_openai_response


def _message_to_deepseek(message: ChatMessage, model_name: str):
    """DeepSeek Chat Completions 使用 system，而不是 developer role。"""

    role = message["role"]
    if role == "system":
        return {
            "role": "system",
            "content": _content_blocks_to_openai_content_blocks(message),
        }
    if role == "user":
        return {
            "role": "user",
            "content": _content_blocks_to_openai_content_blocks(message),
        }
    if role == "assistant":
        output = {
            "role": "assistant",
            "content": _content_blocks_to_openai_content_blocks(message),
        }
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            output["reasoning_content"] = reasoning_content
        if message["tool_calls"]:
            output["tool_calls"] = [
                _tool_call_to_openai(tool_call)
                for tool_call in message["tool_calls"]
            ]
        return output
    if role == "tool":
        tool_call = message["tool_call"]
        return {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "name": tool_call.function,
            "content": message["error"]
            or _content_blocks_to_openai_content_blocks(message),
        }
    raise ValueError(f"Invalid message type: {message}")


def _openai_to_deepseek_assistant(message):
    output = _openai_to_assistant_message(message)
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        output["reasoning_content"] = reasoning_content
    return output


# 与 AgentDojo chat_completion_request 相同的重试策略：临时错误最多尝试 3 次，
# 请求本身有问题（400/422）时不重试。
@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(
        (openai.BadRequestError, openai.UnprocessableEntityError)
    ),
)
def _create_completion(client, **create_args):
    return client.chat.completions.create(**create_args)


class DeepSeekAgentLLM(OpenAILLM):
    """AgentDojo OpenAILLM 的 DeepSeek 兼容版本。"""

    def __init__(
        self,
        client,
        model: str,
        reasoning_effort=None,
        temperature: float | None = 0.0,
        thinking: bool = False,
        cost_period: str = "auto",
        send_thinking_param: bool = True,
    ) -> None:
        # thinking 字段是 DeepSeek 专有参数；Llama/Qwen 等其他 OpenAI-compatible
        # 服务可能拒绝它，因此非 DeepSeek API 时不发送。
        if thinking and not send_thinking_param:
            raise ValueError("Thinking mode is only available on the DeepSeek API.")
        super().__init__(
            client,
            model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        self.thinking = thinking
        self.send_thinking_param = send_thinking_param
        self.cost_period = cost_period
        # 累计 agent 自身的 token 和成本；Judge/reviewer 成本记录在决策日志中。
        self.total_usage = TokenUsage.zero(self.model)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}

        openai_messages = [
            _message_to_deepseek(message, self.model)
            for message in messages
        ]
        openai_tools = [
            _function_to_openai(tool)
            for tool in runtime.functions.values()
        ]
        create_args = {
            "model": self.model,
            "messages": openai_messages,
        }
        if self.send_thinking_param:
            create_args["extra_body"] = {
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled"
                }
            }
        if openai_tools:
            create_args["tools"] = openai_tools
            create_args["tool_choice"] = "auto"
        if not self.thinking:
            create_args["temperature"] = self.temperature
        elif self.reasoning_effort:
            create_args["reasoning_effort"] = self.reasoning_effort

        completion = _create_completion(self.client, **create_args)
        self.total_usage = self.total_usage + usage_from_openai_response(
            completion,
            model=self.model,
            period=self.cost_period,
        )
        output = _openai_to_deepseek_assistant(
            completion.choices[0].message
        )
        return query, runtime, env, [*messages, output], extra_args
