from collections.abc import Sequence

from agentdojo.agent_pipeline import OpenAILLM
from agentdojo.agent_pipeline.llms.openai_llm import (
    _content_blocks_to_openai_content_blocks,
    _function_to_openai,
    _openai_to_assistant_message,
    _tool_call_to_openai,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


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


class DeepSeekAgentLLM(OpenAILLM):
    """AgentDojo OpenAILLM 的 DeepSeek 兼容版本。"""

    def __init__(
        self,
        client,
        model: str,
        reasoning_effort=None,
        temperature: float | None = 0.0,
        thinking: bool = False,
    ) -> None:
        super().__init__(
            client,
            model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        self.thinking = thinking

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
            "extra_body": {
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled"
                }
            },
        }
        if openai_tools:
            create_args["tools"] = openai_tools
            create_args["tool_choice"] = "auto"
        if not self.thinking:
            create_args["temperature"] = self.temperature
        elif self.reasoning_effort:
            create_args["reasoning_effort"] = self.reasoning_effort

        completion = self.client.chat.completions.create(**create_args)
        output = _openai_to_deepseek_assistant(
            completion.choices[0].message
        )
        return query, runtime, env, [*messages, output], extra_args
