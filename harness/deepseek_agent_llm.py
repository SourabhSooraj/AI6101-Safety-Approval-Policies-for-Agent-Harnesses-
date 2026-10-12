from collections.abc import Sequence

from agentdojo.agent_pipeline import OpenAILLM
from agentdojo.agent_pipeline.llms.openai_llm import (
    _content_blocks_to_openai_content_blocks,
    _function_to_openai,
    _openai_to_assistant_message,
    _tool_call_to_openai,
    chat_completion_request,
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
        if message["tool_calls"]:
            return {
                "role": "assistant",
                "content": _content_blocks_to_openai_content_blocks(message),
                "tool_calls": [
                    _tool_call_to_openai(tool_call)
                    for tool_call in message["tool_calls"]
                ],
            }
        return {
            "role": "assistant",
            "content": _content_blocks_to_openai_content_blocks(message),
        }
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


class DeepSeekAgentLLM(OpenAILLM):
    """AgentDojo OpenAILLM 的 DeepSeek 兼容版本。"""

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
        completion = chat_completion_request(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            self.reasoning_effort,
            self.temperature,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args
