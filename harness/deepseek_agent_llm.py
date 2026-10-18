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

from .cost import TokenUsage, usage_from_openai_response
from .llm_providers import strip_think, thinking_request_args, thinking_switch


def _message_to_deepseek(
    message: ChatMessage,
    model_name: str,
    disable_thinking: bool = True,
):
    """DeepSeek Chat Completions 使用 system，而不是 developer role。"""

    role = message["role"]
    if role == "system":
        content = _content_blocks_to_openai_content_blocks(message)
        switch = thinking_switch(model_name) if disable_thinking else ""
        if switch and content:
            # qwen3（Ollama）：在 system prompt 末尾关闭思考模式，节省本地推理时间。
            content = [
                *content[:-1],
                {**content[-1], "text": content[-1]["text"] + switch},
            ]
        return {"role": "system", "content": content}
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
    # 本地推理模型可能把 <think> 段落写在 content 里，不写回对话历史。
    for block in output["content"] or []:
        if block.get("type") == "text" and block.get("content"):
            block["content"] = strip_think(block["content"])
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        output["reasoning_content"] = reasoning_content
    return output


class DeepSeekAgentLLM(OpenAILLM):
    """AgentDojo OpenAILLM 的 DeepSeek / Ollama 兼容版本。"""

    def __init__(
        self,
        client,
        model: str,
        reasoning_effort=None,
        temperature: float | None = 0.0,
        thinking: bool = False,
        cost_period: str = "auto",
    ) -> None:
        super().__init__(
            client,
            model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        self.thinking = thinking
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
            _message_to_deepseek(
                message,
                self.model,
                disable_thinking=not self.thinking,
            )
            for message in messages
        ]
        openai_tools = [
            _function_to_openai(tool)
            for tool in runtime.functions.values()
        ]
        create_args = {
            "model": self.model,
            "messages": openai_messages,
            **thinking_request_args(
                str(getattr(self.client, "base_url", "")),
                self.thinking,
                self.reasoning_effort,
            ),
        }
        if openai_tools:
            create_args["tools"] = openai_tools
            create_args["tool_choice"] = "auto"
        if not self.thinking:
            create_args["temperature"] = self.temperature

        completion = self.client.chat.completions.create(**create_args)
        self.total_usage = self.total_usage + usage_from_openai_response(
            completion,
            model=self.model,
            period=self.cost_period,
        )
        output = _openai_to_deepseek_assistant(
            completion.choices[0].message
        )
        return query, runtime, env, [*messages, output], extra_args
