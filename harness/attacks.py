from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks import FixedJailbreakAttack, register_attack
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME
from agentdojo.attacks.important_instructions_attacks import (
    ImportantInstructionsAttack,
)
from agentdojo.task_suite.task_suite import TaskSuite

# 模型 id 中的关键词 -> 攻击文本里称呼的模型名，按顺序匹配。
_MODEL_NAME_HINTS = (
    ("deepseek", "DeepSeek"),
    ("llama", "Llama"),
    ("qwen", "Qwen"),
    ("mixtral", "Mistral"),
    ("mistral", "Mistral"),
    ("gemma", "Gemma"),
)


def attack_model_name_for(model: str) -> str:
    """根据 agent 的模型 id 推断攻击文本中 {model} 的称呼。"""

    lowered = model.lower()
    for hint, name in _MODEL_NAME_HINTS:
        if hint in lowered:
            return name
    return DEFAULT_MODEL_NAME


@register_attack
class DeepSeekImportantInstructionsAttack(FixedJailbreakAttack):
    """
    AgentDojo important_instructions 攻击的非查表版本。

    原版通过 pipeline 名称在 MODEL_NAMES 中查找模型名，DeepSeek 等不在表中会
    直接报错；这里使用相同模板，{model} 默认为 "DeepSeek"，运行脚本可通过
    --attack-model-name 覆盖 model_name（例如 dev 阶段用 Llama/Qwen）。
    """

    name = "important_instructions_deepseek"

    def __init__(
        self,
        task_suite: TaskSuite,
        target_pipeline: BasePipelineElement,
    ) -> None:
        super().__init__(
            ImportantInstructionsAttack._JB_STRING,
            task_suite,
            target_pipeline,
        )
        self.user_name = "Emma Johnson"
        self.model_name = "DeepSeek"
