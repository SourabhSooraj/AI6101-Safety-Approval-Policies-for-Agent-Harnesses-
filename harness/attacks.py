from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks import FixedJailbreakAttack, register_attack
from agentdojo.attacks.important_instructions_attacks import (
    ImportantInstructionsAttack,
)
from agentdojo.task_suite.task_suite import TaskSuite


@register_attack
class DeepSeekImportantInstructionsAttack(FixedJailbreakAttack):
    """
    AgentDojo important_instructions 攻击的 DeepSeek 版本。

    原版通过 pipeline 名称在 MODEL_NAMES 中查找模型名，DeepSeek 不在表中会直接
    报错；这里使用相同模板，只把 {model} 固定为 "DeepSeek"。
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
