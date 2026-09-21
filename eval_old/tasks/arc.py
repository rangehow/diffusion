import datasets
from typing import Optional

from .base import BaseTask, TaskConfig, register_task

# 帮助函数，用于从 'choices' 字典中提取选项文本
def _preprocess_arc(example: dict) -> dict:
    """
    将原始数据集的 'choices' 字段（包含 'text' 和 'label'）
    处理成一个名为 'options' 的纯文本选项列表。
    """
    example["options"] = example["choices"]["text"]
    return example


class ArcTask(BaseTask):
    """
    一个抽象基类，用于处理 AI2 Reasoning Challenge (ARC) 数据集。
    这个基类包含了 ARC-Easy 和 ARC-Challenge 任务的通用逻辑。
    子类需要定义 ARC_CONFIG_NAME 属性。
    """
    # 数据集中的原始列名
    QUESTION_COL: str = "question"
    ANSWER_COL: str = "answerKey"
    
    # 我们将通过预处理创建 'options' 列
    OPTIONS_COL: str = "options"
    
    # ARC 是一个问答任务，而不是句子补全任务
    IS_SENTENCE_COMPLETION: bool = False
    
    # 子类必须覆盖这个属性
    ARC_CONFIG_NAME: Optional[str] = None

    def _load_dataset(self) -> datasets.Dataset:
        """加载用于评估的测试集。"""
        if self.ARC_CONFIG_NAME is None:
            raise ValueError("ARC_CONFIG_NAME 必须在 ArcTask 的子类中设置！")
        
        # 加载测试集
        dataset = datasets.load_dataset(
            "ai2_arc", 
            self.ARC_CONFIG_NAME, 
            split="test", 
            cache_dir=self.config.local_dir
        )
        # 预处理数据以匹配 BaseTask 的期望格式
        return dataset.map(_preprocess_arc, num_proc=1)

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """如果需要，加载用于 few-shot 采样的训练集。"""
        if self.config.num_fewshot == 0:
            return None
            
        if self.ARC_CONFIG_NAME is None:
            raise ValueError("ARC_CONFIG_NAME 必须在 ArcTask 的子类中设置！")

        # 加载训练集
        fewshot_dataset = datasets.load_dataset(
            "ai2_arc", 
            self.ARC_CONFIG_NAME, 
            split="train",
            cache_dir=self.config.local_dir
        )
        # 预处理数据
        return fewshot_dataset.map(_preprocess_arc, num_proc=1)


@register_task("arc_easy")
class ArcEasyTask(ArcTask):
    """
    AI2 Reasoning Challenge (ARC) - Easy Set
    """
    TASK_NAME: str = "arc_easy"
    ARC_CONFIG_NAME: str = "ARC-Easy"


@register_task("arc_challenge")
class ArcChallengeTask(ArcTask):
    """
    AI2 Reasoning Challenge (ARC) - Challenge Set
    """
    TASK_NAME: str = "arc_challenge"
    ARC_CONFIG_NAME: str = "ARC-Challenge"