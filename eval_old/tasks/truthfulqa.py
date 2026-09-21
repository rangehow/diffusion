# tasks/truthfulqa.py

import datasets
from .base import BaseTask, TaskConfig, register_task

@register_task("truthfulqa_mc1")
class TruthfulQAMC1Task(BaseTask):
    """
    TruthfulQA Multiple Choice (MC1) 任务。
    此任务评估模型在选择正确答案上的准确率。
    问题可能有多于一个的正确答案。
    """
    TASK_NAME: str = "truthfulqa_mc1"
    QUESTION_COL: str = "question"
    OPTIONS_COL: str = "choices"  # 在 map 函数中创建
    ANSWER_COL: str = "labels"    # 在 map 函数中创建
    IS_SENTENCE_COMPLETION: bool = False # 格式为 "Question: ... Answer: "

    def _load_dataset(self) -> datasets.Dataset:
        """
        加载 TruthfulQA 数据集，并将其格式化以适应 BaseTask 的处理流程。
        我们只关心 'multiple_choice' 配置中的 'mc1_targets' 部分。
        """
        # 加载验证集
        dataset = datasets.load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        # 将嵌套的 mc1_targets 字典扁平化，创建 'choices' 和 'labels' 列
        def _flatten_mc1(example):
            example[self.OPTIONS_COL] = example["mc1_targets"]["choices"]
            example[self.ANSWER_COL] = example["mc1_targets"]["labels"]
            return example
            
        return dataset.map(_flatten_mc1, num_proc=4, desc=f"[{self.TASK_NAME}] Flattening MC1 targets")


@register_task("truthfulqa_mc2")
class TruthfulQAMC2Task(BaseTask):
    """
    TruthfulQA Multiple Choice (MC2) 任务。
    此任务评估模型分配给所有正确答案的归一化概率质量。
    你的 main.py 已经有专门的 `calculate_mc2_prob_mass` 函数来处理这个指标。
    """
    TASK_NAME: str = "truthfulqa_mc2"
    QUESTION_COL: str = "question"
    OPTIONS_COL: str = "choices"  # 在 map 函数中创建
    ANSWER_COL: str = "labels"    # 在 map 函数中创建
    IS_SENTENCE_COMPLETION: bool = False # 格式为 "Question: ... Answer: "

    def _load_dataset(self) -> datasets.Dataset:
        """
        加载 TruthfulQA 数据集，并将其格式化以适应 BaseTask 的处理流程。
        我们只关心 'multiple_choice' 配置中的 'mc2_targets' 部分。
        """
        # 加载验证集
        dataset = datasets.load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        # 将嵌套的 mc2_targets 字典扁平化，创建 'choices' 和 'labels' 列
        def _flatten_mc2(example):
            example[self.OPTIONS_COL] = example["mc2_targets"]["choices"]
            example[self.ANSWER_COL] = example["mc2_targets"]["labels"]
            return example
            
        return dataset.map(_flatten_mc2, num_proc=4, desc=f"[{self.TASK_NAME}] Flattening MC2 targets")