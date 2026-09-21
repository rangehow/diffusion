# tasks/piqa.py

import datasets
from typing import Optional

from .base import BaseTask, register_task

@register_task("piqa")
class PIQATask(BaseTask):
    """
    PIQA (Physical Interaction: Question Answering) 任务的实现。
    数据集地址: https://huggingface.co/datasets/baber/piqa
    """
    TASK_NAME: str = "piqa"
    
    # 根据 lm-eval 配置映射列名
    QUESTION_COL: str = "goal"
    OPTIONS_COL: str = "options"  # 我们将通过预处理创建这个列
    ANSWER_COL: str = "label"
    
    # 提示格式是 "Question: ...\nAnswer:"，不是句子补全
    IS_SENTENCE_COMPLETION: bool = False

    def _preprocess_piqa(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        对 PIQA 数据集进行预处理。
        主要任务是将 'sol1' 和 'sol2' 两列合并成一个新的 'options' 列。
        """
        def create_options_column(example):
            # 将 sol1 和 sol2 组合成一个选项列表
            example[self.OPTIONS_COL] = [example["sol1"], example["sol2"]]
            return example

        # 使用 map 函数高效地添加新列
        dataset = dataset.map(create_options_column, num_proc=16)
        return dataset

    def _load_dataset(self) -> datasets.Dataset:
        """
        加载用于评估的 PIQA validation split。
        """
        # 从 Hugging Face Hub 加载 validation split
        dataset = datasets.load_dataset(
            "baber/piqa", 
            split="validation", 
            cache_dir=self.config.local_dir
        )
        # 应用预处理
        return self._preprocess_piqa(dataset)

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """
        加载用于 few-shot 示例的 PIQA train split。
        """
        if self.config.num_fewshot == 0:
            return None
        
        # 从 Hugging Face Hub 加载 train split
        dataset = datasets.load_dataset(
            "baber/piqa", 
            split="train", 
            cache_dir=self.config.local_dir
        )
        # 应用预处理
        return self._preprocess_piqa(dataset)