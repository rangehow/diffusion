# eval/tasks/piqa.py
"""
PIQA (Physical Interaction QA) task.
"""

from typing import Optional
import datasets
from .base import BaseTask
from .registry import register_task

@register_task("piqa")
class PIQATask(BaseTask):
    TASK_NAME = "piqa"
    QUESTION_COL = "goal"
    OPTIONS_COL = "options"
    ANSWER_COL = "label"
    IS_SENTENCE_COMPLETION = False
    
    def _preprocess(self, dataset: datasets.Dataset) -> datasets.Dataset:
        def create_options(example):
            example[self.OPTIONS_COL] = [example["sol1"], example["sol2"]]
            return example
        return dataset.map(create_options, num_proc=16)
    
    def _load_dataset(self) -> datasets.Dataset:
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "baber/piqa",
            split="validation",
            cache_dir=self.config.local_dir,
        )
        return self._preprocess(dataset)
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        if self.config.num_fewshot == 0:
            return None
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "baber/piqa",
            split="train",
            cache_dir=self.config.local_dir,
        )
        return self._preprocess(dataset)