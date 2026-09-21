# eval/tasks/commonsense_qa.py
"""
CommonsenseQA: Multiple-choice QA requiring commonsense knowledge.
"""

from typing import Optional
import datasets
from .base import BaseTask
from .registry import register_task

@register_task("commonsense_qa")
class CommonsenseQATask(BaseTask):
    TASK_NAME = "commonsense_qa"
    QUESTION_COL = "question"
    OPTIONS_COL = "options"
    ANSWER_COL = "answerKey"
    IS_SENTENCE_COMPLETION = False
    
    def _preprocess(self, example: dict) -> dict:
        example[self.OPTIONS_COL] = example["choices"]["text"]
        return example
    
    def _load_dataset(self) -> datasets.Dataset:
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "tau/commonsense_qa",
            split="validation",
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._preprocess)
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        if self.config.num_fewshot == 0:
            return None
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "tau/commonsense_qa",
            split="train",
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._preprocess)