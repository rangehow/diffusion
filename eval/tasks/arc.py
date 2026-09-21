# eval/tasks/arc.py
"""
AI2 Reasoning Challenge (ARC) tasks.
"""

from typing import Optional
import datasets
from .base import BaseTask
from .registry import register_task

class ArcBaseTask(BaseTask):
    QUESTION_COL = "question"
    OPTIONS_COL = "options"
    ANSWER_COL = "answerKey"
    IS_SENTENCE_COMPLETION = False
    ARC_CONFIG_NAME: Optional[str] = None
    
    @staticmethod
    def _preprocess(example: dict) -> dict:
        example["options"] = example["choices"]["text"]
        return example
    
    def _load_dataset(self) -> datasets.Dataset:
        if self.ARC_CONFIG_NAME is None:
            raise ValueError("ARC_CONFIG_NAME must be set in subclass")
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "ai2_arc",
            self.ARC_CONFIG_NAME,
            split="test",
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._preprocess, num_proc=1)
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        if self.config.num_fewshot == 0:
            return None
        if self.ARC_CONFIG_NAME is None:
            raise ValueError("ARC_CONFIG_NAME must be set in subclass")
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "ai2_arc",
            self.ARC_CONFIG_NAME,
            split="train",
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._preprocess, num_proc=1)

@register_task("arc_easy")
class ArcEasyTask(ArcBaseTask):
    TASK_NAME = "arc_easy"
    ARC_CONFIG_NAME = "ARC-Easy"

@register_task("arc_challenge")
class ArcChallengeTask(ArcBaseTask):
    TASK_NAME = "arc_challenge"
    ARC_CONFIG_NAME = "ARC-Challenge"