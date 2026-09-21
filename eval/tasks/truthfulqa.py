# eval/tasks/truthfulqa.py
"""
TruthfulQA evaluation tasks.
"""

import datasets
from .base import BaseTask
from .registry import register_task

class TruthfulQABaseTask(BaseTask):
    QUESTION_COL = "question"
    OPTIONS_COL = "choices"
    ANSWER_COL = "labels"
    IS_SENTENCE_COMPLETION = False
    TARGET_KEY: str = None
    
    def _flatten_targets(self, example: dict) -> dict:
        target = example[self.TARGET_KEY]
        example[self.OPTIONS_COL] = target["choices"]
        example[self.ANSWER_COL] = target["labels"]
        return example
    
    def _load_dataset(self) -> datasets.Dataset:
        if self.TARGET_KEY is None:
            raise ValueError("TARGET_KEY must be set in subclass")
        
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "truthful_qa", "multiple_choice",
            split="validation",
            cache_dir=self.config.local_dir,
        )
        return dataset.map(
            self._flatten_targets,
            num_proc=4,
            desc=f"[{self.TASK_NAME}] Flattening targets",
        )

@register_task("truthfulqa_mc1")
class TruthfulQAMC1Task(TruthfulQABaseTask):
    TASK_NAME = "truthfulqa_mc1"
    TARGET_KEY = "mc1_targets"

@register_task("truthfulqa_mc2")
class TruthfulQAMC2Task(TruthfulQABaseTask):
    TASK_NAME = "truthfulqa_mc2"
    TARGET_KEY = "mc2_targets"