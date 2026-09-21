# eval/tasks/hellaswag.py
"""
HellaSwag: A challenge dataset for grounded commonsense inference.
"""

import re
from typing import Optional
import datasets
from .base import BaseTask
from .registry import register_task

@register_task("hellaswag")
class HellaSwagTask(BaseTask):
    TASK_NAME = "hellaswag"
    QUESTION_COL = "query"
    OPTIONS_COL = "choices"
    ANSWER_COL = "gold_label"
    IS_SENTENCE_COMPLETION = True
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = text.strip().replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", text).replace("  ", " ")
    
    def _transform_doc(self, doc: dict) -> dict:
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": self._preprocess_text(doc["activity_label"] + ": " + ctx),
            "choices": [self._preprocess_text(ending) for ending in doc["endings"]],
            "gold_label": int(doc["label"]),
        }
    
    def _load_dataset(self) -> datasets.Dataset:
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            'Rowan/hellaswag',
            split='validation',
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._transform_doc, remove_columns=dataset.column_names)
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        if self.config.num_fewshot == 0:
            return None
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            'Rowan/hellaswag',
            split='train',
            cache_dir=self.config.local_dir,
        )
        return dataset.map(self._transform_doc, remove_columns=dataset.column_names)