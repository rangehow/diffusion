# eval/tasks/winogrande.py
"""
Winogrande: Large-scale Winograd schema challenge.
"""

from typing import Dict
import datasets
from .base import BaseTask
from .registry import register_task

@register_task("winogrande")
class WinograndeTask(BaseTask):
    TASK_NAME = "winogrande"
    QUESTION_COL = "question"
    OPTIONS_COL = "options"
    ANSWER_COL = "answer"
    IS_SENTENCE_COMPLETION = True
    
    def _preprocess(self, doc: Dict) -> Dict:
        sentence = doc["sentence"]
        try:
            split_idx = sentence.index("_")
            question = sentence[:split_idx].strip()
            rest = sentence[split_idx + 1:].strip()
        except ValueError:
            question = sentence.strip()
            rest = ""
        
        option1 = doc["option1"]
        option2 = doc["option2"]
        
        cont1 = f"{option1} {rest}".strip() if rest else option1
        cont2 = f"{option2} {rest}".strip() if rest else option2
        
        answer_idx = int(doc["answer"]) - 1
        
        return {
            self.QUESTION_COL: question,
            self.OPTIONS_COL: [cont1, cont2],
            self.ANSWER_COL: answer_idx,
        }
    
    def _load_and_process(self, split: str) -> datasets.Dataset:
        # 使用 load_hf_dataset
        dataset = self.load_hf_dataset(
            "winogrande", "winogrande_xl",
            split=split,
            cache_dir=self.config.local_dir,
        )
        return dataset.map(
            self._preprocess,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Preprocessing {split}",
            load_from_cache_file=False,
        )
    
    def _load_dataset(self) -> datasets.Dataset:
        return self._load_and_process("validation")
    
    def _load_fewshot_dataset(self) -> datasets.Dataset:
        if self.config.num_fewshot == 0:
            return None
        return self._load_and_process("train")