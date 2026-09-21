# eval/tasks/sciq.py
"""
SciQ: Science question answering dataset.
"""

import random
from typing import Dict, List
import datasets
from .base import BaseTask
from .registry import register_task

@register_task("sciq")
class SciQTask(BaseTask):
    TASK_NAME = "sciq"
    QUESTION_COL = "question"
    OPTIONS_COL = "options"
    ANSWER_COL = "correct_answer"
    IS_SENTENCE_COMPLETION = False
    
    def _load_dataset(self) -> datasets.Dataset:
        # 使用 load_hf_dataset
        return self.load_hf_dataset("sciq", split="test", cache_dir=self.config.local_dir)
    
    def _load_fewshot_dataset(self) -> datasets.Dataset:
        if self.config.num_fewshot == 0:
            return None
        # 使用 load_hf_dataset
        return self.load_hf_dataset("sciq", split="train", cache_dir=self.config.local_dir)
    
    def _format_fewshot_prompt(self, samples: List[Dict]) -> str:
        prompts = []
        for sample in samples:
            prompt = (
                f"{sample['support'].lstrip()}\n"
                f"Question: {sample['question']}\n"
                f"Answer: {sample['correct_answer']}"
            )
            prompts.append(prompt)
        return "\n\n".join(prompts) + "\n\n" if prompts else ""
    
    def process(self) -> datasets.Dataset:
        dataset = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer required when text_only=False")
        
        return dataset.map(
            self._format_batch_sciq,
            batched=True,
            with_indices=True,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Processing",
            load_from_cache_file=False,
        )
    
    def _format_batch_sciq(self, batch: Dict[str, List], indices: List[int]) -> Dict[str, List]:
        if self.config.text_only:
            new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
        else:
            new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [], "continuation_char_len": []}
        
        rng = self.sampler.rng if self.sampler else random.Random(42)
        
        for i, group_id in enumerate(indices):
            fewshot_str = ""
            if self.sampler and self.config.num_fewshot > 0:
                samples = self.sampler.get_samples(self.config.num_fewshot)
                fewshot_str = self._format_fewshot_prompt(samples)
            
            support = batch['support'][i]
            question = batch['question'][i]
            correct_answer = batch['correct_answer'][i]
            distractors = [batch['distractor1'][i], batch['distractor2'][i], batch['distractor3'][i]]
            
            options = distractors + [correct_answer]
            rng.shuffle(options)
            
            prompt = f"{support.lstrip()}\nQuestion: {question}\nAnswer:"
            full_prompt = fewshot_str + prompt
            
            for opt_text in options:
                is_correct = 1 if opt_text == correct_answer else 0
                continuation = " " + opt_text
                
                if self.config.text_only:
                    new_batch["context_text"].append(full_prompt)
                    new_batch["continuation_text"].append(continuation)
                else:
                    self._tokenize_sample(new_batch, full_prompt, continuation)
                
                new_batch["is_correct"].append(is_correct)
                new_batch["group_id"].append(group_id)
                new_batch["task_name"].append(self.TASK_NAME)
        return new_batch