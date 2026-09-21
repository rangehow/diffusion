# tasks/sciq.py

import datasets
from typing import Dict, List
import random

from .base import BaseTask, register_task, TaskConfig

@register_task("sciq")
class SCIQTask(BaseTask):
    """
    对 SciQ 数据集的任务实现。
    SciQ: https://huggingface.co/datasets/sciq
    """

    def _load_dataset(self) -> datasets.Dataset:
        """加载用于评估的测试集。"""
        return datasets.load_dataset("sciq", split="test")

    def _load_fewshot_dataset(self) -> datasets.Dataset:
        """加载用于 few-shot 学习的训练集。"""
        return datasets.load_dataset("sciq", split="train")

    def _format_fewshot_prompt(self, samples: List[Dict]) -> str:
        """为 SciQ 定制 few-shot 样本的格式化。"""
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
        """
        重写 process 方法以处理 SciQ 的特殊数据结构。
        - Prompt 由 'support' 和 'question' 组成。
        - Choices 由 'distractor1', 'distractor2', 'distractor3', 和 'correct_answer' 组成。
        """
        dataset_to_finalize = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_only is False.")

        # --- FIX: 修改此函数以接收全局索引 ---
        def _format_instance(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]:
            if self.config.text_only:
                new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
            else:
                new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [], "continuation_char_len": []}

            rng = self.sampler.rng if self.sampler else random.Random()

            # --- FIX: 使用 `enumerate(indices)` 来同时获取批内索引 i 和全局 group_id ---
            for i, group_id in enumerate(indices):
                # 1. 准备 Few-shot 上下文
                fewshot_prompt_str = ""
                if self.sampler and self.config.num_fewshot > 0:
                    fewshot_samples = self.sampler.get_samples(self.config.num_fewshot)
                    fewshot_prompt_str = self._format_fewshot_prompt(fewshot_samples)

                # 2. 准备当前问题的 Prompt 和 Choices (使用批内索引 i 来访问 batch 数据)
                support = batch['support'][i]
                question = batch['question'][i]
                correct_answer = batch['correct_answer'][i]
                distractors = [batch['distractor1'][i], batch['distractor2'][i], batch['distractor3'][i]]

                options = distractors + [correct_answer]
                rng.shuffle(options)
                
                prompt_str = f"{support.lstrip()}\nQuestion: {question}\nAnswer:"
                full_prompt_str = fewshot_prompt_str + prompt_str

                # 3. 遍历所有选项并生成样本
                for opt_text in options:
                    is_correct = 1 if opt_text == correct_answer else 0
                    continuation_str = " " + opt_text

                    if self.config.text_only:
                        new_batch["context_text"].append(full_prompt_str)
                        new_batch["continuation_text"].append(continuation_str)
                    else:
                        tokenizer = self.config.tokenizer
                        input_ids = tokenizer(full_prompt_str, add_special_tokens=False)['input_ids']
                        if tokenizer.bos_token_id is not None:
                            input_ids = [tokenizer.bos_token_id] + input_ids
                        continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']
                        
                        if self.config.model_type == 'discrete_diffusion' and tokenizer.eos_token_id is not None:
                            continuation_ids = continuation_ids + [tokenizer.eos_token_id]
                        
                        new_batch["input_ids"].append(input_ids)
                        new_batch["continuation_ids"].append(continuation_ids)
                        new_batch["continuation_len"].append(len(continuation_ids))
                        new_batch["continuation_char_len"].append(len(opt_text))
                    
                    new_batch["is_correct"].append(is_correct)
                    # --- FIX: 此处使用全局唯一的 group_id ---
                    new_batch["group_id"].append(group_id)
                    new_batch["task_name"].append("sciq")
            
            return new_batch

        return dataset_to_finalize.map(
            _format_instance, 
            batched=True, 
            # --- FIX: 添加 `with_indices=True` 以获取全局索引 ---
            with_indices=True,
            num_proc=16,
            remove_columns=dataset_to_finalize.column_names, 
            desc="[sciq] Processing",
            load_from_cache_file=False
        )