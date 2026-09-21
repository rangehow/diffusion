# tasks/base.py
import datasets
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from ..sampler import BaseContextSampler, get_sampler # 假设 sampler 在上一级目录

# --- 自动注册机制 ---
TASK_REGISTRY: Dict[str, Any] = {}
def register_task(name: str):
    """一个类装饰器，用于自动将任务类注册到 TASK_REGISTRY。"""
    def decorator(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f"任务 '{name}' 已被注册！")
        TASK_REGISTRY[name] = cls
        return cls
    return decorator



@dataclass
class TaskConfig:
    tokenizer: Optional[Any] = None
    num_fewshot: int = 0
    sampler_name: str = "first_n"
    sampler_seed: int = 42
    text_only: bool = False
    local_dir: Optional[str] = None
    model_type: str = "causal"  



class BaseTask:
    TASK_NAME: str = "base"
    QUESTION_COL: str = "question"
    OPTIONS_COL: str = "options"
    ANSWER_COL: str = "answer"
    IS_SENTENCE_COMPLETION: bool = False

    def __init__(self, config: TaskConfig):
        print(f"Initializing task: {self.TASK_NAME}")
        self.config = config
        self.fewshot_dataset = self._load_fewshot_dataset()
        self.sampler = self._setup_sampler()

    def _load_dataset(self) -> datasets.Dataset:
        raise NotImplementedError

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        return None

    def _setup_sampler(self) -> Optional[BaseContextSampler]:
        """为整个任务设置一个通用的 Sampler。"""
        if self.config.num_fewshot > 0 and self.fewshot_dataset:
            print(f"[{self.TASK_NAME}] Setting up a single '{self.config.sampler_name}' sampler with seed {self.config.sampler_seed}.")
            sampler_kwargs = {"fewshot_dataset": self.fewshot_dataset, "seed": self.config.sampler_seed}
            return get_sampler(self.config.sampler_name, **sampler_kwargs)
        return None

    def process(self) -> datasets.Dataset:
        dataset_to_finalize = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_only is False.")

        def _format_instance(batch: Dict[str, List]) -> Dict[str, List]:
            if self.config.text_only:
                new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
            else:
                new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [],"continuation_char_len": []}

            for i in range(len(batch[self.QUESTION_COL])):
                fewshot_prompt_str = ""
                if self.sampler:
                    fewshot_prompt_str = self.sampler.get_context(
                        self.config.num_fewshot, self.QUESTION_COL, self.OPTIONS_COL, self.ANSWER_COL, self.IS_SENTENCE_COMPLETION
                    )

                group_id = batch["id"][i]
                question_text = batch[self.QUESTION_COL][i]
                options = batch[self.OPTIONS_COL][i]
                labels = batch[self.ANSWER_COL][i]
                
                original_answer_index = -1
                if isinstance(labels, (str, int)):
                    try: original_answer_index = int(labels)
                    except ValueError:
                        if labels.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": original_answer_index = ord(labels.upper()) - ord('A')
                elif isinstance(labels, list): pass
                is_multi_label = isinstance(labels, list)

                for j, opt_text in enumerate(options):
                    is_correct = 1 if (is_multi_label and j < len(labels) and labels[j] == 1) or (not is_multi_label and j == original_answer_index) else 0
                    
                    prompt_str = fewshot_prompt_str + (f"{question_text.strip()}" if self.IS_SENTENCE_COMPLETION else f"Question: {question_text}\nAnswer:")
                    continuation_str = " " + opt_text

                    if self.config.text_only:
                        new_batch["context_text"].append(prompt_str)
                        new_batch["continuation_text"].append(continuation_str)
                    else:
                        tokenizer = self.config.tokenizer
                        input_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
                        if tokenizer.bos_token_id is not None:
                            input_ids = [tokenizer.bos_token_id] + input_ids
                        continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']
                        
                        # <<< MODIFIED: 核心修改点，只有在 discrete diffusion 时才添加 EOS token >>>
                        # if self.config.model_type == 'discrete_diffusion' and tokenizer.eos_token_id is not None:
                        #     # continuation_ids = continuation_ids + [tokenizer.eos_token_id]
                        #     # new_batch["continuation_len"].append(len(continuation_ids)-1)
                        #     continuation_ids = continuation_ids

                        new_batch["continuation_len"].append(len(continuation_ids))
                            
                        new_batch["input_ids"].append(input_ids)
                        new_batch["continuation_ids"].append(continuation_ids)
                        
                        new_batch["continuation_char_len"].append(len(opt_text))
                    
                    new_batch["is_correct"].append(is_correct)
                    new_batch["group_id"].append(group_id)
                    new_batch["task_name"].append(self.TASK_NAME)
            
            return new_batch

        if "id" not in dataset_to_finalize.column_names:
            dataset_to_finalize = dataset_to_finalize.add_column("id", range(len(dataset_to_finalize)))

        return dataset_to_finalize.map(_format_instance, num_proc=16,batched=True, remove_columns=dataset_to_finalize.column_names, desc=f"[{self.TASK_NAME}] Processing",load_from_cache_file=False)


def get_task(task_name: str, config: TaskConfig) -> BaseTask:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"未知的任务: '{task_name}'. 已注册的任务: {list(TASK_REGISTRY.keys())}")
    task_class = TASK_REGISTRY[task_name]
    return task_class(config)