# eval/tasks/base.py
"""
Base class for evaluation tasks.
"""

from typing import Dict, List, Any, Optional

import datasets

from ..config import TaskConfig, ModelType
from .sampler import BaseSampler, create_sampler
from ..dataset_paths import get_local_path


class BaseTask:
    """
    Abstract base class for evaluation tasks.
    """
    
    # Task metadata - override in subclasses
    TASK_NAME: str = "base"
    QUESTION_COL: str = "question"
    OPTIONS_COL: str = "options"
    ANSWER_COL: str = "answer"
    IS_SENTENCE_COMPLETION: bool = False
    
    def __init__(self, config: TaskConfig):
        """
        Initialize the task.
        """
        print(f"[TASK] Initializing: {self.TASK_NAME}")
        self.config = config
        self.fewshot_dataset = self._load_fewshot_dataset()
        self.sampler = self._setup_sampler()

    def load_hf_dataset(self, hf_id: str, *args, **kwargs) -> datasets.Dataset:
        """
        Helper to load dataset with local fallback priority.
        """
        local_path = get_local_path(self.TASK_NAME)
        
        if local_path:
            print(f"[{self.TASK_NAME}] Found local configuration: {local_path}")
            try:
                kwargs_local = kwargs.copy()
                if 'cache_dir' in kwargs_local:
                    del kwargs_local['cache_dir']
                
                print(f"[{self.TASK_NAME}] Loading from local path...")
                return datasets.load_dataset(local_path, *args, **kwargs_local)
            except Exception as e:
                print(f"[{self.TASK_NAME}] [WARNING] Local load failed: {e}")
                print(f"[{self.TASK_NAME}] Falling back to HuggingFace ID: {hf_id}")
        
        return datasets.load_dataset(hf_id, *args, **kwargs)
    
    def _load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        raise NotImplementedError("Subclass must implement _load_dataset()")
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """Load the dataset for few-shot examples."""
        return None
    
    def _setup_sampler(self) -> Optional[BaseSampler]:
        """Set up the few-shot sampler."""
        if self.config.num_fewshot > 0 and self.fewshot_dataset is not None:
            print(f"[{self.TASK_NAME}] Setting up '{self.config.sampler_type.value}' "
                  f"sampler with seed {self.config.sampler_seed}")
            return create_sampler(
                self.config.sampler_type,
                self.fewshot_dataset,
                self.config.sampler_seed,
            )
        return None
    
    def process(self) -> datasets.Dataset:
        """Process the dataset into evaluation format."""
        dataset = self._load_dataset()
        
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_only=False")
        
        # Add IDs if not present
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))
        
        # Process using map
        return dataset.map(
            self._format_batch,
            batched=True,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Processing",
            load_from_cache_file=False,
        )
    
    def _format_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Format a batch of samples into evaluation format."""
        if self.config.text_only:
            new_batch = {
                "context_text": [],
                "continuation_text": [],
                "is_correct": [],
                "group_id": [],
                "task_name": [],
            }
        else:
            new_batch = {
                "input_ids": [],
                "continuation_ids": [],
                "is_correct": [],
                "group_id": [],
                "task_name": [],
                "continuation_len": [],
                "continuation_char_len": [],
            }
        
        batch_size = len(batch[self.QUESTION_COL])
        
        for i in range(batch_size):
            fewshot_str = ""
            if self.sampler:
                fewshot_str = self.sampler.get_context(
                    self.config.num_fewshot,
                    self.QUESTION_COL,
                    self.OPTIONS_COL,
                    self.ANSWER_COL,
                    self.IS_SENTENCE_COMPLETION,
                )
            
            group_id = batch["id"][i]
            question = batch[self.QUESTION_COL][i]
            options = batch[self.OPTIONS_COL][i]
            answer = batch[self.ANSWER_COL][i]
            
            correct_idx = self._parse_answer_index(answer, options)
            is_multi_label = isinstance(answer, list)
            
            if self.IS_SENTENCE_COMPLETION:
                prompt = f"{fewshot_str}{question.strip()}"
            else:
                prompt = f"{fewshot_str}Question: {question}\nAnswer:"
            
            for j, option_text in enumerate(options):
                if is_multi_label:
                    is_correct = 1 if j < len(answer) and answer[j] == 1 else 0
                else:
                    is_correct = 1 if j == correct_idx else 0
                
                continuation = " " + option_text
                
                if self.config.text_only:
                    new_batch["context_text"].append(prompt)
                    new_batch["continuation_text"].append(continuation)
                else:
                    self._tokenize_sample(new_batch, prompt, continuation)
                
                new_batch["is_correct"].append(is_correct)
                new_batch["group_id"].append(group_id)
                new_batch["task_name"].append(self.TASK_NAME)
        
        return new_batch
    
    def _tokenize_sample(self, batch: Dict[str, List], prompt: str, continuation: str):
        """
        Tokenize a single sample and add to batch.
        
        FIXED: Now properly handles BOS and EOS tokens for both causal and diffusion models.
        - BOS is added at the start of the prompt
        - EOS is added at the end of the continuation for diffusion models
        """
        tokenizer = self.config.tokenizer
        model_type = self.config.model_type
        
        # Tokenize prompt (without special tokens - we'll add them manually)
        input_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        
        # Add BOS at the beginning
        if tokenizer.bos_token_id is not None:
            input_ids = [tokenizer.bos_token_id] + input_ids
        
        # Tokenize continuation
        continuation_ids = tokenizer(continuation, add_special_tokens=False)['input_ids']
        
        if model_type == ModelType.DISCRETE_DIFFUSION:
            if tokenizer.eos_token_id is not None:
                continuation_ids = continuation_ids + [tokenizer.eos_token_id]
        
        batch["input_ids"].append(input_ids)
        batch["continuation_ids"].append(continuation_ids)
        batch["continuation_len"].append(len(continuation_ids))
        batch["continuation_char_len"].append(len(continuation.strip()))
    
    @staticmethod
    def _parse_answer_index(answer: Any, options: List) -> int:
        """Parse answer value into an option index."""
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            answer_upper = answer.upper()
            if answer_upper in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return ord(answer_upper) - ord('A')
            if answer.isdigit():
                return int(answer)
        if isinstance(answer, list) and 1 in answer:
            return answer.index(1)
        return -1