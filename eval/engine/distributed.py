# eval/engine/distributed.py
"""
Distributed evaluation using Ray.
"""

from typing import Dict, List, Any
from itertools import cycle

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..config import ModelConfig
from ..models import load_model
from ..data import LogProbCollator


@ray.remote(num_gpus=1)
class EvaluatorActor:
    """
    Ray actor for distributed evaluation.
    
    Each actor loads a copy of the model on a single GPU and processes batches.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the evaluator actor.
        
        Args:
            config: Model configuration
        """
        self.adapter = load_model(config)
        self.tokenizer = self.adapter.tokenizer
    
    def evaluate_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate a single batch and return results.
        
        Args:
            batch: Collated batch from LogProbCollator
            
        Returns:
            List of result dictionaries, one per item in the batch
        """
        logprobs = self.adapter.compute_logprobs(batch)
        
        return [
            {
                'group_id': gid,
                'logprob': lp,
                'is_correct': ic,
                'continuation_len': cl,
                'continuation_char_len': ccl,
            }
            for gid, lp, ic, cl, ccl in zip(
                batch['group_ids'],
                logprobs,
                batch['is_correct'],
                batch['continuation_len'],
                batch['continuation_char_len'],
            )
        ]


class DistributedEvaluator:
    """
    Manages distributed evaluation across multiple GPUs using Ray.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_gpus: int = None,
    ):
        """
        Initialize the distributed evaluator.
        
        Args:
            config: Model configuration
            num_gpus: Number of GPUs to use. If None, uses all available.
        """
        self.config = config
        
        if not ray.is_initialized():
            ray.init()
        
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            raise ValueError("No GPUs found for distributed evaluation.")
        
        print(f"[INFO] Initializing {num_gpus} evaluator actors...")
        self.actors = [
            EvaluatorActor.remote(config) for _ in range(num_gpus)
        ]
        self.num_gpus = num_gpus
    
    def evaluate(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset across all GPUs.
        
        Args:
            dataset: HuggingFace dataset with processed samples
            tokenizer: Tokenizer for the collator
            batch_size: Batch size per GPU
            num_workers: DataLoader workers
            
        Returns:
            List of all result dictionaries
        """
        print(f"[INFO] Evaluating {len(dataset)} samples on {self.num_gpus} GPUs...")
        
        collator = LogProbCollator(tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )
        
        # Distribute batches across actors in round-robin fashion
        futures = [
            actor.evaluate_batch.remote(batch)
            for actor, batch in zip(cycle(self.actors), dataloader)
        ]
        
        # Collect results with progress bar
        all_results = []
        for future in tqdm(futures, desc="Collecting results"):
            batch_results = ray.get(future)
            all_results.extend(batch_results)
        
        return all_results
    
    def shutdown(self):
        """Shutdown Ray and release resources."""
        ray.shutdown()


def evaluate_dataset(
    config: ModelConfig,
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Convenience function for single-use evaluation.
    
    Args:
        config: Model configuration
        dataset: Dataset to evaluate
        tokenizer: Tokenizer for collation
        batch_size: Batch size
        num_workers: DataLoader workers
        
    Returns:
        List of result dictionaries
    """
    evaluator = DistributedEvaluator(config)
    try:
        return evaluator.evaluate(dataset, tokenizer, batch_size, num_workers)
    finally:
        evaluator.shutdown()