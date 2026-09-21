# eval/tasks/sampler.py
"""
Few-shot context samplers for evaluation tasks.

Samplers are responsible for selecting examples from a few-shot pool
and formatting them into context strings.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import datasets

from ..config import SamplerType


class BaseSampler(ABC):
    """
    Base class for few-shot context samplers.
    
    Samplers hold a random number generator and a pool of few-shot examples.
    They provide methods to sample examples and format them into context strings.
    """
    
    def __init__(
        self, 
        fewshot_dataset: datasets.Dataset, 
        seed: int = 42
    ):
        """
        Initialize the sampler.
        
        Args:
            fewshot_dataset: Dataset to sample from
            seed: Random seed for reproducibility
        """
        self.fewshot_dataset = fewshot_dataset
        self.rng = random.Random(seed)
    
    @abstractmethod
    def _sample_indices(self, num_shots: int) -> List[int]:
        """
        Sample indices from the dataset.
        
        Args:
            num_shots: Number of samples to select
            
        Returns:
            List of indices into fewshot_dataset
        """
        pass
    
    def get_samples(self, num_shots: int) -> List[Dict]:
        """
        Get raw samples from the few-shot pool.
        
        Args:
            num_shots: Number of samples to retrieve
            
        Returns:
            List of sample dictionaries
        """
        if num_shots == 0:
            return []
        indices = self._sample_indices(num_shots)
        return self.fewshot_dataset.select(indices).to_list()
    
    def get_context(
        self,
        num_shots: int,
        question_col: str,
        options_col: str,
        answer_col: str,
        is_sentence_completion: bool = False,
    ) -> str:
        """
        Generate a formatted few-shot context string.
        
        Args:
            num_shots: Number of examples to include
            question_col: Column name for questions
            options_col: Column name for options
            answer_col: Column name for answers
            is_sentence_completion: If True, format as "question answer",
                otherwise as "Question: q\nAnswer: a"
                
        Returns:
            Formatted context string
        """
        if num_shots == 0:
            return ""
        
        samples = self.get_samples(num_shots)
        prompts = []
        
        for sample in samples:
            question = sample[question_col]
            options = sample[options_col]
            answer_val = sample[answer_col]
            
            # Determine the correct answer index
            answer_idx = self._parse_answer_index(answer_val, options)
            
            if 0 <= answer_idx < len(options):
                correct_answer = options[answer_idx]
                if is_sentence_completion:
                    prompt = f"{question.strip()} {correct_answer}"
                else:
                    prompt = f"Question: {question}\nAnswer: {correct_answer}"
                prompts.append(prompt)
        
        return "\n\n".join(prompts) + "\n\n" if prompts else ""
    
    @staticmethod
    def _parse_answer_index(answer_val: Any, options: List) -> int:
        """Parse answer value into an index."""
        if isinstance(answer_val, int):
            return answer_val
        if isinstance(answer_val, str):
            if answer_val.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return ord(answer_val.upper()) - ord('A')
            if answer_val.isdigit():
                return int(answer_val)
        if isinstance(answer_val, list) and 1 in answer_val:
            return answer_val.index(1)
        return -1


class FirstNSampler(BaseSampler):
    """Always selects the first N samples from the dataset."""
    
    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        return list(range(num_to_select))


class RandomSampler(BaseSampler):
    """Randomly selects N samples from the dataset."""
    
    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        return self.rng.sample(range(len(self.fewshot_dataset)), num_to_select)


class BalancedSampler(BaseSampler):
    """
    Selects samples with balanced class distribution.
    
    Tries to include equal numbers of samples from each answer class.
    """
    
    def __init__(
        self,
        fewshot_dataset: datasets.Dataset,
        answer_col: str,
        seed: int = 42,
    ):
        super().__init__(fewshot_dataset, seed)
        self.answer_col = answer_col
        self.indices_by_label = self._group_indices_by_label()
    
    def _group_indices_by_label(self) -> Dict[Any, List[int]]:
        """Group dataset indices by their answer label."""
        indices_map: Dict[Any, List[int]] = {}
        for i, item in enumerate(self.fewshot_dataset):
            label = item[self.answer_col]
            if label not in indices_map:
                indices_map[label] = []
            indices_map[label].append(i)
        return indices_map
    
    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        if not self.indices_by_label or num_to_select == 0:
            return []
        
        final_indices = []
        labels = list(self.indices_by_label.keys())
        self.rng.shuffle(labels)
        
        # Create consumable copies
        label_pools = {
            label: self.rng.sample(indices, len(indices))
            for label, indices in self.indices_by_label.items()
        }
        
        # Round-robin selection
        while len(final_indices) < num_to_select:
            found_any = False
            for label in labels:
                if label_pools[label]:
                    final_indices.append(label_pools[label].pop(0))
                    found_any = True
                    if len(final_indices) == num_to_select:
                        break
            if not found_any:
                break
        
        return final_indices


# Registry of available samplers
SAMPLER_REGISTRY = {
    SamplerType.FIRST_N: FirstNSampler,
    SamplerType.RANDOM: RandomSampler,
    SamplerType.BALANCED: BalancedSampler,
}


def create_sampler(
    sampler_type: SamplerType,
    fewshot_dataset: datasets.Dataset,
    seed: int = 42,
    **kwargs,
) -> BaseSampler:
    """
    Factory function to create a sampler.
    
    Args:
        sampler_type: Type of sampler to create
        fewshot_dataset: Dataset to sample from
        seed: Random seed
        **kwargs: Additional arguments (e.g., answer_col for BalancedSampler)
        
    Returns:
        Sampler instance
        
    Raises:
        ValueError: If sampler type is not recognized
    """
    # Convert string to enum if needed
    if isinstance(sampler_type, str):
        sampler_type = SamplerType(sampler_type)
    
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    if sampler_class is None:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            f"Available: {list(SAMPLER_REGISTRY.keys())}"
        )
    
    # BalancedSampler needs answer_col
    if sampler_type == SamplerType.BALANCED:
        if 'answer_col' not in kwargs:
            raise ValueError("BalancedSampler requires 'answer_col' argument")
        return sampler_class(fewshot_dataset, kwargs['answer_col'], seed)
    
    return sampler_class(fewshot_dataset, seed)