"""
Ablation collators for decomposing CARD's contributions.

Variants:
1. PrefixLMCollator  — Prefix-LM baseline: random prefix, causal NTP on suffix (no diffusion)
2. UniformCausalMLMCollator — Causal masking but uniform (no tail-bias), with/without DAUM
"""

import random
from typing import List, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collators.base import BaseCollator
from collators.mixins import MaskedLanguageModelingMixin


class PrefixLMCollator(BaseCollator):
    """
    Prefix-LM baseline collator (Reviewer 2Bg4 request).
    
    Randomly selects a prefix length, keeps the prefix intact, and trains 
    the model autoregressively on the suffix. No masking, no diffusion framing.
    This tests whether diffusion adds value over a simple prefix-LM approach.
    
    Concretely, for each sample:
    - Sample prefix_ratio ~ Uniform(min_prefix_ratio, max_prefix_ratio)
    - prefix = tokens[:prefix_len], suffix = tokens[prefix_len:]
    - input_ids = full sequence (no masking)
    - labels = -100 for prefix positions, original tokens for suffix positions
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        text_key: str = "text",
        min_prefix_ratio: float = 0.0,
        max_prefix_ratio: float = 0.9,
        pad_to_max_length: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer, max_length, text_key,
            add_special_tokens=True,
            pad_to_max_length=pad_to_max_length,
        )
        self.min_prefix_ratio = min_prefix_ratio
        self.max_prefix_ratio = max_prefix_ratio

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)

        batch_inputs = []
        batch_labels = []

        for seq in raw_seqs:
            ids = seq.tolist()
            seq_len = len(ids)

            # Sample random prefix ratio
            ratio = random.uniform(self.min_prefix_ratio, self.max_prefix_ratio)
            # Ensure at least 1 token for BOS in prefix, 1 token for suffix
            prefix_len = max(1, min(seq_len - 1, int(seq_len * ratio)))

            # Labels: -100 for prefix, real tokens for suffix
            labels = [-100] * prefix_len + ids[prefix_len:]

            batch_inputs.append(torch.tensor(ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": None,  # No diffusion
            "causal": True,
            "use_daum": False,
        }


class UniformCausalMLMCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    Causal MLM with UNIFORM masking (no tail-bias).
    
    This ablation variant removes the tail-biased masking schedule
    while keeping the diffusion framing. It helps isolate the 
    contribution of tail-biased masking from the diffusion objective itself.
    
    Parameters:
        use_daum: If True, apply DAUM weighting. If False, no weighting.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        text_key: str = "text",
        start_prob: float = 0.0,
        end_prob: float = 1.0,
        use_daum: bool = False,
        pad_to_max_length: bool = False,
        is_eval: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer, max_length, text_key,
            add_special_tokens=True,
            pad_to_max_length=pad_to_max_length,
        )
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.use_daum = use_daum
        self.is_eval = is_eval

    def _uniform_mask_indices(self, seq_len: int, prob: float) -> List[int]:
        """
        Uniform random masking over all maskable positions (excluding BOS/EOS).
        No tail-bias applied.
        """
        maskable = list(range(1, seq_len - 1))
        if not maskable:
            return []
        num = max(1, int(len(maskable) * prob))
        # Key difference from CausalMLMCollator: sample from ALL positions uniformly
        return random.sample(maskable, min(num, len(maskable)))

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)
        B = len(raw_seqs)

        if self.is_eval:
            input_ids = self._pad_batch(raw_seqs, padding_value=self.pad_token_id)
            labels = self._pad_batch(raw_seqs, padding_value=-100)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "current_mlm_prob": None,
                "causal": True,
                "use_daum": False,
                "is_eval": True,
            }

        # Sample masking probabilities using antithetic sampling
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        t = self.sample_antithetic_probs(B, low=p_min, high=p_max)
        batch_probs_tensor = (1.0 - 1e-5) * t

        batch_inputs = []
        batch_labels = []

        for idx, seq in enumerate(raw_seqs):
            ids = seq.tolist()
            prob = batch_probs_tensor[idx].item()
            # UNIFORM masking (no tail bias)
            indices = self._uniform_mask_indices(len(ids), prob)

            inputs = ids[:]
            for i in indices:
                inputs[i] = self.mask_token_id

            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_labels.append(torch.tensor(ids, dtype=torch.long))

        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": batch_probs_tensor,
            "causal": True,
            "use_daum": self.use_daum,
        }
