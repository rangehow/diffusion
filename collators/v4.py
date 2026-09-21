# collators/v4.py
"""
CARD V4 Collator: Topological Reordering with Sequential Position IDs

The simplest possible approach to giving masked tokens access to all observed context.

Core idea:
    Physically reorder tokens: observed first, masked second, EOS last.
    Use STANDARD SEQUENTIAL position IDs (the model's default).
    Standard causal attention makes every masked token attend to every observed token.

    No custom position_ids. No model-side changes. No custom attention operators.
    Just a smarter arrangement of training data.

Layout after reordering:
    [BOS, obs₁, obs₂, ..., obs_k, [M]₁, [M]₂, ..., [M]_m, EOS]
    Position IDs: [0, 1, 2, ..., L-1]  (sequential — model default)

Why this works:
    Standard causal mask + physical reordering = masked tokens see all observed tokens.
    Sequential position IDs = all RoPE distances are positive = consistent with inference.
    Inference input is already [clean_prefix | masks] = exactly this layout.

Why NOT logical position IDs (V3):
    V3 assigns logical positions to reordered tokens. When a masked token at logical
    position 2 attends to an observed token at logical position 5, the RoPE distance
    is 2-5 = -3 (NEGATIVE). During inference, all distances are positive. This creates
    a train-inference mismatch where ~20-25% of training attention patterns are wasted.

Training signal decomposition:
    1. BOS → obs₁:          Standard next-token prediction
    2. obs_i → obs_{i+1}:   NTP on subsampled sequence (positional regularization)
    3. obs_k → [M]₁:        Transition: predict first gap token from full observed context
    4. [M]_i → [M]_{i+1}:   Causal denoising chain among masked positions
    5. [M]_m → EOS:         Predict end-of-sequence

Properties:
    - Sequence length: L (no doubling)
    - Attention operator: standard causal (flash_attn causal=True)
    - RoPE: standard sequential (no custom position_ids)
    - Model changes: NONE
    - Training cost: 1× ARM
    - Masked tokens see: ALL observed tokens
    - DAUM weights: naturally correct on reordered layout
"""

from typing import List, Dict, Any
import random

import torch
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin


class CausalMLMCollatorV4(BaseCollator, MaskedLanguageModelingMixin):
    """
    CARD V4 Data Collator: topological reordering with sequential position IDs.

    For each training sample:
      1. Sample noise level t, determine mask positions (soft-tail masking)
      2. Stable partition: BOS + observed (in order) + masked (in order) + EOS
      3. Labels = clean tokens in the reordered physical order
      4. No position_ids produced — model uses default sequential positions

    Args:
        tokenizer: HuggingFace tokenizer with mask_token_id
        max_length: maximum sequence length
        text_key: key to extract text from dataset examples
        start_prob: minimum masking ratio
        end_prob: maximum masking ratio
        tail_bias_factor: controls tail window width (λ in the paper)
        use_daum: whether to enable DAUM weighting in the model
        pad_to_max_length: whether to pad all sequences to max_length
        is_eval: if True, produce clean NTP data (no masking, no reordering)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        text_key: str = "text",
        start_prob: float = 0.0,
        end_prob: float = 1.0,
        tail_bias_factor: float = 1.5,
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
        self.tail_bias_factor = max(1.0, tail_bias_factor)
        self.use_daum = use_daum
        self.is_eval = is_eval

    def _tail_bias_indices(self, seq_len: int, prob: float) -> List[int]:
        """Sample mask positions from a tail-biased window.
        Positions 0 (BOS) and seq_len-1 (EOS) are always excluded."""
        maskable = list(range(1, seq_len - 1))
        if not maskable:
            return []
        num = max(1, int(len(maskable) * prob))
        pool = min(int(num * self.tail_bias_factor), len(maskable))
        pool = max(pool, num)
        candidates = maskable[-pool:]
        return random.sample(candidates, num)

    @staticmethod
    def _reorder(
        masked_ids: List[int],
        clean_ids: List[int],
        mask_indices_set: set,
        bos_idx: int = 0,
        eos_idx: int = -1,  # will be computed as len-1
    ):
        """
        Stable partition: BOS, then observed content tokens, then masked tokens, then EOS.
        Both observed and masked groups preserve their original relative order.

        Returns:
            reordered_input: input tokens in reordered physical order
            reordered_labels: clean tokens in reordered physical order
        """
        L = len(masked_ids)
        eos_idx = L - 1

        # Content indices: everything except BOS (0) and EOS (L-1)
        observed_content = [i for i in range(1, eos_idx) if i not in mask_indices_set]
        masked_content = sorted(mask_indices_set)  # already content-only by construction

        # Physical order: BOS → observed content → masked content → EOS
        perm = [bos_idx] + observed_content + masked_content + [eos_idx]

        reordered_input = [masked_ids[i] for i in perm]
        reordered_labels = [clean_ids[i] for i in perm]

        return reordered_input, reordered_labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)
        B = len(raw_seqs)

        # ========== Eval mode: clean NTP, no masking, no reordering ==========
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

        # ========== Train mode: mask → reorder ==========
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        t = self.sample_antithetic_probs(B, low=p_min, high=p_max)
        batch_probs_tensor = (1.0 - 1e-5) * t

        batch_inputs = []
        batch_labels = []

        for idx, seq in enumerate(raw_seqs):
            clean_ids = seq.tolist()
            prob = batch_probs_tensor[idx].item()

            # 1. Determine mask positions (soft-tail masking)
            mask_indices = self._tail_bias_indices(len(clean_ids), prob)
            mask_indices_set = set(mask_indices)

            # 2. Create masked version
            masked_ids = clean_ids[:]
            for i in mask_indices:
                masked_ids[i] = self.mask_token_id

            # 3. Reorder: BOS → observed → masked → EOS
            #    with SEQUENTIAL position IDs (model default)
            reordered_input, reordered_labels = self._reorder(
                masked_ids, clean_ids, mask_indices_set
            )

            batch_inputs.append(torch.tensor(reordered_input, dtype=torch.long))
            batch_labels.append(torch.tensor(reordered_labels, dtype=torch.long))

        # Pad to batch — NO position_ids produced (model uses sequential default)
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
