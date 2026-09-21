# collators/v3.py
"""
CARD V3 Collator: Topological Reordering for Causal Diffusion

Core idea:
    Physically reorder tokens so that ALL observed (unmasked) tokens appear
    before ALL masked tokens in the sequence. Under standard causal attention
    (lower-triangular mask), every masked token can now attend to every
    observed token — achieving bidirectional observed-context without any
    custom attention operator.

    RoPE uses the LOGICAL position IDs (original positions), so attention
    scores correctly reflect true token distances despite the physical reordering.

Physical layout after reordering:
    [BOS, obs₁, obs₂, ..., obs_k, EOS, [M]₁, [M]₂, ..., [M]_m]
     └───── observed prefix (clean) ─────┘ └── masked suffix ──────┘
    
    Position IDs: [0, pos(obs₁), ..., pos(obs_k), pos(EOS), pos(M₁), ..., pos(M_m)]
    Labels:       [obs₁, obs₂, ..., obs_k, EOS, clean(M₁), ..., clean(M_m)]
    (These are pre-shift; the model's label shift handles the "predict next" convention.)

Key properties:
    - Sequence length: L (same as ARM — NO doubling like WeDLM/BD3LM)
    - Attention operator: flash_attn_varlen_qkvpacked_func(causal=True) — STANDARD
    - Training cost: 1× ARM
    - Masked tokens see: ALL observed tokens (bidirectional in logical space)
    - DAUM weights: computed on REORDERED layout by the model (no pre-computation needed)
    - Inference compatible: "clean prefix + [M]...[M]" = exactly the V3 training pattern

Training signal decomposition:
    1. Observed→Observed: pure NTP among clean tokens (bonus supervised signal)
    2. Observed→Masked: transition predicts first masked token (strongest denoising signal)
    3. Masked→Masked: causal denoising chain among masked tokens

Requires model-side change:
    flash_attention_forward must accept custom position_ids for RoPE application.
    When position_ids is None, the original cu_seqlens-based sequential RoPE is used.
"""

from typing import List, Dict, Any
import random

import torch
import torch.nn.functional as F
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin


class CausalMLMCollatorV3(BaseCollator, MaskedLanguageModelingMixin):
    """
    CARD V3 Data Collator with topological reordering.
    
    For each training sample:
      1. Sample noise level t, determine mask positions (soft-tail masking)
      2. Partition positions into observed (O) and masked (M) sets
      3. Physically reorder: [O in original order] + [M in original order]
      4. Produce position_ids = logical (original) position of each physical slot
      5. Labels = clean tokens in the reordered order (model shift handles prediction target)
    
    DAUM weights are NOT pre-computed — the model computes them on the reordered
    input_ids, which naturally gives high weights to early-masked positions (they
    have clean observed context) and low weights to late-masked positions (they
    have accumulated masks in context). This is exactly the correct behavior for V3.
    
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

    # ------------------------------------------------------------------
    # Masking (same as V1)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Topological reordering
    # ------------------------------------------------------------------

    @staticmethod
    def _reorder(
        masked_ids: List[int],
        clean_ids: List[int],
        mask_indices_set: set,
    ):
        """
        Stable partition: observed positions first, masked positions second.
        Both groups preserve their original relative order.
        
        Returns:
            reordered_input: masked_ids in reordered physical order
            reordered_labels: clean_ids in reordered physical order
            position_ids: logical (original) position of each physical slot
        """
        L = len(masked_ids)
        observed = [i for i in range(L) if i not in mask_indices_set]
        masked = sorted(mask_indices_set)
        perm = observed + masked

        reordered_input = [masked_ids[i] for i in perm]
        reordered_labels = [clean_ids[i] for i in perm]

        return reordered_input, reordered_labels, perm

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)
        B = len(raw_seqs)

        # ========== Eval mode: clean NTP, no masking, no reorder ==========
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
        batch_position_ids = []

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

            # 3. Reorder: observed first, then masked
            reordered_input, reordered_labels, perm = \
                self._reorder(masked_ids, clean_ids, mask_indices_set)

            batch_inputs.append(torch.tensor(reordered_input, dtype=torch.long))
            batch_labels.append(torch.tensor(reordered_labels, dtype=torch.long))
            batch_position_ids.append(torch.tensor(perm, dtype=torch.long))

        # Pad to batch
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        position_ids = self._pad_batch(batch_position_ids, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "current_mlm_prob": batch_probs_tensor,
            "causal": True,
            "use_daum": self.use_daum,
        }
