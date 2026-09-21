"""
CARD V5 Suffix Mask Collator — for adapter-based training.

Core idea:
    Sample a random "information horizon" c. Positions 1..c keep their
    original tokens; positions c+1..L are replaced with [MASK].
    Labels are always the clean token sequence.
    No per-position weights. No reordering. No custom position_ids.

    This is the simplest possible diffusion collator and produces
    exactly the inference-time pattern: [clean prefix | mask block].

    The noise schedule is controlled by a single parameter alpha:
      - alpha = 1.0: uniform t, equivalent to linear schedule ELBO
      - alpha < 1.0: more high-noise samples (denoising emphasis)
      - alpha > 1.0: more low-noise samples (NTP emphasis)

Output fields:
    input_ids:        [BOS, x₁, ..., x_c, [M], ..., [M], EOS]
    labels:           [BOS, x₁, ..., x_c, x_{c+1}, ..., x_L, EOS]
    attention_mask:   1s for non-pad
    noise_fraction:   [B, L] tensor, cumulative mask fraction at each position
    causal:           True
    use_daum:         False (always — V5 uses no per-position weighting)
"""
from typing import List, Dict, Any
import random
import math

import torch
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin


class SuffixMaskCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    Suffix masking collator for CARD adapter training.
    
    Args:
        tokenizer: HuggingFace tokenizer with mask_token_id
        max_length: maximum sequence length
        text_key: key to extract text from dataset examples
        alpha: noise schedule power (1.0 = uniform = linear ELBO)
        pad_to_max_length: whether to pad to max_length
        is_eval: if True, produce clean NTP data (no masking)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        text_key: str = "text",
        alpha: float = 1.0,
        pad_to_max_length: bool = False,
        is_eval: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer, max_length, text_key,
            add_special_tokens=True,
            pad_to_max_length=pad_to_max_length,
        )
        self.alpha = alpha
        self.is_eval = is_eval

    def _sample_horizon(self, L: int) -> int:
        """Sample information horizon c from power-law distribution.
        
        c/L ~ Beta(1/alpha, 1), so P(c/L < x) = x^(1/alpha).
        When alpha=1: uniform on [1, L-1].
        When alpha>1: biased toward large c (more NTP-like).
        When alpha<1: biased toward small c (more denoising).
        """
        u = random.random()
        frac = u ** (1.0 / self.alpha)
        c = max(1, min(int(L * frac), L - 1))
        return c

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)
        B = len(raw_seqs)

        # ──── Eval mode: pure NTP ────
        if self.is_eval:
            input_ids = self._pad_batch(raw_seqs, padding_value=self.pad_token_id)
            labels = self._pad_batch(raw_seqs, padding_value=-100)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "causal": True,
                "use_daum": False,
            }

        # ──── Train mode: suffix masking ────
        batch_inputs = []
        batch_labels = []
        batch_noise_fracs = []

        for seq in raw_seqs:
            ids = seq.tolist()
            L = len(ids)

            # Sample horizon (positions 0=BOS and L-1=EOS are always clean)
            # Maskable range is [1, L-2]. Horizon c means positions 1..c are clean.
            maskable_len = L - 2  # exclude BOS and EOS
            if maskable_len <= 1:
                # Sequence too short to mask
                batch_inputs.append(seq.clone())
                batch_labels.append(seq.clone())
                batch_noise_fracs.append(torch.zeros(L))
                continue

            c = self._sample_horizon(maskable_len)
            # c is in [1, maskable_len-1] = number of clean content tokens
            # Positions 0 (BOS), 1..c (clean content), c+1..L-2 (masked), L-1 (EOS)

            input_ids = ids[:]
            for i in range(c + 1, L - 1):  # mask from c+1 to second-to-last
                input_ids[i] = self.mask_token_id

            # Noise fraction: cumulative mask count / position index
            noise_frac = torch.zeros(L)
            mask_count = 0
            for n in range(L):
                if input_ids[n] == self.mask_token_id:
                    mask_count += 1
                noise_frac[n] = mask_count / (n + 1) if (n + 1) > 0 else 0.0

            batch_inputs.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(seq.clone())
            batch_noise_fracs.append(noise_frac)

        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # Pad noise_fraction
        max_len = input_ids.size(1)
        padded_nf = torch.zeros(B, max_len)
        for i, nf in enumerate(batch_noise_fracs):
            padded_nf[i, :len(nf)] = nf

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "noise_fraction": padded_nf,
            "causal": True,
            "use_daum": False,
        }
