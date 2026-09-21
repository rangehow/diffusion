# collators/v2.py
"""
CARD V2 Collator: Block-Pattern Noise Injection + Soft-Tail Masking Mixture

Design rationale:
    CARD V1 (CausalMLMCollator) only uses soft-tail masking during training.
    During inference, the model sees "clean prefix + contiguous [MASK] block" — 
    a pattern that NEVER appears in V1 training. This train-inference mismatch
    is the root cause of weak parallel generation.
    
    V2 mixes three masking modes:
      1. Soft-tail masking     (1 - p_block - eps)  — original CARD training
      2. Block-pattern masking (p_block)             — mimics inference exactly
      3. i.i.d. masking        (eps)                 — theoretical support coverage
    
    Mode 2 directly trains the model on inference-time noise patterns.
    Mode 3 ensures formal ELBO support coverage (Lemma A.2 of the paper).
    
    The V2 collator also supports partial-block patterns (simulating intermediate
    denoising steps) and adjustable reweighting aggressiveness per mode.
"""

from typing import List, Dict, Any, Optional
import random
import math

import torch
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin


class CausalMLMCollatorV2(BaseCollator, MaskedLanguageModelingMixin):
    """
    CARD V2 Data Collator with block-pattern noise injection.
    
    Three masking modes per sample:
        - "tail"  : soft-tail masking (original CARD)
        - "block" : clean prefix + contiguous mask block (inference pattern)
        - "iid"   : standard i.i.d. masking (support coverage)
    
    Args:
        tokenizer: HuggingFace tokenizer with mask_token_id
        max_length: maximum sequence length
        text_key: key to extract text from dataset examples
        start_prob: minimum masking ratio  (for tail mode)
        end_prob: maximum masking ratio    (for tail mode)
        tail_bias_factor: controls tail window width (λ in the paper)
        use_daum: whether to pass DAUM flag to model forward
        p_block: probability of block-pattern mode per sample
        eps_iid: probability of i.i.d. mode per sample (for support coverage)
        block_k_min: minimum block size for block-pattern mode
        block_k_max: maximum block size for block-pattern mode
        partial_block_prob: within block mode, probability of partial-block pattern
        block_weight_gamma: reweighting attenuation factor for block mode (0=uniform, 1=full DAUM)
        pad_to_max_length: whether to pad all sequences to max_length
        is_eval: if True, produce clean NTP data (no masking)
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
        # === V2 new parameters ===
        p_block: float = 0.2,
        eps_iid: float = 0.01,
        block_k_min: int = 4,
        block_k_max: int = 32,
        partial_block_prob: float = 0.5,
        block_weight_gamma: float = 0.3,
        # === existing parameters ===
        pad_to_max_length: bool = False,
        is_eval: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer, max_length, text_key,
            add_special_tokens=True,
            pad_to_max_length=pad_to_max_length
        )
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.tail_bias_factor = max(1.0, tail_bias_factor)
        self.use_daum = use_daum
        self.is_eval = is_eval

        # V2 parameters
        self.p_block = p_block
        self.eps_iid = eps_iid
        self.block_k_min = block_k_min
        self.block_k_max = block_k_max
        self.partial_block_prob = partial_block_prob
        self.block_weight_gamma = block_weight_gamma

        # Validate
        assert 0.0 <= p_block <= 1.0
        assert 0.0 <= eps_iid <= 1.0
        assert p_block + eps_iid <= 1.0, "p_block + eps_iid must be <= 1.0"
        assert block_k_min >= 1
        assert block_k_max >= block_k_min

    # ------------------------------------------------------------------
    # Masking mode implementations
    # ------------------------------------------------------------------

    def _apply_tail_masking(self, ids: List[int], prob: float) -> List[int]:
        """
        Original CARD soft-tail masking.
        Masks are sampled from a tail window of size min(λ * num_masks, maskable_len).
        """
        maskable = list(range(1, len(ids) - 1))  # exclude BOS and EOS
        if not maskable:
            return ids

        num = max(1, int(len(maskable) * prob))
        pool = min(int(num * self.tail_bias_factor), len(maskable))
        pool = max(pool, num)
        candidates = maskable[-pool:]
        indices_to_mask = random.sample(candidates, num)

        result = ids[:]
        for i in indices_to_mask:
            result[i] = self.mask_token_id
        return result

    def _apply_block_masking(self, ids: List[int]) -> List[int]:
        """
        Block-pattern masking: clean prefix + contiguous [MASK] block.
        Directly mimics inference-time noise pattern.
        
        Optionally produces a partial-block pattern (some positions within
        the block filled with ground-truth) to simulate intermediate denoising steps.
        """
        maskable_len = len(ids) - 2  # exclude BOS, EOS
        if maskable_len <= 0:
            return ids

        # Sample block size K
        k = random.randint(
            min(self.block_k_min, maskable_len),
            min(self.block_k_max, maskable_len)
        )

        # Block starts at the tail of maskable region
        # maskable indices: 1 to len(ids)-2
        block_start = len(ids) - 1 - k  # last maskable position is len(ids)-2
        block_end = len(ids) - 1         # exclusive (EOS position)

        result = ids[:]

        # Decide: pure block or partial block
        if random.random() < self.partial_block_prob:
            # Partial block: randomly fill some positions with ground-truth
            fill_fraction = random.random()  # uniform [0, 1)
            num_to_fill = int(fill_fraction * k)
            block_indices = list(range(block_start, block_end))
            
            # First mask everything in the block
            for i in block_indices:
                result[i] = self.mask_token_id
            
            # Then reveal some positions (simulate partial denoising)
            if num_to_fill > 0 and num_to_fill < k:
                reveal_indices = random.sample(block_indices, num_to_fill)
                for i in reveal_indices:
                    result[i] = ids[i]  # restore ground-truth
        else:
            # Pure block: all positions in block are masked
            for i in range(block_start, block_end):
                result[i] = self.mask_token_id

        return result

    def _apply_iid_masking(self, ids: List[int], prob: float) -> List[int]:
        """
        Standard i.i.d. masking: each position independently masked with probability prob.
        Used for theoretical support coverage (epsilon-smoothing).
        """
        result = ids[:]
        for i in range(1, len(ids) - 1):  # exclude BOS, EOS
            if random.random() < prob:
                result[i] = self.mask_token_id

        # Ensure at least one mask
        masked_count = sum(1 for i in range(1, len(ids) - 1) if result[i] == self.mask_token_id)
        if masked_count == 0:
            candidates = list(range(1, len(ids) - 1))
            if candidates:
                result[random.choice(candidates)] = self.mask_token_id

        return result

    def _select_mode(self) -> str:
        """Sample masking mode for one example."""
        r = random.random()
        if r < self.eps_iid:
            return "iid"
        elif r < self.eps_iid + self.p_block:
            return "block"
        else:
            return "tail"

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _, _, raw_seqs = self.prepare_batch(examples)
        B = len(raw_seqs)

        # ========== Eval mode: clean NTP ==========
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

        # ========== Train mode ==========
        # Sample noise levels using antithetic/stratified sampling
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        t = self.sample_antithetic_probs(B, low=p_min, high=p_max)
        batch_probs_tensor = (1.0 - 1e-5) * t

        batch_inputs = []
        batch_labels = []
        batch_modes = []        # track mode per sample for potential logging
        batch_weight_gammas = [] # per-sample gamma for reweighting attenuation

        for idx, seq in enumerate(raw_seqs):
            ids = seq.tolist()
            prob = batch_probs_tensor[idx].item()
            mode = self._select_mode()
            batch_modes.append(mode)

            if mode == "tail":
                corrupted = self._apply_tail_masking(ids, prob)
                batch_weight_gammas.append(1.0)  # full DAUM weighting
            elif mode == "block":
                corrupted = self._apply_block_masking(ids)
                batch_weight_gammas.append(self.block_weight_gamma)  # attenuated
            else:  # "iid"
                corrupted = self._apply_iid_masking(ids, prob)
                batch_weight_gammas.append(1.0)  # full DAUM weighting

            batch_inputs.append(torch.tensor(corrupted, dtype=torch.long))
            batch_labels.append(torch.tensor(ids, dtype=torch.long))

        # Pad to batch
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Weight gammas: [B] tensor, passed to model for per-sample reweighting control
        weight_gammas = torch.tensor(batch_weight_gammas, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": batch_probs_tensor,
            "causal": True,
            "use_daum": self.use_daum,
            "weight_gammas": weight_gammas,  # V2: per-sample DAUM attenuation
        }
