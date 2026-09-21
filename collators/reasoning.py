"""
Seq2seq-aware collators for reasoning tasks (Countdown, Sudoku, etc.)

These collators handle prompt/response splits:
- Prompt tokens are kept clean (never masked) — treated as conditioning context
- Response tokens are masked/corrupted according to the model paradigm
- Loss is computed ONLY on response tokens

Supports four modes:
- 'card': CARD's shifted causal attention with tail masking on response portion
- 'ar':   Standard causal LM (labels=-100 for prompt, real labels for response)
- 'mdlm': Bidirectional masking on response tokens only (standard MDLM DAUM 1/t)
- 'mdm':  Multi-granularity Diffusion Modeling (Ye et al. 2024) — discrete t, linear time weight, focal loss
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin


SEP_TOKEN_TEXT = " [SEP] "


def _tokenize_seq2seq_common(example, text_key, tokenizer, max_length):
    """
    Shared tokenization for all reasoning collators.
    Returns (token_ids, prompt_length).
    Format: [BOS] prompt_tokens response_tokens [EOS]
    """
    text = example[text_key]
    
    if SEP_TOKEN_TEXT in text:
        prompt_part, response_part = text.split(SEP_TOKEN_TEXT, 1)
    elif "[SEP]" in text:
        prompt_part, response_part = text.split("[SEP]", 1)
    else:
        prompt_part = ""
        response_part = text
    
    if prompt_part:
        prompt_text = prompt_part
        if tokenizer.bos_token:
            prompt_text = tokenizer.bos_token + prompt_text
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, truncation=False
        )
    else:
        prompt_ids = []
        if tokenizer.bos_token:
            prompt_ids = [tokenizer.bos_token_id]
    
    response_text = response_part
    if tokenizer.eos_token:
        response_text = response_text + tokenizer.eos_token
    response_ids = tokenizer.encode(
        response_text, add_special_tokens=False, truncation=False
    )
    
    all_ids = prompt_ids + response_ids
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
    
    prompt_length = min(len(prompt_ids), len(all_ids))
    return all_ids, prompt_length


class ReasoningCARDCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    CARD collator for seq2seq reasoning tasks.
    
    - Prompt tokens: kept clean, loss weight = 0 (labels = -100)
    - Response tokens: masked with CARD's tail-biased strategy, 
      loss computed with context-aware reweighting
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
        text_key: str = "text",
        start_prob: float = 1.0,
        end_prob: float = 0.0001,
        tail_bias_factor: float = 1.5,
        use_daum: bool = True,
        pad_to_max_length: bool = True,
        **kwargs,
    ):
        super().__init__(tokenizer, max_length, text_key, 
                        add_special_tokens=True, pad_to_max_length=pad_to_max_length)
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.tail_bias_factor = max(1.0, tail_bias_factor)
        self.use_daum = use_daum
    
    def _tokenize_seq2seq(self, example):
        return _tokenize_seq2seq_common(example, self.text_key, self.tokenizer, self.max_length)
    
    def _tail_bias_indices_in_range(self, start: int, end: int, prob: float) -> List[int]:
        maskable = list(range(start, end))
        if not maskable:
            return []
        num = max(1, int(len(maskable) * prob))
        pool = min(int(num * self.tail_bias_factor), len(maskable))
        pool = max(pool, num)
        candidates = maskable[-pool:]
        return random.sample(candidates, min(num, len(candidates)))
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(examples)
        
        all_ids = []
        prompt_lengths = []
        for ex in examples:
            ids, plen = self._tokenize_seq2seq(ex)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            prompt_lengths.append(plen)
        
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        t = self.sample_antithetic_probs(B, low=p_min, high=p_max)
        batch_probs_tensor = (1.0 - 1e-5) * t
        
        batch_inputs = []
        batch_labels = []
        
        for idx in range(B):
            ids = all_ids[idx].tolist()
            prob = batch_probs_tensor[idx].item()
            plen = prompt_lengths[idx]
            seq_len = len(ids)
            
            response_start = plen
            response_end = seq_len - 1 if seq_len > plen else seq_len
            
            mask_indices = self._tail_bias_indices_in_range(response_start, response_end, prob)
            
            inputs = ids[:]
            for i in mask_indices:
                inputs[i] = self.mask_token_id
            
            labels = [-100] * plen + ids[plen:]
            
            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
        
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


class ReasoningARCollator(BaseCollator):
    """
    Standard causal LM collator for seq2seq reasoning tasks.
    """
    
    def __init__(self, tokenizer, max_length: int = 256, text_key: str = "text",
                 pad_to_max_length: bool = True, **kwargs):
        super().__init__(tokenizer, max_length, text_key,
                        add_special_tokens=True, pad_to_max_length=pad_to_max_length)
    
    def _tokenize_seq2seq(self, example):
        return _tokenize_seq2seq_common(example, self.text_key, self.tokenizer, self.max_length)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_inputs = []
        batch_labels = []
        
        for ex in examples:
            ids, plen = self._tokenize_seq2seq(ex)
            batch_inputs.append(torch.tensor(ids, dtype=torch.long))
            labels = [-100] * plen + ids[plen:]
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
        
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ReasoningMDMCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    MDM (Multi-granularity Diffusion Modeling) collator for seq2seq reasoning tasks.
    Matches Ye et al. (2024) "Beyond Autoregression" EXACTLY.
    
    Key differences from vanilla MDLM:
    - Discrete timesteps t ∈ {0, ..., T-1} (not continuous)
    - Masking: each maskable position masked with prob (t+1)/T
    - Time reweighting: 'linear' = (T - t) weight, not 1/t DAUM
    - Token reweighting: focal loss = alpha * (1 - exp(-loss))^gamma * loss
    - Loss only on [MASK] positions (not all positions like MDLM SUBS)
    """
    
    def __init__(self, tokenizer, max_length: int = 256, text_key: str = "text",
                 pad_to_max_length: bool = True,
                 diffusion_steps: int = 20,
                 time_reweighting: str = 'linear',
                 token_reweighting: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 **kwargs):
        super().__init__(tokenizer, max_length, text_key,
                        add_special_tokens=True, pad_to_max_length=pad_to_max_length)
        self.diffusion_steps = diffusion_steps
        self.time_reweighting = time_reweighting
        self.token_reweighting = token_reweighting
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def _tokenize_seq2seq(self, example):
        return _tokenize_seq2seq_common(example, self.text_key, self.tokenizer, self.max_length)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        MDM collator matching Ye et al. exactly:
        - Discrete t ∈ {0, ..., T-1}
        - Mask prob per position = (t+1)/T
        - Only response positions are maskable
        - Passes discrete_t and loss_mask for MDM-specific loss computation
        """
        B = len(examples)
        
        all_ids = []
        prompt_lengths = []
        for ex in examples:
            ids, plen = self._tokenize_seq2seq(ex)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            prompt_lengths.append(plen)
        
        # Sample discrete timestep t ∈ {0, ..., T-1}
        T = self.diffusion_steps
        t = torch.randint(0, T, (B,))
        
        batch_inputs = []
        batch_labels = []
        batch_loss_masks = []
        
        for idx in range(B):
            ids = all_ids[idx].tolist()
            t_val = t[idx].item()
            plen = prompt_lengths[idx]
            seq_len = len(ids)
            
            inputs = ids[:]
            labels = ids[:]
            loss_mask = [False] * seq_len
            
            # Mask response tokens with prob (t+1)/T
            # This includes EOS — model must learn to predict it
            mask_prob = (t_val + 1) / T
            for i in range(plen, seq_len):
                if random.random() < mask_prob:
                    inputs[i] = self.mask_token_id
                    loss_mask[i] = True
            
            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.bool))
        
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        loss_mask = self._pad_batch(
            [m.long() for m in batch_loss_masks], padding_value=0
        ).bool()
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # MDM-specific fields for loss computation in model forward
            "discrete_t": t,  # [B], int in {0,...,T-1}
            "loss_mask": loss_mask,  # [B, L], True at [MASK] positions
            "mdm_config": {
                "diffusion_steps": T,
                "time_reweighting": self.time_reweighting,
                "token_reweighting": self.token_reweighting,
                "focal_alpha": self.focal_alpha,
                "focal_gamma": self.focal_gamma,
            },
        }


class ReasoningMDLMCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    MDLM collator for seq2seq reasoning tasks (standard DAUM 1/t weighting).
    
    - Prompt tokens: kept clean, included in bidirectional context
    - Response tokens: randomly masked with probability t
    - Loss on all response tokens (MDLM SUBS handles unmasked → 0 loss)
    """
    
    def __init__(self, tokenizer, max_length: int = 256, text_key: str = "text",
                 pad_to_max_length: bool = True, **kwargs):
        super().__init__(tokenizer, max_length, text_key,
                        add_special_tokens=True, pad_to_max_length=pad_to_max_length)
    
    def _tokenize_seq2seq(self, example):
        return _tokenize_seq2seq_common(example, self.text_key, self.tokenizer, self.max_length)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(examples)
        
        all_ids = []
        prompt_lengths = []
        for ex in examples:
            ids, plen = self._tokenize_seq2seq(ex)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            prompt_lengths.append(plen)
        
        # Sample continuous t
        t = self.sample_antithetic_probs(B, low=0.001, high=0.999)
        probs = (1.0 - 1e-5) * t
        
        batch_inputs = []
        batch_labels = []
        
        for idx in range(B):
            ids = all_ids[idx].tolist()
            prob = probs[idx].item()
            plen = prompt_lengths[idx]
            seq_len = len(ids)
            
            inputs = ids[:]
            labels = ids[:]
            
            # Mask response tokens INCLUDING EOS
            for i in range(plen, seq_len):
                if random.random() < prob:
                    inputs[i] = self.mask_token_id
            
            # NOTE: Do NOT set prompt labels to -100 here!
            # Prompt tokens are never masked, so _apply_subs forces their logits
            # to one-hot → loss=0 automatically. Setting labels to -100 would
            # shrink num_items_in_batch denominator, inflating the loss.
            
            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
        
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": probs,
            "zero_mask_prob": True,
        }
