# eval/data/eval_collators.py
"""
Evaluation-specific collators for Monte Carlo estimation of diffusion models.

These collators apply noise to sequences following the same logic as training collators,
enabling proper MC estimation using the model's forward pass.

IMPORTANT: Noise is ONLY applied to continuation tokens, not prompt tokens.
The prompt serves as conditioning context and should remain unmasked.
"""

from typing import List, Dict, Any, Tuple, Optional
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from ..config import DiffusionType


class BaseMCCollator:
    """
    Base class for MC evaluation collators.
    
    Handles common operations like padding, special token masking, and
    antithetic sampling of noise levels.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mc_num: int = 128,
        mc_batch_size: int = 16,
    ):
        self.tokenizer = tokenizer
        self.mc_num = mc_num
        self.mc_batch_size = mc_batch_size
        
        # Token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have mask_token_id for diffusion evaluation")
        
        # Build special tokens set for masking protection
        self.special_token_ids = {
            tok_id for tok_id in [
                self.pad_token_id, self.bos_token_id, self.eos_token_id,
                tokenizer.cls_token_id, tokenizer.sep_token_id, self.mask_token_id
            ] if tok_id is not None
        }
    
    def build_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create boolean mask where True = special token (should not be masked)."""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            mask = mask | (input_ids == token_id)
        return mask
    
    def build_non_maskable_mask(
        self, 
        input_ids: torch.Tensor, 
        continuation_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create boolean mask where True = should NOT be masked.
        
        This combines:
        1. Special tokens (BOS, EOS, PAD, etc.) - should never be masked
        2. Prompt tokens (continuation_mask == False) - should never be masked
        
        Only continuation tokens that are not special can be masked.
        """
        # Ensure continuation_mask is on the same device as input_ids
        continuation_mask = continuation_mask.to(input_ids.device)
        special_mask = self.build_special_tokens_mask(input_ids)
        prompt_mask = ~continuation_mask  # True for prompt tokens
        return special_mask | prompt_mask
    
    def sample_antithetic_probs(
        self, 
        batch_size: int, 
        low: float = 0.001, 
        high: float = 0.999
    ) -> torch.Tensor:
        """
        Generate stratified/antithetic sampling of noise probabilities.
        
        Divides [low, high] into batch_size intervals and samples one point per interval,
        then shuffles. This reduces variance in MC estimation.
        """
        steps = torch.arange(batch_size, dtype=torch.float32)
        noise = torch.rand(batch_size)
        probs_01 = (steps + noise) / batch_size
        probs = low + probs_01 * (high - low)
        shuffled_indices = torch.randperm(batch_size)
        return probs[shuffled_indices]
    
    def _pad_sequences(
        self, 
        sequences: List[torch.Tensor], 
        padding_value: int
    ) -> torch.Tensor:
        """Pad list of tensors to same length."""
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


class CausalMCCollator(BaseMCCollator):
    """
    MC collator for Causal MLM style diffusion (tail-biased masking).
    
    Masks tokens with higher probability towards the end of the sequence,
    mimicking autoregressive generation patterns.
    
    NOTE: Only continuation tokens are masked; prompt tokens remain unchanged.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mc_num: int = 128,
        mc_batch_size: int = 16,
        tail_bias_factor: float = 1.5,
        start_prob: float = 0.0,
        end_prob: float = 1.0,
    ):
        super().__init__(tokenizer, mc_num, mc_batch_size)
        self.tail_bias_factor = max(1.0, tail_bias_factor)
        self.start_prob = start_prob
        self.end_prob = end_prob
    
    def _tail_bias_indices(
        self, 
        seq_len: int, 
        prob: float,
        non_maskable_mask: torch.Tensor
    ) -> List[int]:
        """
        Select indices to mask with tail bias.
        
        Focuses masking on later positions in the sequence.
        Only considers positions where non_maskable_mask is False.
        """
        # Get maskable positions (excluding special tokens AND prompt tokens)
        maskable = [i for i in range(seq_len) if not non_maskable_mask[i].item()]
        if not maskable:
            return []
        
        num = max(1, int(len(maskable) * prob))
        pool = min(int(num * self.tail_bias_factor), len(maskable))
        pool = max(pool, num)
        
        # Focus on later positions within the maskable set
        candidates = maskable[-pool:]
        return random.sample(candidates, min(num, len(candidates)))
    
    def create_mc_samples(
        self,
        input_ids: torch.Tensor,
        continuation_mask: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create MC samples with tail-biased masking.
        
        Args:
            input_ids: (seq_len,) token IDs
            continuation_mask: (seq_len,) boolean mask for continuation tokens
            device: Target device
            
        Returns:
            List of batch dictionaries for MC estimation
        """
        seq_len = input_ids.size(0)
        
        # Build mask: True = should NOT be masked (special tokens OR prompt tokens)
        non_maskable_mask = self.build_non_maskable_mask(input_ids, continuation_mask)
        
        # Sample noise levels
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        
        batches = []
        samples_generated = 0
        
        while samples_generated < self.mc_num:
            current_batch_size = min(self.mc_batch_size, self.mc_num - samples_generated)
            probs = self.sample_antithetic_probs(current_batch_size, p_min, p_max)
            
            batch_input_ids = []
            batch_labels = []
            batch_probs = []
            
            for i in range(current_batch_size):
                prob = probs[i].item()
                indices_to_mask = self._tail_bias_indices(seq_len, prob, non_maskable_mask)
                
                # Create masked input
                masked_ids = input_ids.clone()
                for idx in indices_to_mask:
                    masked_ids[idx] = self.mask_token_id
                
                batch_input_ids.append(masked_ids)
                batch_labels.append(input_ids.clone())
                batch_probs.append(probs[i])
            
            # Stack into batch
            stacked_input_ids = torch.stack(batch_input_ids).to(device)
            stacked_labels = torch.stack(batch_labels).to(device)
            stacked_labels[stacked_labels == self.pad_token_id] = -100
            attention_mask = (stacked_input_ids != self.pad_token_id).long()
            
            # Calculate num_items_in_batch (number of non-ignored tokens)
            num_items_in_batch = (stacked_labels != -100).sum()
            
            batches.append({
                'input_ids': stacked_input_ids,
                'attention_mask': attention_mask,
                'labels': stacked_labels,
                'current_mlm_prob': torch.stack(batch_probs).to(device),
                'continuation_mask': continuation_mask.unsqueeze(0).expand(current_batch_size, -1).to(device),
                'causal': True,
                'use_daum': False,
                'num_items_in_batch': num_items_in_batch,
            })
            
            samples_generated += current_batch_size
        
        return batches


class MDLMMCCollator(BaseMCCollator):
    """
    MC collator for MDLM (Masked Diffusion Language Model).
    
    Applies uniform random masking with probability t sampled from [0, 1].
    
    NOTE: Only continuation tokens are masked; prompt tokens remain unchanged.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mc_num: int = 128,
        mc_batch_size: int = 16,
        eps: float = 1e-5,
    ):
        super().__init__(tokenizer, mc_num, mc_batch_size)
        self.eps = eps
    
    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        mlm_probability: torch.Tensor,
        non_maskable_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking with given probability per sample.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            mlm_probability: (batch,) masking probability per sample
            non_maskable_mask: (batch, seq_len) boolean mask where True = should NOT be masked
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        prob_matrix = mlm_probability.unsqueeze(1).expand_as(input_ids)
        masked_indices = torch.rand_like(input_ids, dtype=torch.float) < prob_matrix
        
        # Only mask where non_maskable_mask is False (i.e., continuation tokens that aren't special)
        masked_indices = masked_indices & (~non_maskable_mask)
        
        # Ensure at least one token is masked per sample (within maskable region)
        maskable_mask = ~non_maskable_mask
        for i in range(input_ids.size(0)):
            if not masked_indices[i].any():
                candidates = maskable_mask[i].nonzero(as_tuple=True)[0]
                if len(candidates) > 0:
                    chosen = candidates[torch.randint(len(candidates), (1,))]
                    masked_indices[i, chosen] = True
        
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        inputs = input_ids.clone()
        inputs[masked_indices] = self.mask_token_id
        
        return inputs, labels
    
    def create_mc_samples(
        self,
        input_ids: torch.Tensor,
        continuation_mask: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create MC samples with uniform random masking.
        
        Only continuation tokens are masked.
        """
        seq_len = input_ids.size(0)
        
        # Build mask: True = should NOT be masked (special tokens OR prompt tokens)
        non_maskable_mask = self.build_non_maskable_mask(input_ids, continuation_mask)
        
        batches = []
        samples_generated = 0
        
        while samples_generated < self.mc_num:
            current_batch_size = min(self.mc_batch_size, self.mc_num - samples_generated)
            
            # Expand input for batch processing
            batch_input_ids = input_ids.unsqueeze(0).expand(current_batch_size, -1).clone()
            batch_non_maskable = non_maskable_mask.unsqueeze(0).expand(current_batch_size, -1)
            
            # Sample noise levels with antithetic sampling
            probs = self.sample_antithetic_probs(current_batch_size, 0.001, 0.999)
            probs = (1.0 - self.eps) * probs
            
            # Apply masking (only to continuation tokens)
            masked_ids, labels = self.mask_tokens(
                batch_input_ids.to(device),
                probs.to(device),
                batch_non_maskable.to(device),
            )
            
            attention_mask = (masked_ids != self.pad_token_id).long()
            
            # Calculate num_items_in_batch (number of non-ignored tokens)
            num_items_in_batch = (labels != -100).sum()
            
            batches.append({
                'input_ids': masked_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'current_mlm_prob': probs.to(device),
                'continuation_mask': continuation_mask.unsqueeze(0).expand(current_batch_size, -1).to(device),
                'zero_mask_prob': True,
                'num_items_in_batch': num_items_in_batch,
            })
            
            samples_generated += current_batch_size
        
        return batches


class BD3LMMCCollator(BaseMCCollator):
    """
    MC collator for BD3LM (Block Denoising Discrete Diffusion LM).
    
    Applies block-structured masking with time-varying noise levels.
    
    NOTE: Only continuation tokens are masked; prompt tokens remain unchanged.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mc_num: int = 128,
        mc_batch_size: int = 16,
        block_size: int = 16,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
        resample: bool = True,
    ):
        super().__init__(tokenizer, mc_num, mc_batch_size)
        self.block_size = block_size
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.resample = resample
    
    def _sample_block_t(
        self, 
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample block-wise timesteps for BD3LM.
        
        Each block gets a different timestep, creating a structured noise pattern.
        """
        B, L = input_ids.shape
        num_blocks = (L + self.block_size - 1) // self.block_size
        total_units = B * num_blocks
        
        offset = torch.arange(total_units, device=input_ids.device) / total_units
        offset = offset.view(B, num_blocks)
        
        rand_noise = torch.rand((B, num_blocks), device=input_ids.device)
        t_blocks = (rand_noise / total_units + offset) % 1.0
        
        # Expand to sequence length
        t_expanded = t_blocks.repeat_interleave(self.block_size, dim=-1)
        t_expanded = t_expanded[:, :L]
        
        # Scale to sampling range
        t_final = t_expanded * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
        
        return t_final, num_blocks
    
    def _resample_mask(
        self,
        input_ids: torch.Tensor,
        xt: torch.Tensor,
        t_map: torch.Tensor,
        non_maskable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Resample masks to ensure proper coverage per block."""
        if not self.resample:
            return xt
        
        B, L = input_ids.shape
        pad_len = (self.block_size - (L % self.block_size)) % self.block_size
        
        def to_blocks(x, pad_val=0):
            if pad_len > 0:
                x = F.pad(x, (0, pad_len), value=pad_val)
            return x.view(B, -1, self.block_size)
        
        def from_blocks(x_blocked):
            return x_blocked.view(B, -1)[:, :L]
        
        t_blocks = to_blocks(t_map)[:, :, 0]
        
        for _ in range(5):
            xt_blocked = to_blocks(xt, pad_val=self.pad_token_id)
            is_masked = (xt_blocked == self.mask_token_id).float()
            perc_masked = is_masked.sum(dim=-1) / self.block_size
            
            too_low = perc_masked < self.sampling_eps_min
            too_high = perc_masked > self.sampling_eps_max
            
            if self.sampling_eps_min <= 1e-3:
                too_low = torch.zeros_like(too_low, dtype=torch.bool)
            if self.sampling_eps_max >= 1.0:
                too_high = torch.zeros_like(too_high, dtype=torch.bool)
            
            needs_regen = too_low | too_high
            if not needs_regen.any():
                break
            
            current_p = t_blocks
            current_p_expanded = current_p.unsqueeze(-1).expand_as(xt_blocked)
            new_decisions = torch.rand_like(current_p_expanded) < current_p_expanded
            
            mask_needs_update = needs_regen.unsqueeze(-1).expand_as(xt_blocked)
            input_blocked = to_blocks(input_ids, pad_val=self.pad_token_id)
            non_maskable_blocked = to_blocks(non_maskable_mask, pad_val=1)  # pad with 1 = non-maskable
            
            new_xt_blocked = torch.where(
                new_decisions,
                torch.full_like(xt_blocked, self.mask_token_id),
                input_blocked
            )
            # Only update where: needs update AND is maskable (continuation, not special)
            update_locs = mask_needs_update & (~non_maskable_blocked)
            xt_blocked = torch.where(update_locs, new_xt_blocked, xt_blocked)
            xt = from_blocks(xt_blocked)
        
        return xt
    
    def create_mc_samples(
        self,
        input_ids: torch.Tensor,
        continuation_mask: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create MC samples with block-structured masking.
        
        Only continuation tokens are masked.
        """
        seq_len = input_ids.size(0)
        
        batches = []
        samples_generated = 0
        
        while samples_generated < self.mc_num:
            current_batch_size = min(self.mc_batch_size, self.mc_num - samples_generated)
            
            # Expand input for batch
            batch_input_ids = input_ids.unsqueeze(0).expand(current_batch_size, -1).clone().to(device)
            batch_continuation_mask = continuation_mask.unsqueeze(0).expand(current_batch_size, -1).to(device)
            
            # Build non-maskable mask: True = should NOT be masked (special tokens OR prompt tokens)
            non_maskable_mask = self.build_non_maskable_mask(batch_input_ids[0], continuation_mask).to(device)
            non_maskable_mask = non_maskable_mask.unsqueeze(0).expand(current_batch_size, -1)
            
            # Also include padding in non-maskable
            non_maskable_mask = non_maskable_mask | (batch_input_ids == self.pad_token_id)
            
            # Sample block-wise timesteps
            t_map, _ = self._sample_block_t(batch_input_ids)
            
            # Generate initial mask - only where maskable (continuation tokens, not special)
            rand_matrix = torch.rand_like(t_map)
            mask_decision = rand_matrix < t_map
            mask_decision = mask_decision & (~non_maskable_mask)  # Only mask continuation tokens
            
            xt = batch_input_ids.clone()
            xt[mask_decision] = self.mask_token_id
            
            # Resample if needed
            if self.resample:
                xt = self._resample_mask(batch_input_ids, xt, t_map, non_maskable_mask)
            
            # Prepare labels
            labels = batch_input_ids.clone()
            labels[batch_input_ids == self.pad_token_id] = -100
            
            attention_mask = (xt != self.pad_token_id).long()
            
            # Calculate num_items_in_batch (number of non-ignored tokens)
            num_items_in_batch = (labels != -100).sum()
            
            batches.append({
                'input_ids': xt,
                'attention_mask': attention_mask,
                'labels': labels,
                'cond_ids': batch_input_ids,
                'timesteps': t_map,
                'continuation_mask': batch_continuation_mask,
                'causal': False,
                'num_items_in_batch': num_items_in_batch,
            })
            
            samples_generated += current_batch_size
        
        return batches


class PrefixLMMCCollator(BaseMCCollator):
    """
    MC collator for Prefix LM: masks the TAIL of continuation tokens.
    
    This matches the PrefixLM training setup where a random suffix ratio t
    determines how many tokens from the tail are masked. The prefix (early tokens)
    remains clean and visible, while the suffix (last t fraction) is masked.
    
    For MC estimation, we sample different t values and mask the last t fraction
    of continuation tokens, weighted by 1/t for proper ELBO estimation.
    
    IMPORTANT (label convention):
    Labels keep ALL non-pad tokens as real token IDs (including prefix and special
    tokens like BOS/EOS). The model's _apply_subs forces unmasked positions' logits
    to one-hot of the correct token, so their CE loss is automatically 0.
    This ensures num_items_in_batch counts ALL content tokens, keeping the loss
    scale consistent with MDLMMCCollator and the fixed training PrefixLMCollator.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mc_num: int = 128,
        mc_batch_size: int = 16,
        eps: float = 1e-5,
    ):
        super().__init__(tokenizer, mc_num, mc_batch_size)
        self.eps = eps
    
    def create_mc_samples(
        self,
        input_ids: torch.Tensor,
        continuation_mask: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create MC samples with tail-based suffix masking (matching PrefixLM training).
        
        For each MC sample:
        1. Sample suffix ratio t from [0.001, 0.999]
        2. Mask the LAST t fraction of continuation tokens (tail masking)
        3. Keep the first (1-t) fraction of continuation tokens clean
        4. Weight loss by 1/t for ELBO estimation (via current_mlm_prob)
        5. Labels keep ALL non-pad tokens as real IDs (_apply_subs handles the rest)
        
        This exactly mirrors the fixed PrefixLMCollator training behavior.
        """
        seq_len = input_ids.size(0)
        continuation_mask = continuation_mask.to(device)
        
        # Build special tokens mask to exclude BOS, EOS, PAD from masking
        # (matching training where special_tokens_mask protects these)
        special_mask = self.build_special_tokens_mask(input_ids.to(device))
        
        # Get MASKABLE continuation positions: continuation=True AND not special
        maskable_cont = continuation_mask & (~special_mask)
        cont_positions = maskable_cont.nonzero(as_tuple=True)[0]
        n_cont = cont_positions.size(0)
        
        if n_cont == 0:
            return []
        
        batches = []
        samples_generated = 0
        
        while samples_generated < self.mc_num:
            current_batch_size = min(self.mc_batch_size, self.mc_num - samples_generated)
            
            # Sample suffix ratios with antithetic sampling (matching training)
            t_values = self.sample_antithetic_probs(current_batch_size, 0.001, 0.999)
            suffix_ratios = (1.0 - self.eps) * t_values
            
            batch_input_ids = []
            batch_labels = []
            
            for i in range(current_batch_size):
                t = suffix_ratios[i].item()
                
                # Number of continuation tokens to mask (from the tail)
                n_suffix = max(1, int(n_cont * t))
                
                # Suffix = last n_suffix continuation positions (masked)
                suffix_positions = cont_positions[-n_suffix:]
                
                # Create masked input: mask only suffix positions
                masked_ids = input_ids.clone().to(device)
                masked_ids[suffix_positions] = self.mask_token_id
                
                # Create labels: keep ALL original token IDs (matching training convention)
                # Do NOT set prefix labels to -100!
                # _apply_subs forces unmasked positions' logits to one-hot → loss=0 automatically
                # This ensures num_items_in_batch includes ALL content tokens
                labels = input_ids.clone().to(device)
                # Only PAD tokens get -100 (same as MDLMMCCollator and training collator)
                labels[labels == self.pad_token_id] = -100
                
                batch_input_ids.append(masked_ids)
                batch_labels.append(labels)
            
            stacked_input_ids = torch.stack(batch_input_ids)
            stacked_labels = torch.stack(batch_labels)
            attention_mask = (stacked_input_ids != self.pad_token_id).long()
            
            # num_items_in_batch counts ALL non-pad tokens (consistent with MDLM)
            num_items_in_batch = (stacked_labels != -100).sum()
            
            batches.append({
                'input_ids': stacked_input_ids,
                'attention_mask': attention_mask,
                'labels': stacked_labels,
                'current_mlm_prob': suffix_ratios.to(device),
                'continuation_mask': continuation_mask.unsqueeze(0).expand(current_batch_size, -1).to(device),
                'num_items_in_batch': num_items_in_batch,
            })
            
            samples_generated += current_batch_size
        
        return batches


def get_mc_collator(
    diffusion_type: DiffusionType,
    tokenizer: PreTrainedTokenizer,
    mc_num: int = 128,
    mc_batch_size: int = 16,
    **kwargs,
) -> BaseMCCollator:
    """
    Factory function to create the appropriate MC collator.
    
    Args:
        diffusion_type: Type of diffusion model (causal, mdlm, bd3lm)
        tokenizer: Tokenizer with mask_token_id
        mc_num: Number of MC samples
        mc_batch_size: Batch size for MC sampling
        **kwargs: Additional arguments for specific collators
            - For BD3LM: block_size, sampling_eps_min, sampling_eps_max, resample
            - For Causal: tail_bias_factor, start_prob, end_prob
            - For MDLM: eps
        
    Returns:
        Appropriate MC collator instance
    """
    if diffusion_type == DiffusionType.CAUSAL:
        # Filter kwargs to only include parameters accepted by CausalMCCollator
        causal_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['tail_bias_factor', 'start_prob', 'end_prob']
        }
        return CausalMCCollator(tokenizer, mc_num, mc_batch_size, **causal_kwargs)
    
    elif diffusion_type == DiffusionType.MDLM:
        # Filter kwargs to only include parameters accepted by MDLMMCCollator
        mdlm_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['eps']
        }
        return MDLMMCCollator(tokenizer, mc_num, mc_batch_size, **mdlm_kwargs)
    
    elif diffusion_type == DiffusionType.BD3LM:
        # Filter kwargs to only include parameters accepted by BD3LMMCCollator
        bd3lm_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['block_size', 'sampling_eps_min', 'sampling_eps_max', 'resample']
        }
        return BD3LMMCCollator(tokenizer, mc_num, mc_batch_size, **bd3lm_kwargs)
    
    elif diffusion_type == DiffusionType.PREFIXLM:
        prefixlm_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['eps']
        }
        return PrefixLMMCCollator(tokenizer, mc_num, mc_batch_size, **prefixlm_kwargs)
    
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")