# eval/models/diffusion.py
"""
Adapter for discrete diffusion language models.

Supports three diffusion architectures:
- Causal: Tail-biased masking (CausalMLM style)
- MDLM: Uniform random masking
- BD3LM: Block-structured masking

Evaluation modes:
- Monte Carlo (MC): Uses model's forward pass with proper noise sampling
- Pseudo-Log-Likelihood (PLL): Masks one token at a time

The MC implementation uses the same masking logic as training collators,
and relies on the model's forward pass to compute properly weighted losses.
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .base import ModelAdapter
from ..config import ModelConfig, DiffusionEvalMode, DiffusionType
from ..data.eval_collators import get_mc_collator


# Disable torch.compile/dynamo for evaluation to avoid issues with dynamic shapes
# This is especially important for BD3LM which uses flex_attention
def _disable_dynamo():
    """Disable torch dynamo to avoid compilation errors with dynamic shapes."""
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        # Optionally disable completely
        torch._dynamo.disable()
        print("[INFO] Disabled torch.dynamo for evaluation")
    except (ImportError, AttributeError):
        pass  # torch._dynamo not available in older versions


class DiffusionModelAdapter(ModelAdapter):
    """
    Adapter for discrete diffusion language models.
    
    Supports MC and PLL evaluation modes, with three diffusion variants.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mc_collator = None
    
    @property
    def mc_collator(self):
        """Lazy initialization of MC collator."""
        if self._mc_collator is None:
            # Build kwargs based on diffusion type
            collator_kwargs = {}
            
            # Only pass block_size for BD3LM
            if self.config.diffusion_type == DiffusionType.BD3LM:
                block_size = getattr(self.config, 'block_size', 16)
                collator_kwargs['block_size'] = block_size
            
            self._mc_collator = get_mc_collator(
                diffusion_type=self.config.diffusion_type,
                tokenizer=self.tokenizer,
                mc_num=self.config.mc_num,
                mc_batch_size=self.config.mc_batch_size,
                **collator_kwargs,
            )
        return self._mc_collator
    
    def compute_logprobs(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute log-probabilities for continuation tokens.
        
        Dispatches to the appropriate method based on eval_mode.
        """
        eval_mode = self.config.diffusion_eval_mode
        
        if eval_mode == DiffusionEvalMode.MONTE_CARLO:
            return self._compute_mc(batch)
        else:
            return self._compute_pll(batch)
    
    def _compute_mc(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute log-probs using Monte Carlo estimation.
        
        Uses the model's forward pass with proper noise sampling from collators.
        The model's loss computation handles the ELBO weighting automatically.
        """
        input_ids = batch['input_ids']
        continuation_mask = batch['continuation_mask']
        
        batch_size = input_ids.shape[0]
        results = [0.0] * batch_size
        
        pad_token_id = self.tokenizer.pad_token_id
        
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(batch_size):
                sample_ids = input_ids[i]
                sample_cont_mask = continuation_mask[i]
                
                # Get actual sequence length (excluding padding)
                seq_len = (
                    (sample_ids != pad_token_id).sum().item()
                    if pad_token_id is not None
                    else len(sample_ids)
                )
                
                # Extract unpadded sequence
                seq = sample_ids[:seq_len]
                cont_mask_unpadded = sample_cont_mask[:seq_len]
                
                target_len = cont_mask_unpadded.sum().item()
                if target_len == 0:
                    continue
                
                # Create MC samples using collator
                mc_batches = self.mc_collator.create_mc_samples(
                    input_ids=seq,
                    continuation_mask=cont_mask_unpadded,
                    device=self.device,
                )
                
                # Compute losses using model's forward pass
                total_loss = 0.0
                num_samples = 0
                
                for mc_batch in mc_batches:
                    loss = self._forward_loss(mc_batch, cont_mask_unpadded)
                    batch_samples = mc_batch['input_ids'].size(0)
                    total_loss += loss * batch_samples
                    num_samples += batch_samples
                
                # Average loss across MC samples -> negative log-prob
                avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
                results[i] = -avg_loss  # Convert loss to log-prob
        
        return results
    
    def _forward_loss(
        self,
        mc_batch: Dict[str, torch.Tensor],
        continuation_mask: torch.Tensor,
    ) -> float:
        """
        Compute loss using model's forward pass.
        
        The model should implement proper ELBO/loss computation that accounts
        for the noise level (current_mlm_prob, timesteps, etc.).
        
        Args:
            mc_batch: Batch from MC collator with masked inputs and metadata
            continuation_mask: Mask indicating continuation tokens
            
        Returns:
            Average loss over continuation tokens
        """
        # Prepare model inputs
        model_inputs = {
            'input_ids': mc_batch['input_ids'],
            'attention_mask': mc_batch['attention_mask'],
        }
        
        # Add labels for loss computation
        if 'labels' in mc_batch:
            model_inputs['labels'] = mc_batch['labels']
        
        # Add num_items_in_batch - required by diffusion models for loss calculation
        if 'num_items_in_batch' in mc_batch:
            model_inputs['num_items_in_batch'] = mc_batch['num_items_in_batch']
        else:
            # Calculate it if not provided by the collator
            labels = mc_batch.get('labels')
            if labels is not None:
                num_items_in_batch = (labels != -100).sum()
                model_inputs['num_items_in_batch'] = num_items_in_batch
        
        # Add diffusion-specific parameters if model expects them
        optional_keys = [
            'current_mlm_prob', 'timesteps', 'cond_ids',
            'causal', 'use_daum', 'zero_mask_prob'
        ]
        for key in optional_keys:
            if key in mc_batch:
                model_inputs[key] = mc_batch[key]
        
        # Forward pass - wrapped to handle potential compilation issues
        try:
            outputs = self.model(**model_inputs)
        except Exception as e:
            # If forward fails (e.g., due to dynamo issues), try with dynamo disabled
            if "dynamo" in str(e).lower() or "inductor" in str(e).lower() or "Symbol" in str(e):
                print(f"[WARNING] Dynamo/Inductor error detected, retrying with compilation disabled...")
                _disable_dynamo()
                # Reset model to non-compiled version if possible
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                outputs = self.model(**model_inputs)
            else:
                raise
        
        # If model returns loss directly, use it
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss.mean().item()
        
        # Otherwise compute loss from logits
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        labels = mc_batch['labels']
        
        # Compute cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(logits_flat, labels_flat)
        per_token_loss = per_token_loss.view(batch_size, seq_len)
        
        # Apply continuation mask - only count loss on continuation tokens
        cont_mask_expanded = continuation_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Handle potential length mismatch
        min_len = min(per_token_loss.size(1), cont_mask_expanded.size(1))
        per_token_loss = per_token_loss[:, :min_len]
        cont_mask_expanded = cont_mask_expanded[:, :min_len].float()
        
        masked_loss = per_token_loss * cont_mask_expanded
        
        # Average over continuation tokens
        num_cont_tokens = cont_mask_expanded.sum(dim=1).clamp(min=1)
        sample_losses = masked_loss.sum(dim=1) / num_cont_tokens
        
        return sample_losses.mean().item()
    
    def _compute_pll(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute Pseudo-Log-Likelihood by masking one token at a time.
        
        PLL(x) = sum_i log P(x_i | x_{-i})
        
        This is more expensive but provides a different estimate.
        """
        input_ids = batch['input_ids']
        continuation_mask = batch['continuation_mask']
        
        mask_token_id = self.tokenizer.mask_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        batch_size = input_ids.shape[0]
        results = [0.0] * batch_size
        
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(batch_size):
                sample_ids = input_ids[i]
                sample_cont_mask = continuation_mask[i]
                
                seq_len = (
                    (sample_ids != pad_token_id).sum().item()
                    if pad_token_id is not None
                    else len(sample_ids)
                )
                
                seq = sample_ids[:seq_len].to(self.device)
                cont_mask_unpadded = sample_cont_mask[:seq_len]
                prompt_len = (~cont_mask_unpadded).sum().item()
                target_len = cont_mask_unpadded.sum().item()
                
                if target_len == 0:
                    continue
                
                results[i] = self._pll_single(seq, prompt_len, target_len, mask_token_id)
        
        return results
    
    def _pll_single(
        self,
        seq: torch.Tensor,
        prompt_len: int,
        target_len: int,
        mask_token_id: int,
    ) -> float:
        """Compute PLL for a single sequence by masking one token at a time."""
        # Create batch with all masked positions
        seq_batch = seq.unsqueeze(0).expand(target_len, -1).clone()
        
        # Vectorized mask creation
        target_indices = torch.arange(target_len, device=self.device)
        mask_positions = prompt_len + target_indices
        
        # Apply masks using advanced indexing
        seq_batch[target_indices, mask_positions] = mask_token_id
        
        # Forward pass - handle potential compilation issues
        try:
            outputs = self.model(seq_batch)
        except Exception as e:
            if "dynamo" in str(e).lower() or "inductor" in str(e).lower() or "Symbol" in str(e):
                print(f"[WARNING] Dynamo/Inductor error in PLL, retrying with compilation disabled...")
                _disable_dynamo()
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                outputs = self.model(seq_batch)
            else:
                raise
        
        logits_batch = outputs.logits
        
        # Extract logits at masked positions and compute log probs
        logits_at_masked = logits_batch[target_indices, mask_positions]
        original_tokens = seq[mask_positions]
        
        log_probs = F.log_softmax(logits_at_masked, dim=-1)
        token_log_probs = log_probs.gather(1, original_tokens.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.sum().item()
    
    @classmethod
    def load(cls, config: ModelConfig) -> "DiffusionModelAdapter":
        """Load a discrete diffusion language model."""
        # Disable dynamo BEFORE loading to prevent compilation during model init
        _disable_dynamo()
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        
        # Ensure mask token exists
        if tokenizer.mask_token_id is None:
            raise ValueError(
                f"Tokenizer for {config.name_or_path} does not have mask_token_id. "
                "This is required for diffusion model evaluation."
            )
        
        dtype = getattr(torch, config.torch_dtype)
        
        # Load model with torch.compile disabled
        model = AutoModel.from_pretrained(
            config.name_or_path,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
        
        # If model was compiled, try to get the original
        if hasattr(model, '_orig_mod'):
            print("[INFO] Unwrapping compiled model for evaluation")
            model = model._orig_mod
        
        print(f"[INFO] Loaded diffusion model: {config.name_or_path}")
        print(f"[INFO] Diffusion type: {config.diffusion_type.value}")
        print(f"[INFO] Eval mode: {config.diffusion_eval_mode.value}")
        
        adapter = cls(model, tokenizer, config)
        return adapter.to(config.device).eval()