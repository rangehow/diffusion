# eval/models/causal.py
"""
Adapter for causal (autoregressive) language models.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from .base import ModelAdapter
from ..config import ModelConfig


class CausalModelAdapter(ModelAdapter):
    """
    Adapter for causal language models (GPT-style).
    
    Computes log-probabilities by extracting the probability of each token
    given all previous tokens in the sequence.
    """
    
    def compute_logprobs(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute log-probabilities for continuation tokens using causal LM.
        
        For each position, we use the logits from the previous position
        to compute P(token_t | token_1, ..., token_{t-1}).
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        continuation_mask = batch['continuation_mask'].to(self.device)
        
        with torch.inference_mode():
            # Get logits: shape (batch, seq_len, vocab_size)
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            
            # Shift: logits[t-1] predicts token[t]
            # So we take logits[:, :-1] to predict labels[:, 1:]
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather the log prob of the actual token at each position
            token_logprobs = torch.gather(
                log_probs, 
                dim=-1, 
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask to only count continuation tokens
            # continuation_mask[:, 1:] aligns with our shifted labels
            masked_logprobs = token_logprobs * continuation_mask[:, 1:]
            
            # Sum over sequence dimension to get total log prob per sample
            total_logprobs = masked_logprobs.sum(dim=1)
        
        return total_logprobs.cpu().tolist()
    
    @classmethod
    def load(cls, config: ModelConfig) -> "CausalModelAdapter":
        """Load a causal language model."""
        tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Try AutoModelForCausalLM first, fallback to AutoModel
        dtype = getattr(torch, config.torch_dtype)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.name_or_path,
                torch_dtype=dtype,
                trust_remote_code=config.trust_remote_code,
            )
            print(f"[INFO] Loaded model using AutoModelForCausalLM: {config.name_or_path}")
        except (ValueError, OSError, ImportError) as e:
            print(f"[WARNING] AutoModelForCausalLM failed: {e}")
            print(f"[INFO] Falling back to AutoModel...")
            model = AutoModel.from_pretrained(
                config.name_or_path,
                torch_dtype=dtype,
                trust_remote_code=config.trust_remote_code,
            )
            print(f"[INFO] Loaded model using AutoModel: {config.name_or_path}")
        
        adapter = cls(model, tokenizer, config)
        return adapter.to(config.device).eval()