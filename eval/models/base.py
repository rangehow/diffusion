# eval/models/base.py
"""
Abstract base class for model adapters.
Model adapters encapsulate model-specific logic for computing log-probabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import ModelConfig


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Each model type (causal LM, diffusion, etc.) should have its own adapter
    that implements the compute_logprobs method.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return self.model.device
    
    @abstractmethod
    def compute_logprobs(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute log-probabilities for each item in the batch.
        
        Args:
            batch: A dictionary containing:
                - input_ids: Tensor of shape (batch_size, seq_len)
                - attention_mask: Tensor of shape (batch_size, seq_len)
                - continuation_mask: Boolean tensor marking continuation tokens
                
        Returns:
            List of log-probabilities, one per item in the batch.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, config: ModelConfig) -> "ModelAdapter":
        """
        Load a model and create an adapter instance.
        
        Args:
            config: Model configuration
            
        Returns:
            Initialized ModelAdapter instance
        """
        pass
    
    def to(self, device: str) -> "ModelAdapter":
        """Move model to specified device."""
        self.model = self.model.to(device)
        return self
    
    def eval(self) -> "ModelAdapter":
        """Set model to evaluation mode."""
        self.model.eval()
        return self