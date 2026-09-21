# eval/models/__init__.py
"""
Model adapters package.

Provides a unified interface for different model types (causal, diffusion, etc.)
"""

from .base import ModelAdapter
from .causal import CausalModelAdapter
from .diffusion import DiffusionModelAdapter
from ..config import ModelConfig, ModelType


def load_model(config: ModelConfig) -> ModelAdapter:
    """
    Factory function to load the appropriate model adapter.
    
    Args:
        config: Model configuration specifying type and parameters
        
    Returns:
        Initialized ModelAdapter instance
        
    Raises:
        ValueError: If model type is not supported
    """
    adapters = {
        ModelType.CAUSAL: CausalModelAdapter,
        ModelType.DISCRETE_DIFFUSION: DiffusionModelAdapter,
    }
    
    adapter_class = adapters.get(config.model_type)
    if adapter_class is None:
        raise ValueError(
            f"Unsupported model type: {config.model_type}. "
            f"Supported types: {list(adapters.keys())}"
        )
    
    return adapter_class.load(config)


__all__ = [
    "ModelAdapter",
    "CausalModelAdapter", 
    "DiffusionModelAdapter",
    "load_model",
]