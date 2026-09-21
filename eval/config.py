# eval/config.py
"""
Centralized configuration for the evaluation framework.
All configuration dataclasses and constants live here.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Literal
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    CAUSAL = "causal"
    DISCRETE_DIFFUSION = "discrete_diffusion"


class DiffusionEvalMode(str, Enum):
    """Evaluation modes for discrete diffusion models."""
    MONTE_CARLO = "mc"
    PSEUDO_LOG_LIKELIHOOD = "pll"


class DiffusionType(str, Enum):
    """Diffusion model architectures/training objectives."""
    CAUSAL = "causal"      # CausalMLM style (tail-biased masking)
    MDLM = "mdlm"          # Masked Diffusion LM (uniform random masking)
    BD3LM = "bd3lm"        # Block Denoising Discrete Diffusion LM
    PREFIXLM = "prefixlm"  # Prefix LM (solid suffix masking)


class SamplerType(str, Enum):
    """Available few-shot sampling strategies."""
    FIRST_N = "first_n"
    RANDOM = "random"
    BALANCED = "balanced"


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    name_or_path: str
    model_type: ModelType = ModelType.CAUSAL
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    
    # Diffusion-specific settings
    diffusion_eval_mode: DiffusionEvalMode = DiffusionEvalMode.PSEUDO_LOG_LIKELIHOOD
    diffusion_type: DiffusionType = DiffusionType.MDLM  # NEW: which diffusion variant
    mc_num: int = 128
    mc_batch_size: int = 16
    
    # BD3LM specific
    block_size: int = 16


@dataclass
class TaskConfig:
    """Configuration for task processing."""
    tokenizer: Optional[Any] = None
    num_fewshot: int = 0
    sampler_type: SamplerType = SamplerType.RANDOM
    sampler_seed: int = 42
    text_only: bool = False
    local_dir: Optional[str] = None
    model_type: ModelType = ModelType.CAUSAL

    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.sampler_type, str):
            self.sampler_type = SamplerType(self.sampler_type)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    model: ModelConfig
    tasks: list[str]
    batch_size: int = 8
    limit: int = 0  # 0 means no limit
    num_workers: int = 4
    output_dir: str = "results"
    
    @classmethod
    def from_args(cls, args) -> "EvaluationConfig":
        """Create config from argparse namespace."""
        model_config = ModelConfig(
            name_or_path=args.model_name_or_path,
            model_type=ModelType(args.model_type),
            trust_remote_code=args.trust_remote_code,
            diffusion_eval_mode=DiffusionEvalMode(args.diffusion_eval_mode),
            diffusion_type=DiffusionType(getattr(args, 'diffusion_type', 'mdlm')),
            mc_num=args.mc_num,
            block_size=getattr(args, 'block_size', 16),
        )
        
        tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
        
        return cls(
            model=model_config,
            tasks=tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
        )


@dataclass
class RunMetadata:
    """Metadata about a single evaluation run."""
    model_name_or_path: str
    model_short_name: str
    model_type: str
    task_name: str
    timestamp_utc: str
    run_args: dict = field(default_factory=dict)