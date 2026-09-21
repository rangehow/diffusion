# eval/__init__.py
"""
Evaluation Framework for Language Models

A modular framework for evaluating language models on standard benchmarks.

Supported model types:
    - Causal (autoregressive) language models
    - Discrete diffusion language models

Supported tasks:
    - HellaSwag
    - ARC (Easy and Challenge)
    - MMLU
    - PIQA
    - Winogrande
    - TruthfulQA (MC1 and MC2)
    - CommonsenseQA
    - SciQ

Example usage:
    from eval import Evaluator, ModelConfig, ModelType
    
    config = ModelConfig(
        name_or_path="path/to/model",
        model_type=ModelType.CAUSAL,
    )
    
    with Evaluator(config) as evaluator:
        result = evaluator.evaluate_task("hellaswag", num_fewshot=10)
        print(result.metrics)
"""

from .config import (
    ModelConfig,
    TaskConfig,
    EvaluationConfig,
    ModelType,
    DiffusionEvalMode,
    SamplerType,
)
from .engine import Evaluator, EvaluationResult
from .tasks import TaskRegistry, get_task, BaseTask
from .io import ResultsManager, export_to_excel

__version__ = "2.0.0"

__all__ = [
    # Config
    "ModelConfig",
    "TaskConfig",
    "EvaluationConfig",
    "ModelType",
    "DiffusionEvalMode",
    "SamplerType",
    
    # Engine
    "Evaluator",
    "EvaluationResult",
    
    # Tasks
    "TaskRegistry",
    "get_task",
    "BaseTask",
    
    # IO
    "ResultsManager",
    "export_to_excel",
]