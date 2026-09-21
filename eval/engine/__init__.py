# eval/engine/__init__.py
"""
Evaluation engine package.

Contains the core evaluation orchestration, distributed processing,
and metrics calculation.
"""

from .evaluator import Evaluator, EvaluationResult
from .simple_evaluator import SimpleEvaluator, evaluate_dataset_simple
from .distributed import DistributedEvaluator, EvaluatorActor
from .metrics import (
    calculate_accuracy_metrics,
    calculate_mc2_metrics,
    aggregate_results,
    AccuracyMetrics,
    MC2Metrics,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "SimpleEvaluator",
    "evaluate_dataset_simple",
    "DistributedEvaluator",
    "EvaluatorActor",
    "calculate_accuracy_metrics",
    "calculate_mc2_metrics",
    "aggregate_results",
    "AccuracyMetrics",
    "MC2Metrics",
]