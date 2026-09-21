# eval/engine/metrics.py
"""
Metrics calculation for evaluation tasks.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class AccuracyMetrics:
    """Standard accuracy metrics for multiple choice tasks."""
    accuracy: float
    accuracy_norm_char: float
    accuracy_norm_token: float
    correct_count: int
    correct_norm_char_count: int
    correct_norm_token_count: int
    total: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "acc": self.accuracy,
            "acc_norm_char": self.accuracy_norm_char,
            "acc_norm_token": self.accuracy_norm_token,
            "acc_correct": self.correct_count,
            "acc_norm_char_correct": self.correct_norm_char_count,
            "acc_norm_token_correct": self.correct_norm_token_count,
            "total": self.total,
        }


@dataclass
class MC2Metrics:
    """Metrics for TruthfulQA MC2 task."""
    prob_mass_score: float
    scored_count: int
    total_questions: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mc2_prob_mass_score": self.prob_mass_score,
            "scored_count": self.scored_count,
            "total_questions": self.total_questions,
        }


def calculate_accuracy_metrics(
    group_results: Dict[str, List[Dict[str, Any]]]
) -> AccuracyMetrics:
    """
    Calculate accuracy metrics from grouped results.
    
    Args:
        group_results: Dictionary mapping group_id to list of option results.
            Each option result should have:
                - logprob: Log probability of the option
                - is_correct: Whether this option is correct (0 or 1)
                - continuation_len: Length in tokens
                - continuation_char_len: Length in characters
                
    Returns:
        AccuracyMetrics dataclass with all metrics
    """
    total_groups = len(group_results)
    
    if total_groups == 0:
        return AccuracyMetrics(
            accuracy=0.0,
            accuracy_norm_char=0.0,
            accuracy_norm_token=0.0,
            correct_count=0,
            correct_norm_char_count=0,
            correct_norm_token_count=0,
            total=0,
        )
    
    correct_raw = 0
    correct_norm_char = 0
    correct_norm_token = 0
    
    for group_id, group_data in group_results.items():
        # Raw accuracy: highest log-prob
        best_option = max(group_data, key=lambda x: x['logprob'])
        if best_option['is_correct'] == 1:
            correct_raw += 1
        
        # Character-normalized accuracy
        best_norm_char = max(
            group_data,
            key=lambda x: x['logprob'] / max(x['continuation_char_len'], 1)
        )
        if best_norm_char['is_correct'] == 1:
            correct_norm_char += 1
        
        # Token-normalized accuracy
        best_norm_token = max(
            group_data,
            key=lambda x: x['logprob'] / max(x['continuation_len'], 1)
        )
        if best_norm_token['is_correct'] == 1:
            correct_norm_token += 1
    
    return AccuracyMetrics(
        accuracy=correct_raw / total_groups,
        accuracy_norm_char=correct_norm_char / total_groups,
        accuracy_norm_token=correct_norm_token / total_groups,
        correct_count=correct_raw,
        correct_norm_char_count=correct_norm_char,
        correct_norm_token_count=correct_norm_token,
        total=total_groups,
    )


def calculate_mc2_metrics(
    group_results: Dict[str, List[Dict[str, Any]]]
) -> MC2Metrics:
    """
    Calculate MC2 probability mass metric for TruthfulQA.
    
    This metric measures the probability mass assigned to all correct answers.
    
    Args:
        group_results: Dictionary mapping group_id to list of option results.
            
    Returns:
        MC2Metrics dataclass
    """
    total_groups = len(group_results)
    
    if total_groups == 0:
        return MC2Metrics(prob_mass_score=0.0, scored_count=0, total_questions=0)
    
    prob_mass_scores = []
    
    for group_id, group_data in group_results.items():
        if not group_data:
            continue
        
        logprobs = np.array([item['logprob'] for item in group_data])
        labels = np.array([item['is_correct'] for item in group_data])
        
        # Convert to probabilities via softmax
        probs = np.exp(logprobs - np.max(logprobs))  # Subtract max for numerical stability
        probs = probs / np.sum(probs)
        
        # Sum probability mass for correct answers
        prob_mass_true = np.sum(probs[labels == 1])
        prob_mass_scores.append(prob_mass_true)
    
    score = np.mean(prob_mass_scores) if prob_mass_scores else 0.0
    
    return MC2Metrics(
        prob_mass_score=score,
        scored_count=len(prob_mass_scores),
        total_questions=total_groups,
    )


def aggregate_results(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate flat results into groups by group_id.
    
    Args:
        results: List of individual result dictionaries
        
    Returns:
        Dictionary mapping group_id to list of results
    """
    group_results: Dict[str, List[Dict[str, Any]]] = {}
    
    for result in results:
        group_id = result['group_id']
        if group_id not in group_results:
            group_results[group_id] = []
        group_results[group_id].append(result)
    
    return group_results