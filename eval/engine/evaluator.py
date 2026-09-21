# eval/engine/evaluator.py
"""
Core evaluation orchestrator.

Ties together models, tasks, and metrics into a cohesive evaluation pipeline.

Uses SimpleEvaluator by default which supports:
- Single GPU: Sequential processing (default, most reliable)
- Multi GPU: Multiprocessing with one process per GPU (no NCCL needed)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from transformers import AutoTokenizer

from ..config import ModelConfig, TaskConfig, EvaluationConfig, ModelType, DiffusionEvalMode, DiffusionType
from ..tasks import get_task, TaskRegistry
from .simple_evaluator import SimpleEvaluator
from .metrics import (
    calculate_accuracy_metrics,
    calculate_mc2_metrics,
    aggregate_results,
    AccuracyMetrics,
    MC2Metrics,
)
from ..io.results import ResultsManager


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_name: str
    metrics: Dict[str, Any]
    detailed_results: list
    metadata: Dict[str, Any]
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"Results for: {self.task_name.upper()}")
        print(f"Model: {self.metadata.get('model_short_name', 'Unknown')}")
        print(f"{'='*60}")
        
        if self.task_name == 'truthfulqa_mc2':
            score = self.metrics.get('mc2_prob_mass_score', 0.0)
            total = self.metrics.get('total_questions', 0)
            scored = self.metrics.get('scored_count', 0)
            print(f"Metric: Normalized Probability Mass (MC2)")
            print(f"Score: {score:.4f} (Evaluated on {scored}/{total} questions)")
        else:
            total = self.metrics.get('total', 0)
            for name, value in self.metrics.items():
                if name.startswith('acc') and 'correct' not in name:
                    correct = self.metrics.get(f"{name}_correct", 0)
                    print(f"{name}: {value:.4f} ({correct}/{total})")
        
        print(f"{'='*60}\n")


class Evaluator:
    """
    Main evaluator class that orchestrates the evaluation pipeline.
    
    By default, uses SimpleEvaluator which is more reliable than Ray.
    Set use_ray=True to use distributed Ray evaluation if needed.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        batch_size: int = 8,
        num_workers: int = 4,
        use_ray: bool = False,
        use_multi_gpu: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_config: Configuration for the model
            batch_size: Batch size for evaluation (per GPU if multi-GPU)
            num_workers: DataLoader workers
            use_ray: If True, use Ray distributed evaluation (may hang)
            use_multi_gpu: If True and not using Ray, use multiprocessing across GPUs
        """
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ray = use_ray
        self.use_multi_gpu = use_multi_gpu
        
        # Load tokenizer for task processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        
        # Set pad token for causal models
        if (
            self.tokenizer.pad_token_id is None and 
            model_config.model_type == ModelType.CAUSAL
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self._simple_evaluator: Optional[SimpleEvaluator] = None
        self._distributed_evaluator = None  # Only created if use_ray=True
    
    @property
    def simple_evaluator(self) -> SimpleEvaluator:
        """Lazy initialization of simple evaluator."""
        if self._simple_evaluator is None:
            self._simple_evaluator = SimpleEvaluator(
                self.model_config,
                use_multi_gpu=self.use_multi_gpu,
            )
        return self._simple_evaluator
    
    @property
    def distributed_evaluator(self):
        """Lazy initialization of distributed (Ray) evaluator."""
        if self._distributed_evaluator is None:
            from .distributed import DistributedEvaluator
            self._distributed_evaluator = DistributedEvaluator(self.model_config)
        return self._distributed_evaluator
    
    def _build_run_signature(
        self, 
        task_name: str, 
        num_fewshot: int, 
        sampler_name: str, 
        sampler_seed: int,
        limit: int
    ) -> Dict[str, Any]:
        """
        Construct the dictionary that defines the identity of a run.
        Changes to these parameters will result in a new evaluation run.
        """
        sig = {
            "task": task_name,
            "fewshot": {
                "num": num_fewshot,
                "sampler": sampler_name,
                "seed": sampler_seed
            },
            "limit": limit,
            "model_type": self.model_config.model_type.value,
        }
        
        # Add diffusion specific args if applicable
        if self.model_config.model_type == ModelType.DISCRETE_DIFFUSION:
            sig["diffusion"] = {
                "type": self.model_config.diffusion_type.value,
                "mode": self.model_config.diffusion_eval_mode.value,
            }
            # Only add MC params if mode is MC
            if self.model_config.diffusion_eval_mode == DiffusionEvalMode.MONTE_CARLO:
                sig["diffusion"]["mc_num"] = self.model_config.mc_num
                
                # BD3LM specific
                if self.model_config.diffusion_type == DiffusionType.BD3LM:
                    sig["diffusion"]["block_size"] = getattr(self.model_config, 'block_size', 16)
        
        return sig

    def evaluate_task(
        self,
        task_name: str,
        num_fewshot: int = 0,
        sampler_name: str = "random",
        sampler_seed: int = 42,
        limit: int = 0,
        output_dir: str = "results",
    ) -> EvaluationResult:
        """
        Evaluate a single task with automatic result reuse.
        
        Args:
            task_name: Name of the task to evaluate
            num_fewshot: Number of few-shot examples
            sampler_name: Few-shot sampler type
            sampler_seed: Random seed for sampling
            limit: Limit number of samples (0 for no limit)
            output_dir: Directory to store/check results
            
        Returns:
            EvaluationResult with metrics and detailed results
        """
        # 1. Build Run Signature
        run_params = self._build_run_signature(
            task_name, num_fewshot, sampler_name, sampler_seed, limit
        )
        
        # 2. Check for existing result
        results_manager = ResultsManager(output_dir, self.model_config.name_or_path, task_name)
        existing_entry = results_manager.find_existing_run(run_params)
        
        if existing_entry:
            run_id = existing_entry['id']
            print(f"\n{'='*60}")
            print(f"Skipping: {task_name} ({num_fewshot}-shot)")
            print(f"Reason: Found existing run (ID: {run_id})")
            print(f"{'='*60}")
            
            cached_result = results_manager.load_result(run_id)
            if cached_result:
                cached_result.print_summary()
                return cached_result
        
        # 3. Proceed with Evaluation
        print(f"\n{'='*60}")
        print(f"Evaluating: {task_name} ({num_fewshot}-shot)")
        if self.model_config.model_type == ModelType.DISCRETE_DIFFUSION:
            print(f"Diffusion eval mode: {self.model_config.diffusion_eval_mode.value}")
        
        if self.use_ray:
            eval_method = "Ray distributed"
        elif self.use_multi_gpu:
            eval_method = "Multi-GPU (multiprocessing)"
        else:
            eval_method = "Single GPU"
        print(f"Evaluation method: {eval_method}")
        print(f"{'='*60}")
        
        # Create task configuration
        task_config = TaskConfig(
            tokenizer=self.tokenizer,
            num_fewshot=num_fewshot,
            sampler_type=sampler_name,
            sampler_seed=sampler_seed,
            model_type=self.model_config.model_type,
        )
        
        # Load and process task
        task = get_task(task_name, task_config)
        dataset = task.process()
        
        if limit > 0:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Log sample info
        self._log_sample_info(dataset, task_name)
        
        # Run evaluation - choose method based on flags
        if self.use_ray:
            results = self.distributed_evaluator.evaluate(
                dataset,
                self.tokenizer,
                self.batch_size,
                self.num_workers,
            )
        else:
            results = self.simple_evaluator.evaluate(
                dataset,
                self.tokenizer,
                self.batch_size,
                self.num_workers,
            )
        
        # Aggregate and compute metrics
        group_results = aggregate_results(results)
        
        if task_name == 'truthfulqa_mc2':
            metrics_obj = calculate_mc2_metrics(group_results)
        else:
            metrics_obj = calculate_accuracy_metrics(group_results)
        
        # Log option length statistics
        self._log_option_stats(group_results, task_name)
        
        # Create result object
        result = EvaluationResult(
            task_name=task_name,
            metrics=metrics_obj.to_dict(),
            detailed_results=results,
            metadata={
                'model_name_or_path': self.model_config.name_or_path,
                'model_short_name': self._get_model_short_name(),
                'model_type': self.model_config.model_type.value,
                'task_name': task_name,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'run_params': run_params,
            },
        )
        
        # 4. Save results
        results_manager.save(result, run_params)
        
        result.print_summary()
        return result
    
    def _get_model_short_name(self) -> str:
        """Extract a short name from the model path."""
        from pathlib import Path
        path = Path(self.model_config.name_or_path)
        
        if path.name == '':
            path = path.parent
        
        if path.name.startswith('checkpoint-'):
            return f"{path.parent.name}_{path.name}"
        
        return path.name
    
    def _log_sample_info(self, dataset, task_name: str):
        """Log information about sample structure."""
        if not dataset:
            return
        
        try:
            example_group_id = dataset[0]['group_id']
            
            prompt_info = None
            choices_info = []
            
            for item in dataset:
                if item['group_id'] == example_group_id:
                    if prompt_info is None:
                        prompt_info = {
                            'text': self.tokenizer.decode(
                                item['input_ids'], skip_special_tokens=True
                            ),
                            'ids': item['input_ids'],
                        }
                    choices_info.append({
                        'text': self.tokenizer.decode(
                            item['continuation_ids'], skip_special_tokens=True
                        ).strip(),
                        'ids': item['continuation_ids'],
                    })
            
            if prompt_info and choices_info:
                print(f"\n--- Sample Preview ({task_name}) ---")
                print(f"Prompt:\n{prompt_info['text'][:500]}...")
                print(f"Token IDs (first 20): {prompt_info['ids'][:20]}")
                print(f"\nChoices ({len(choices_info)} options):")
                for i, choice in enumerate(choices_info[:4]):  # Show first 4
                    print(f"  {i+1}. '{choice['text'][:50]}...' -> {choice['ids']}")
                print("---\n")
        except Exception as e:
            print(f"[WARNING] Could not log sample info: {e}")
    
    def _log_option_stats(
        self, 
        group_results: Dict[str, list], 
        task_name: str
    ):
        """Log statistics about option lengths."""
        token_lengths = []
        char_lengths = []
        
        for group_data in group_results.values():
            for item in group_data:
                token_lengths.append(item['continuation_len'])
                char_lengths.append(item['continuation_char_len'])
        
        if token_lengths:
            print(f"\n[STATS] {task_name} option lengths:")
            print(f"  Tokens - avg: {sum(token_lengths)/len(token_lengths):.2f}, "
                  f"min: {min(token_lengths)}, max: {max(token_lengths)}")
            print(f"  Chars  - avg: {sum(char_lengths)/len(char_lengths):.2f}")
    
    def shutdown(self):
        """Release resources."""
        if self._simple_evaluator is not None:
            self._simple_evaluator.shutdown()
        if self._distributed_evaluator is not None:
            self._distributed_evaluator.shutdown()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()