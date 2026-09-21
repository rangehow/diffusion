#!/usr/bin/env python3
# eval/main.py
"""
Command-line interface for the evaluation framework.

Usage:
    python -m eval.main --model_name_or_path /path/to/model --tasks hellaswag,mmlu --num_fewshot 5

For multi-GPU evaluation (no NCCL needed):
    python -m eval.main ... --use_multi_gpu

For diffusion models:
    python -m eval.main --model_name_or_path /path/to/diffusion_model \\
        --model_type discrete_diffusion \\
        --diffusion_type mdlm \\
        --diffusion_eval_mode mc \\
        --tasks hellaswag
"""

import argparse
import sys

from .config import ModelConfig, ModelType, DiffusionEvalMode, DiffusionType
from .engine import Evaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models on standard benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or HuggingFace model name",
    )
    model_group.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['causal', 'discrete_diffusion'],
        help="Type of language model",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action='store_true',
        help="Trust remote code when loading model",
    )
    
    # Task arguments
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of tasks to evaluate",
    )
    task_group.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    task_group.add_argument(
        "--sampler",
        type=str,
        default="random",
        choices=['first_n', 'random', 'balanced'],
        help="Few-shot sampling strategy",
    )
    task_group.add_argument(
        "--sampler_seed",
        type=int,
        default=42,
        help="Random seed for few-shot sampling",
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU",
    )
    eval_group.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples (0 = no limit)",
    )
    eval_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    eval_group.add_argument(
        "--use_ray",
        action='store_true',
        help="Use Ray distributed evaluation (may hang on some systems). "
             "Default is to use simple single-node evaluation.",
    )
    eval_group.add_argument(
        "--use_multi_gpu",
        action='store_true',
        help="Use multiprocessing across GPUs (no NCCL). Each GPU runs independently.",
    )
    
    # Diffusion-specific arguments
    diff_group = parser.add_argument_group("Diffusion Model Settings")
    diff_group.add_argument(
        "--diffusion_eval_mode",
        type=str,
        default="pll",
        choices=['mc', 'pll'],
        help="Evaluation mode: 'mc' (Monte Carlo) or 'pll' (Pseudo-Log-Likelihood)",
    )
    diff_group.add_argument(
        "--diffusion_type",
        type=str,
        default="mdlm",
        choices=['causal', 'mdlm', 'bd3lm', 'prefixlm'],
        help="Diffusion model architecture: 'causal' (tail-biased), 'mdlm' (uniform), 'bd3lm' (block), 'prefixlm' (contiguous suffix)",
    )
    diff_group.add_argument(
        "--mc_num",
        type=int,
        default=32,
        help="Number of Monte Carlo samples",
    )
    diff_group.add_argument(
        "--mc_batch_size",
        type=int,
        default=16,
        help="Batch size for Monte Carlo sampling",
    )
    diff_group.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for BD3LM (only used when diffusion_type=bd3lm)",
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for saving results",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Evaluation Framework v2.0")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Type: {args.model_type}")
    print(f"Tasks: {args.tasks}")
    print(f"Few-shot: {args.num_fewshot}")
    
    # Report evaluation method
    if args.use_ray:
        print(f"Evaluation: Ray distributed")
    elif args.use_multi_gpu:
        print(f"Evaluation: Multi-GPU (multiprocessing, no NCCL)")
    else:
        print(f"Evaluation: Single GPU")
    
    if args.model_type == 'discrete_diffusion':
        print(f"Diffusion Type: {args.diffusion_type}")
        print(f"Eval Mode: {args.diffusion_eval_mode}")
        if args.diffusion_eval_mode == 'mc':
            print(f"MC Samples: {args.mc_num}")
    print("=" * 60)
    
    # Create model configuration
    model_config = ModelConfig(
        name_or_path=args.model_name_or_path,
        model_type=ModelType(args.model_type),
        trust_remote_code=args.trust_remote_code,
        diffusion_eval_mode=DiffusionEvalMode(args.diffusion_eval_mode),
        diffusion_type=DiffusionType(args.diffusion_type),
        mc_num=args.mc_num,
        mc_batch_size=args.mc_batch_size if hasattr(args, 'mc_batch_size') else args.batch_size,
    )
    
    # Parse task list
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    
    if not tasks:
        print("Error: No tasks specified")
        sys.exit(1)
    
    # Run evaluation - track failed tasks
    failed_tasks = []
    
    with Evaluator(
        model_config, 
        args.batch_size, 
        args.num_workers,
        use_ray=args.use_ray,
        use_multi_gpu=args.use_multi_gpu,
    ) as evaluator:
        for task_name in tasks:
            try:
                # Evaluation logic handles caching and saving internally
                evaluator.evaluate_task(
                    task_name=task_name,
                    num_fewshot=args.num_fewshot,
                    sampler_name=args.sampler,
                    sampler_seed=args.sampler_seed,
                    limit=args.limit,
                    output_dir=args.output_dir,
                )
                
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_tasks.append((task_name, str(e)))
    
    print("\n" + "=" * 60)
    if failed_tasks:
        print("Evaluation FAILED!")
        print("=" * 60)
        print(f"Failed tasks ({len(failed_tasks)}/{len(tasks)}):")
        for task, error in failed_tasks:
            # Truncate long error messages
            error_short = error[:200] + "..." if len(error) > 200 else error
            print(f"  - {task}: {error_short}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("Evaluation complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()