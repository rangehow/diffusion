"""
Simple single-node multi-GPU evaluation without Ray or NCCL.

This module provides evaluation approaches that:
1. Single GPU: Loads model once, processes batches sequentially
2. Multi GPU: Spawns one process per GPU, each loads model independently

No Ray, no NCCL - just simple multiprocessing with independent GPU processes.
"""

from typing import Dict, List, Any, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..config import ModelConfig
from ..models import load_model
from ..data import LogProbCollator


def _disable_dynamo():
    """Disable torch dynamo to avoid compilation errors with dynamic shapes."""
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
    except (ImportError, AttributeError):
        pass


def _suppress_worker_logging():
    """Suppress verbose logging in worker processes to avoid interleaved output.
    
    Workers report status via the result queue; the main process prints
    a consolidated summary. This prevents:
    - 8x interleaved 'Loading weights' tqdm bars from transformers
    - 8x duplicate [INFO] messages from model adapters
    - Chaotic multi-device output that's impossible to read
    """
    import os
    import sys
    
    # 1. Suppress transformers INFO logs and progress bars
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    
    # 2. Suppress tqdm globally for this process
    try:
        from tqdm import tqdm as _tqdm
        from functools import partialmethod
        _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
    except Exception:
        pass
    
    # 3. Redirect stdout to devnull to suppress print() from model adapters
    #    Keep stderr alive for genuine error messages
    sys.stdout = open(os.devnull, 'w')


def _worker_init(
    gpu_id: int,
    config_dict: dict,
    dataset_indices: List[int],
    dataset,
    tokenizer_path: str,
    batch_size: int,
    num_workers: int,
    result_queue: mp.Queue,
    trust_remote_code: bool,
):
    """
    Worker function that runs on each GPU.
    
    Each worker:
    1. Loads the model and explicitly moves it to the assigned GPU
    2. Processes its assigned subset of data
    3. Puts results in the shared queue
    
    Logging is suppressed in workers; status is reported via the queue.
    """
    try:
        # CRITICAL: Suppress verbose logging FIRST to avoid interleaved output
        _suppress_worker_logging()
        
        # CRITICAL: Disable dynamo FIRST before any model operations
        _disable_dynamo()
        
        import time
        t_start = time.time()
        
        device = f"cuda:{gpu_id}"
        
        # Reconstruct config with explicit device
        from ..config import ModelConfig, ModelType, DiffusionEvalMode, DiffusionType
        config = ModelConfig(
            name_or_path=config_dict['name_or_path'],
            model_type=ModelType(config_dict['model_type']),
            trust_remote_code=config_dict['trust_remote_code'],
            torch_dtype=config_dict['torch_dtype'],
            device=device,
            diffusion_eval_mode=DiffusionEvalMode(config_dict['diffusion_eval_mode']),
            diffusion_type=DiffusionType(config_dict['diffusion_type']),
            mc_num=config_dict['mc_num'],
            mc_batch_size=config_dict['mc_batch_size'],
            block_size=config_dict.get('block_size', 16),
        )
        
        # Load model and move to target device
        adapter = load_model(config).to(device).eval()
        
        t_loaded = time.time()
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Create subset dataloader
        subset = Subset(dataset, dataset_indices)
        collator = LogProbCollator(tokenizer)
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )
        
        # Process batches
        results = []
        for batch in dataloader:
            # Move batch tensors to target device
            batch = _move_batch_to_device(batch, device)
            logprobs = adapter.compute_logprobs(batch)
            
            for gid, lp, ic, cl, ccl in zip(
                batch['group_ids'],
                logprobs,
                batch['is_correct'],
                batch['continuation_len'],
                batch['continuation_char_len'],
            ):
                results.append({
                    'group_id': gid,
                    'logprob': lp,
                    'is_correct': ic,
                    'continuation_len': cl,
                    'continuation_char_len': ccl,
                })
        
        t_done = time.time()
        # Report timing via queue so main process can print consolidated summary
        timing = {
            'load_seconds': round(t_loaded - t_start, 1),
            'eval_seconds': round(t_done - t_loaded, 1),
            'total_seconds': round(t_done - t_start, 1),
        }
        result_queue.put((gpu_id, results, None, timing))
        
    except Exception as e:
        import traceback
        import sys
        error_msg = f"Worker {gpu_id} error: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        result_queue.put((gpu_id, [], error_msg, None))


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move tensor values in a batch dict to the specified device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class SimpleEvaluator:
    """
    Simple evaluator for single-node evaluation.
    
    Supports:
    - Single GPU: Sequential processing (default)
    - Multi GPU: Multiprocessing with one process per GPU (no NCCL)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        use_multi_gpu: bool = False,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Initialize the simple evaluator.
        
        Args:
            config: Model configuration
            use_multi_gpu: If True, use multiprocessing across GPUs
            device_ids: List of GPU IDs to use (None = all available)
        """
        # Disable dynamo at initialization
        _disable_dynamo()
        
        self.config = config
        self.use_multi_gpu = use_multi_gpu
        
        # Determine devices
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids
        
        self.num_gpus = len(self.device_ids)
        
        print(f"[INFO] Evaluator initialized")
        print(f"[INFO] Available GPUs: {self.device_ids}")
        print(f"[INFO] Multi-GPU mode: {use_multi_gpu}")
        
        # For single GPU mode, load model immediately
        if not use_multi_gpu or self.num_gpus <= 1:
            self.use_multi_gpu = False
            device = f"cuda:{self.device_ids[0]}" if self.device_ids else "cuda:0"
            print(f"[INFO] Loading model on {device}...")
            self.adapter = load_model(config).to(device).eval()
            self.device = device
            self.tokenizer = self.adapter.tokenizer
        else:
            self.adapter = None
            self.tokenizer = None
            self.device = None
    
    def evaluate(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset.
        
        Args:
            dataset: HuggingFace dataset with processed samples
            tokenizer: Tokenizer for the collator
            batch_size: Batch size per GPU
            num_workers: DataLoader workers per GPU
            
        Returns:
            List of all result dictionaries
        """
        if self.use_multi_gpu and self.num_gpus > 1:
            return self._evaluate_multi_gpu(dataset, tokenizer, batch_size, num_workers)
        else:
            return self._evaluate_single_gpu(dataset, tokenizer, batch_size, num_workers)
    
    def _evaluate_single_gpu(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
    ) -> List[Dict[str, Any]]:
        """Single GPU evaluation - simple and reliable."""
        print(f"[INFO] Evaluating {len(dataset)} samples on {self.device}...")
        
        collator = LogProbCollator(tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        all_results = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = _move_batch_to_device(batch, self.device)
            logprobs = self.adapter.compute_logprobs(batch)
            
            for gid, lp, ic, cl, ccl in zip(
                batch['group_ids'],
                logprobs,
                batch['is_correct'],
                batch['continuation_len'],
                batch['continuation_char_len'],
            ):
                all_results.append({
                    'group_id': gid,
                    'logprob': lp,
                    'is_correct': ic,
                    'continuation_len': cl,
                    'continuation_char_len': ccl,
                })
        
        return all_results
    
    def _evaluate_multi_gpu(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
    ) -> List[Dict[str, Any]]:
        """Multi-GPU evaluation using multiprocessing (no NCCL)."""
        print(f"[INFO] Evaluating {len(dataset)} samples across {self.num_gpus} GPUs...")
        print(f"[INFO] Batch size per GPU: {batch_size}")
        
        # Split dataset indices across GPUs
        total_samples = len(dataset)
        indices_per_gpu = self._split_indices(total_samples, self.num_gpus)
        
        # Track expected samples per worker for validation
        expected_samples_per_worker = {}
        active_gpu_ids = []
        for i, indices in enumerate(indices_per_gpu):
            if indices:
                expected_samples_per_worker[self.device_ids[i]] = len(indices)
                active_gpu_ids.append(self.device_ids[i])
        
        # Print consolidated startup info
        if active_gpu_ids:
            gpu_range = (
                f"cuda:{active_gpu_ids[0]}-{active_gpu_ids[-1]}" 
                if len(active_gpu_ids) > 1 
                else f"cuda:{active_gpu_ids[0]}"
            )
            samples_counts = list(expected_samples_per_worker.values())
            if len(set(samples_counts)) == 1:
                print(f"[INFO] Starting {len(active_gpu_ids)} workers on {gpu_range}, {samples_counts[0]} samples each")
            else:
                print(f"[INFO] Starting {len(active_gpu_ids)} workers on {gpu_range}, {min(samples_counts)}~{max(samples_counts)} samples each")
        
        # Prepare config dict for serialization
        config_dict = {
            'name_or_path': self.config.name_or_path,
            'model_type': self.config.model_type.value,
            'trust_remote_code': self.config.trust_remote_code,
            'torch_dtype': self.config.torch_dtype,
            'diffusion_eval_mode': self.config.diffusion_eval_mode.value,
            'diffusion_type': self.config.diffusion_type.value,
            'mc_num': self.config.mc_num,
            'mc_batch_size': self.config.mc_batch_size,
            'block_size': getattr(self.config, 'block_size', 16),
        }
        
        # Use spawn to avoid CUDA issues
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        # Start worker processes
        processes = []
        worker_gpu_ids = []  # Track which GPU each process is assigned to
        for i, gpu_id in enumerate(self.device_ids):
            if not indices_per_gpu[i]:
                continue
            
            p = ctx.Process(
                target=_worker_init,
                args=(
                    gpu_id,
                    config_dict,
                    indices_per_gpu[i],
                    dataset,
                    self.config.name_or_path,
                    batch_size,
                    max(1, num_workers // self.num_gpus),
                    result_queue,
                    self.config.trust_remote_code,
                ),
            )
            p.start()
            processes.append(p)
            worker_gpu_ids.append(gpu_id)
        
        # Collect results with timeout to detect crashed workers
        all_results = []
        errors = []
        received_from = set()
        worker_timings = {}  # gpu_id -> timing dict
        
        print(f"[INFO] Waiting for {len(processes)} workers to complete...")
        
        # Use timeout-based collection to detect hung/crashed workers
        timeout_per_worker = 3600  # 1 hour timeout per worker (adjust as needed)
        
        for idx in range(len(processes)):
            try:
                # Check if any process has died before waiting
                for i, p in enumerate(processes):
                    if not p.is_alive() and worker_gpu_ids[i] not in received_from:
                        # Process died without sending results
                        exit_code = p.exitcode
                        if exit_code is not None and exit_code != 0:
                            error_msg = (
                                f"Worker on GPU {worker_gpu_ids[i]} crashed with exit code {exit_code}. "
                                f"This is likely due to OOM or another fatal error."
                            )
                            errors.append(error_msg)
                            received_from.add(worker_gpu_ids[i])
                
                # Try to get result with timeout
                try:
                    gpu_id, results, error, timing = result_queue.get(timeout=timeout_per_worker)
                    received_from.add(gpu_id)
                    
                    if timing:
                        worker_timings[gpu_id] = timing
                    
                    if error:
                        errors.append(error)
                    else:
                        # Validate result count
                        expected = expected_samples_per_worker.get(gpu_id, 0)
                        if len(results) != expected:
                            error_msg = (
                                f"Worker on GPU {gpu_id} returned {len(results)} results "
                                f"but expected {expected}. Possible partial failure (OOM mid-batch)."
                            )
                            errors.append(error_msg)
                        else:
                            all_results.extend(results)
                            
                except Exception as e:
                    # Queue.get failed - check for dead processes
                    dead_workers = []
                    for i, p in enumerate(processes):
                        if not p.is_alive() and worker_gpu_ids[i] not in received_from:
                            dead_workers.append((worker_gpu_ids[i], p.exitcode))
                            received_from.add(worker_gpu_ids[i])
                    
                    if dead_workers:
                        for gpu_id, exit_code in dead_workers:
                            error_msg = (
                                f"Worker on GPU {gpu_id} died unexpectedly "
                                f"(exit code: {exit_code}). Likely OOM or crash."
                            )
                            errors.append(error_msg)
                    else:
                        errors.append(f"Timeout or error waiting for worker results: {e}")
                    
            except Exception as e:
                errors.append(f"Unexpected error collecting results: {e}")
        
        # Print consolidated worker timing summary
        if worker_timings:
            load_times = [t['load_seconds'] for t in worker_timings.values()]
            eval_times = [t['eval_seconds'] for t in worker_timings.values()]
            gpu_ids = sorted(worker_timings.keys())
            gpu_range = f"cuda:{gpu_ids[0]}-{gpu_ids[-1]}" if len(gpu_ids) > 1 else f"cuda:{gpu_ids[0]}"
            print(f"[INFO] All {len(worker_timings)} workers on {gpu_range} finished:")
            print(f"[INFO]   Model loading: {min(load_times):.1f}s ~ {max(load_times):.1f}s")
            print(f"[INFO]   Evaluation:    {min(eval_times):.1f}s ~ {max(eval_times):.1f}s")
        
        # Wait for processes to finish and check exit codes
        for i, p in enumerate(processes):
            p.join(timeout=60)  # Give 60s for cleanup
            if p.is_alive():
                print(f"[WARNING] Worker on GPU {worker_gpu_ids[i]} did not terminate, killing...")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
            
            # Check exit code
            if p.exitcode is not None and p.exitcode != 0:
                if worker_gpu_ids[i] not in received_from:
                    error_msg = (
                        f"Worker on GPU {worker_gpu_ids[i]} exited with code {p.exitcode} "
                        f"without returning results."
                    )
                    if p.exitcode == -9:
                        error_msg += " (Killed by OOM killer)"
                    elif p.exitcode < 0:
                        import signal
                        try:
                            sig_name = signal.Signals(-p.exitcode).name
                            error_msg += f" (Signal: {sig_name})"
                        except (ValueError, AttributeError):
                            error_msg += f" (Signal: {-p.exitcode})"
                    errors.append(error_msg)
        
        # CRITICAL: Raise error if any worker failed
        if errors:
            error_summary = "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(
                f"Evaluation failed due to {len(errors)} worker error(s):\n{error_summary}\n\n"
                f"Only {len(all_results)} of {total_samples} samples were processed successfully.\n"
                f"This is likely due to OOM. Try reducing batch_size or mc_num."
            )
        
        # Validate total results
        if len(all_results) != total_samples:
            raise RuntimeError(
                f"Result count mismatch: got {len(all_results)} results "
                f"but expected {total_samples}. Some samples were lost."
            )
        
        print(f"[INFO] Successfully collected {len(all_results)} results from all workers")
        
        return all_results
    
    def _split_indices(self, total: int, num_parts: int) -> List[List[int]]:
        """Split indices as evenly as possible across parts."""
        indices = list(range(total))
        result = [[] for _ in range(num_parts)]
        
        for i, idx in enumerate(indices):
            result[i % num_parts].append(idx)
        
        return result
    
    def shutdown(self):
        """Release resources."""
        if hasattr(self, 'adapter') and self.adapter is not None:
            del self.adapter
        torch.cuda.empty_cache()


def evaluate_dataset_simple(
    config: ModelConfig,
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int = 4,
    use_multi_gpu: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function for single-use evaluation.
    
    Args:
        config: Model configuration
        dataset: Dataset to evaluate
        tokenizer: Tokenizer for collation
        batch_size: Batch size
        num_workers: DataLoader workers
        use_multi_gpu: Whether to use multi-GPU multiprocessing
        
    Returns:
        List of result dictionaries
    """
    evaluator = SimpleEvaluator(config, use_multi_gpu=use_multi_gpu)
    try:
        return evaluator.evaluate(dataset, tokenizer, batch_size, num_workers)
    finally:
        evaluator.shutdown()