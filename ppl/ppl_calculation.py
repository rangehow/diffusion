"""
PPL Calculator for Generation Outputs
Calculates PPL using a specified evaluation model.
Auto-detects GPUs and uses multi-processing for parallel evaluation.
"""

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import json
import math
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings("ignore")


# ============================================================================
# PPL CALCULATION FUNCTIONS
# ============================================================================

def calculate_ppl_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 2048,
    device: str = "cuda",
    show_progress: bool = True,
    desc: str = "Calculating PPL",
) -> Tuple[List[float], float]:
    """
    Calculate perplexity for a list of texts.
    Returns list of per-sample PPLs and average PPL.
    """
    all_ppls = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, unit="batch", leave=False)

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        try:
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(device)
        except Exception as e:
            print(f"Warning: Tokenization error: {e}")
            all_ppls.extend([float('inf')] * len(batch_texts))
            continue

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                mask = attention_mask[:, 1:].contiguous().float()

                for j in range(logits.shape[0]):
                    sample_logits = logits[j]
                    sample_labels = labels[j]
                    sample_mask = mask[j]

                    log_probs = F.log_softmax(sample_logits.float(), dim=-1)
                    token_log_probs = log_probs.gather(
                        dim=-1, 
                        index=sample_labels.unsqueeze(-1)
                    ).squeeze(-1)

                    masked_log_probs = token_log_probs * sample_mask
                    num_tokens = sample_mask.sum()

                    if num_tokens > 0:
                        avg_nll = -masked_log_probs.sum() / num_tokens
                        ppl = torch.exp(avg_nll).item()
                        ppl = min(ppl, 1e6)
                    else:
                        ppl = float('inf')

                    all_ppls.append(ppl)

            except Exception as e:
                print(f"Warning: Forward pass error: {e}")
                all_ppls.extend([float('inf')] * len(batch_texts))

    valid_ppls = [p for p in all_ppls if p != float('inf') and p < 1e6]
    avg_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float('inf')

    return all_ppls, avg_ppl


# ============================================================================
# MULTI-GPU WORKER
# ============================================================================

def gpu_worker(
    rank: int,
    ppl_model_path: str,
    texts_subset: List[str],
    indices_subset: List[int],
    batch_size: int,
    max_length: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """
    Worker process for a single GPU.
    """
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        progress_queue.put((rank, "loading", 0, len(texts_subset)))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            ppl_model_path, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        use_flash = "gpt2" not in ppl_model_path.lower()

        model = AutoModelForCausalLM.from_pretrained(
            ppl_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash else "eager",
        ).to(device).eval()

        progress_queue.put((rank, "calculating", 0, len(texts_subset)))

        # Calculate PPL
        ppls = []
        num_batches = math.ceil(len(texts_subset) / batch_size)

        for batch_idx, i in enumerate(range(0, len(texts_subset), batch_size)):
            batch_texts = texts_subset[i:i + batch_size]

            try:
                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                ).to(device)
            except Exception as e:
                ppls.extend([float('inf')] * len(batch_texts))
                continue

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            with torch.inference_mode():
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    logits = outputs.logits[:, :-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous()
                    mask = attention_mask[:, 1:].contiguous().float()

                    for j in range(logits.shape[0]):
                        sample_logits = logits[j]
                        sample_labels = labels[j]
                        sample_mask = mask[j]

                        log_probs = F.log_softmax(sample_logits.float(), dim=-1)
                        token_log_probs = log_probs.gather(
                            dim=-1, 
                            index=sample_labels.unsqueeze(-1)
                        ).squeeze(-1)

                        masked_log_probs = token_log_probs * sample_mask
                        num_tokens = sample_mask.sum()

                        if num_tokens > 0:
                            avg_nll = -masked_log_probs.sum() / num_tokens
                            ppl = torch.exp(avg_nll).item()
                            ppl = min(ppl, 1e6)
                        else:
                            ppl = float('inf')

                        ppls.append(ppl)

                except Exception as e:
                    ppls.extend([float('inf')] * len(batch_texts))

            # Update progress
            progress_queue.put((rank, "calculating", batch_idx + 1, num_batches))

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

        # Return results with original indices
        results = list(zip(indices_subset, ppls))
        result_queue.put((rank, results, None))
        progress_queue.put((rank, "done", 0, 0))

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put((rank, None, error_msg))
        progress_queue.put((rank, "error", 0, 0))


def progress_monitor(
    progress_queue: mp.Queue,
    num_gpus: int,
    stop_event: mp.Event,
):
    """Monitor and display progress from all GPU workers."""
    gpu_status = {i: ("initializing", 0, 0) for i in range(num_gpus)}
    completed = set()

    while len(completed) < num_gpus and not stop_event.is_set():
        try:
            rank, status, current, total = progress_queue.get(timeout=0.5)
            gpu_status[rank] = (status, current, total)

            if status == "done" or status == "error":
                completed.add(rank)

            # Print status line
            status_parts = []
            for gpu_id in range(num_gpus):
                s, c, t = gpu_status[gpu_id]
                if s == "loading":
                    status_parts.append(f"GPU{gpu_id}:Loading")
                elif s == "calculating":
                    pct = int(100 * c / t) if t > 0 else 0
                    status_parts.append(f"GPU{gpu_id}:{pct}%")
                elif s == "done":
                    status_parts.append(f"GPU{gpu_id}:✓")
                elif s == "error":
                    status_parts.append(f"GPU{gpu_id}:✗")
                else:
                    status_parts.append(f"GPU{gpu_id}:Init")

            print(f"\r  Progress: {' | '.join(status_parts)}", end="", flush=True)

        except:
            continue

    print()  # New line after progress


# ============================================================================
# DATA HELPERS
# ============================================================================

def split_data(data: List, num_splits: int) -> List[List]:
    """Split data into approximately equal parts."""
    k, m = divmod(len(data), num_splits)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits)]


def extract_model_name(model_path: str) -> str:
    """Extract a short model name from the path."""
    parts = model_path.rstrip("/").split("/")
    skip_names = {"main", "checkpoint", "model", "models"}

    for part in reversed(parts):
        if part.lower() not in skip_names and part:
            return part

    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def load_generations(gen_dir: str) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Load all generation files and their metadata."""
    generation_data = {}
    meta_data = {}

    for filename in os.listdir(gen_dir):
        if filename.endswith(".jsonl") and filename != "prompts.jsonl":
            method_name = filename.replace(".jsonl", "")
            filepath = os.path.join(gen_dir, filename)

            samples = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            generation_data[method_name] = samples

            meta_path = os.path.join(gen_dir, f"{method_name}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta_data[method_name] = json.load(f)
            else:
                meta_data[method_name] = {}

    return generation_data, meta_data


# ============================================================================
# MAIN CALCULATION FUNCTIONS
# ============================================================================

def calculate_ppl_multi_gpu(
    ppl_model_path: str,
    texts: List[str],
    num_gpus: int,
    batch_size: int,
    max_length: int,
) -> List[float]:
    """
    Calculate PPL using multiple GPUs in parallel.
    Returns PPL for each text in original order.
    """
    # Split texts and indices across GPUs
    indices = list(range(len(texts)))
    texts_splits = split_data(texts, num_gpus)
    indices_splits = split_data(indices, num_gpus)

    print(f"  Distributing {len(texts)} samples across {num_gpus} GPUs:")
    for i, split in enumerate(texts_splits):
        print(f"    GPU {i}: {len(split)} samples")

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    stop_event = mp.Event()

    # Start progress monitor
    monitor = mp.Process(
        target=progress_monitor,
        args=(progress_queue, num_gpus, stop_event)
    )
    monitor.start()

    # Start workers
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                rank,
                ppl_model_path,
                texts_splits[rank],
                indices_splits[rank],
                batch_size,
                max_length,
                result_queue,
                progress_queue,
            )
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    errors = []

    for _ in range(num_gpus):
        rank, results, error = result_queue.get()
        if error:
            errors.append((rank, error))
        else:
            all_results.extend(results)

    # Wait for processes
    for p in processes:
        p.join()

    # Stop monitor
    stop_event.set()
    monitor.join(timeout=2)
    if monitor.is_alive():
        monitor.terminate()

    # Report errors
    if errors:
        print("\n  ⚠️  Errors occurred:")
        for rank, error in errors:
            print(f"    GPU {rank}: {error[:100]}...")

    # Sort results by original index and extract PPLs
    all_results.sort(key=lambda x: x[0])
    ppls = [r[1] for r in all_results]

    return ppls


def calculate_ppl_single_gpu(
    ppl_model_path: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str = "cuda:0",
) -> List[float]:
    """Calculate PPL using a single GPU."""
    print(f"  Using single GPU: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        ppl_model_path, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_flash = "gpt2" not in ppl_model_path.lower()

    model = AutoModelForCausalLM.from_pretrained(
        ppl_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash else "eager",
    ).to(device).eval()

    ppls, _ = calculate_ppl_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=True,
    )

    del model, tokenizer
    torch.cuda.empty_cache()

    return ppls


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate PPL using a specified evaluation model (auto multi-GPU)"
    )

    parser.add_argument(
        "--gen_dir", 
        type=str, 
        required=True, 
        help="Directory containing generation outputs (.jsonl files)"
    )
    parser.add_argument(
        "--ppl_model_path",
        type=str,
        required=True,
        help="Path to the PPL evaluation model"
    )
    parser.add_argument(
        "--ppl_model_name",
        type=str,
        default=None,
        help="Short name for the PPL model (auto-extracted from path if not provided)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Output directory for PPL results (default: same as gen_dir)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for PPL calculation (default: 4)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all available)"
    )

    args = parser.parse_args()

    gen_dir = args.gen_dir
    output_dir = args.output_dir or gen_dir
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect GPUs
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    else:
        num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("❌ No GPUs available!")
        return

    # Extract model name
    ppl_model_name = args.ppl_model_name or extract_model_name(args.ppl_model_path)

    print("=" * 80)
    print(f"PPL CALCULATOR - {ppl_model_name}")
    print("=" * 80)
    print(f"Generation Directory: {gen_dir}")
    print(f"PPL Model: {ppl_model_name}")
    print(f"PPL Model Path: {args.ppl_model_path}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # Load generations
    print("\n📂 Loading generations...")
    generation_data, meta_data = load_generations(gen_dir)

    if not generation_data:
        print("❌ No generation files found!")
        return

    print(f"   Found {len(generation_data)} generation methods:")
    for method_name, samples in generation_data.items():
        print(f"      - {method_name}: {len(samples)} samples")

    # Calculate PPL for each method
    print(f"\n📊 Calculating PPL using {num_gpus} GPU(s)...")

    results = {}

    for method_name, samples in generation_data.items():
        print(f"\n📈 [{method_name}]")

        meta = meta_data.get(method_name, {})

        # Prepare texts
        texts = [s.get("prompt", "") + s.get("generation", "") for s in samples]

        # Calculate PPL
        if num_gpus > 1:
            ppls = calculate_ppl_multi_gpu(
                ppl_model_path=args.ppl_model_path,
                texts=texts,
                num_gpus=num_gpus,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        else:
            ppls = calculate_ppl_single_gpu(
                ppl_model_path=args.ppl_model_path,
                texts=texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

        # Calculate statistics
        valid_ppls = [p for p in ppls if p != float('inf') and p < 1e6]
        avg_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float('inf')

        results[method_name] = {
            "ppl_mean": avg_ppl,
            "ppl_std": (sum((p - avg_ppl) ** 2 for p in valid_ppls) / len(valid_ppls)) ** 0.5 if valid_ppls else 0,
            "ppl_median": sorted(valid_ppls)[len(valid_ppls) // 2] if valid_ppls else float('inf'),
            "ppl_min": min(valid_ppls) if valid_ppls else float('inf'),
            "ppl_max": max(valid_ppls) if valid_ppls else float('inf'),
            "num_valid": len(valid_ppls),
            "num_total": len(ppls),
            "generation_time_sec": meta.get("generation_time", 0),
            "tokens_per_second": meta.get("tokens_per_second", 0),
        }

        print(f"  ✓ Mean PPL: {avg_ppl:.4f} (±{results[method_name]['ppl_std']:.4f})")

    # Save results
    output_data = {
        "ppl_model_name": ppl_model_name,
        "ppl_model_path": args.ppl_model_path,
        "gen_dir": gen_dir,
        "timestamp": datetime.now().isoformat(),
        "num_gpus_used": num_gpus,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "results": results,
    }

    json_filename = f"ppl_{ppl_model_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY - {ppl_model_name}")
    print("=" * 80)
    print(f"{'Method':<25} {'PPL Mean':<12} {'PPL Std':<12} {'Gen Time(s)':<12} {'Tok/s':<10}")
    print("-" * 80)

    for method_name, data in sorted(results.items()):
        print(f"{method_name:<25} {data['ppl_mean']:<12.4f} {data['ppl_std']:<12.4f} {data['generation_time_sec']:<12.2f} {data['tokens_per_second']:<10.1f}")

    print("\n" + "=" * 80)
    print("✅ COMPLETED")
    print("=" * 80)
    print(f"   Output: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()