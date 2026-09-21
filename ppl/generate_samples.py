"""
Diffusion Generation Sampling on Full HellaSwag Validation Set
Multi-GPU version using multiprocessing (8 GPUs)
Supports running single config via --config argument
"""

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import math
import os
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Model paths
# DIFFUSION_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335"
DIFFUSION_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
LLAMA_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp"
HELLASWAG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main"

# Generation settings
NUM_GPUS = 8
MAX_NEW_TOKENS = 128
NUM_SAMPLES = None  # None = use ALL samples (~10,042)
GEN_BATCH_SIZE = 128

# Output directory
OUTPUT_DIR = "generation_outputs"

# Sampling settings
USE_GUMBEL = True
GUMBEL_TEMPERATURE = 0.7
TEMPERATURE = 0.7

# All diffusion configurations
DIFFUSION_CONFIGS = {
    "quality_cosine_b8_s8": {
        "block_size": 8,
        "steps": 8,
        "gumbel_temperature": 0.7,
        "final_temperature": 0.2,
    },
    "quality_cosine_b16_s8": {
        "block_size": 16,
        "steps": 8,
        "gumbel_temperature": 0.7,
        "final_temperature": 0.2,
    },
    "quality_cosine_b32_s8": {
        "block_size": 32,
        "steps": 8,
        "gumbel_temperature": 0.7,
        "final_temperature": 0.2,
    },
    "quality_cosine_b64_s8": {
        "block_size": 64,
        "steps": 8,
        "gumbel_temperature": 0.7,
        "final_temperature": 0.2,
    },
}


# ============================================================================
# GENERATION METHOD (QUALITY-FOCUSED WITH COSINE SCHEDULE)
# ============================================================================

def generate_diffusion_quality(
    self,
    input_ids: torch.LongTensor,
    mask_token_id: int = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 50,
    steps: int = 10,
    temperature: float = 1.0,
    block_size: int = 8,
    tokenizer=None,
    final_temperature: float = 0.1,
    min_refinement_steps: int = 2,
    use_gumbel: bool = True,
    gumbel_temperature: float = 0.7,
    **kwargs
) -> torch.LongTensor:
    """Quality-optimized with cosine temperature annealing."""
    mask_token_id = mask_token_id or self.config.mask_token_id
    batch_size, prefix_len = input_ids.shape
    device, dtype = input_ids.device, self.lm_head.weight.dtype

    def get_cosine_temperature(step: int, total_steps: int, base_temp: float, final_temp: float) -> float:
        if total_steps <= 1:
            return base_temp
        progress = step / max(total_steps - 1, 1)
        return final_temp + (base_temp - final_temp) * (1 + math.cos(math.pi * progress)) / 2

    cache = self._init_kv_cache(batch_size, prefix_len + max_new_tokens, device, dtype)

    self.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache.get_all_layer_caches(),
        cache_seqlens=cache.get_cache_seq_lens(),
        use_cache=True,
        causal=True,
        return_dict=True
    )
    cache.advance_seq_len(prefix_len)

    committed = input_ids.clone()
    committed_len = prefix_len
    generated = []
    num_generated = 0

    while num_generated < max_new_tokens:
        curr_block_size = min(block_size, max_new_tokens - num_generated)

        block = torch.full(
            (batch_size, curr_block_size),
            mask_token_id,
            dtype=input_ids.dtype,
            device=device
        )

        last_block_state = None
        effective_steps = min(steps, curr_block_size + min_refinement_steps)

        for step in range(effective_steps):
            ar_pointer = step

            if ar_pointer >= curr_block_size and step >= min_refinement_steps:
                if last_block_state is not None and (block == last_block_state).all():
                    break

            current_gumbel_temp = get_cosine_temperature(
                step, effective_steps, gumbel_temperature, final_temperature
            )

            cache.set_seq_len(committed_len - 1)
            block_input = torch.cat([committed[:, -1:], block], dim=1)

            outputs = self.forward(
                input_ids=block_input,
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seq_lens(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

            relevant_logits = outputs.logits[:, :curr_block_size, :]

            if use_gumbel and current_gumbel_temp > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, current_gumbel_temp)
                candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            elif temperature > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, temperature)
                candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            else:
                candidate_tokens = torch.argmax(relevant_logits, dim=-1).to(input_ids.dtype)

            old_block = block.clone()

            update_mask = torch.zeros(batch_size, curr_block_size, dtype=torch.bool, device=device)
            if ar_pointer < curr_block_size:
                update_mask[:, ar_pointer:] = True
            else:
                update_mask[:, -1:] = True

            block = torch.where(update_mask, candidate_tokens, old_block)
            last_block_state = old_block

            if step >= min_refinement_steps - 1 and (block == old_block).all():
                break

        cache.set_seq_len(committed_len + curr_block_size)
        committed_len += curr_block_size
        committed = torch.cat([committed, block], dim=1)
        generated.append(block)
        num_generated += curr_block_size

    return torch.cat([input_ids] + generated, dim=1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tokenize_with_bos(tokenizer, texts, device, **kwargs):
    """Tokenize without special tokens, then manually prepend BOS."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        **kwargs
    ).to(device)

    batch_size = inputs["input_ids"].shape[0]
    bos_token_id = tokenizer.bos_token_id

    bos_tokens = torch.full(
        (batch_size, 1), 
        bos_token_id, 
        dtype=inputs["input_ids"].dtype, 
        device=device
    )
    bos_mask = torch.ones(
        (batch_size, 1), 
        dtype=inputs["attention_mask"].dtype, 
        device=device
    )

    inputs["input_ids"] = torch.cat([bos_tokens, inputs["input_ids"]], dim=1)
    inputs["attention_mask"] = torch.cat([bos_mask, inputs["attention_mask"]], dim=1)

    return inputs


def split_data(data: List, num_splits: int) -> List[List]:
    """Split data into approximately equal parts."""
    k, m = divmod(len(data), num_splits)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits)]


# ============================================================================
# DIFFUSION WORKER FUNCTION
# ============================================================================

def diffusion_worker_process(
    rank: int,
    world_size: int,
    prompts_subset: List[str],
    indices_subset: List[int],
    config_name: str,
    config: Dict,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """Worker function for diffusion model generation on a single GPU."""
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        progress_queue.put(("status", rank, "loading_model", 0))

        # Load Diffusion Model
        diffusion_tokenizer = AutoTokenizer.from_pretrained(DIFFUSION_MODEL_PATH)
        if diffusion_tokenizer.pad_token is None:
            diffusion_tokenizer.pad_token = diffusion_tokenizer.eos_token
        diffusion_tokenizer.padding_side = "left"

        diffusion_model = AutoModel.from_pretrained(
            DIFFUSION_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        import types
        diffusion_model.generate_diffusion_quality = types.MethodType(
            generate_diffusion_quality, diffusion_model
        )

        mask_token_id = diffusion_model.config.mask_token_id

        progress_queue.put(("status", rank, "warming_up", 0))

        # Warmup
        if len(prompts_subset) >= 2:
            warmup_prompts = prompts_subset[:2]
        else:
            warmup_prompts = prompts_subset[:1] * 2

        warmup_inputs = diffusion_tokenizer(
            warmup_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2560
        ).to(device)

        with torch.inference_mode():
            _ = diffusion_model.generate_diffusion_quality(
                input_ids=warmup_inputs["input_ids"],
                mask_token_id=mask_token_id,
                max_new_tokens=16,
                block_size=config["block_size"],
                steps=config["steps"],
                use_gumbel=USE_GUMBEL,
                gumbel_temperature=config["gumbel_temperature"],
            )

        torch.cuda.synchronize()

        progress_queue.put(("status", rank, f"generating_{config_name}", 0))

        # Generate
        results = []
        total_time = 0
        num_batches = math.ceil(len(prompts_subset) / GEN_BATCH_SIZE)

        for batch_idx, i in enumerate(range(0, len(prompts_subset), GEN_BATCH_SIZE)):
            batch_prompts = prompts_subset[i:i + GEN_BATCH_SIZE]
            batch_indices = indices_subset[i:i + GEN_BATCH_SIZE]

            # inputs = diffusion_tokenizer(
            #     batch_prompts,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=2560,
            #     add_special_tokens=False
            # ).to(device)

            inputs = tokenize_with_bos(
                diffusion_tokenizer, batch_prompts, device,
                padding=True, truncation=True, max_length=2560
            )


            prompt_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                output_ids = diffusion_model.generate_diffusion_block_kvcache(
                    input_ids=inputs["input_ids"],
                    mask_token_id=mask_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    block_size=config["block_size"],
                    steps=config["steps"],
                    temperature=TEMPERATURE,
                    use_gumbel=USE_GUMBEL,
                    gumbel_temperature=config["gumbel_temperature"],
                    final_temperature=config["final_temperature"],
                )

            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            total_time += batch_time

            generated_ids = output_ids[:, prompt_len:]
            generations = diffusion_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, gen in zip(batch_indices, generations):
                results.append({"id": idx, "generation": gen})

            progress_queue.put(("progress", rank, config_name, batch_idx + 1, num_batches))

        del diffusion_model
        torch.cuda.empty_cache()

        progress_queue.put(("done", rank, None, 0))
        result_queue.put((rank, {
            "results": results,
            "time": total_time,
            "num_samples": len(prompts_subset),
        }))

    except Exception as e:
        import traceback
        error_msg = f"[GPU {rank}] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        progress_queue.put(("error", rank, str(e), 0))
        result_queue.put((rank, {"error": error_msg}))


# ============================================================================
# LLAMA WORKER FUNCTION
# ============================================================================

def llama_worker_process(
    rank: int,
    world_size: int,
    prompts_subset: List[str],
    indices_subset: List[int],
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """Worker function for Llama model generation on a single GPU."""
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        progress_queue.put(("status", rank, "loading_model", 0))

        # Load Llama Model
        llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.padding_side = "left"

        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        progress_queue.put(("status", rank, "warming_up", 0))

        # Warmup
        if len(prompts_subset) >= 2:
            warmup_prompts = prompts_subset[:2]
        else:
            warmup_prompts = prompts_subset[:1] * 2

        llama_warmup = tokenize_with_bos(
            llama_tokenizer, warmup_prompts, device,
            padding=True, truncation=True, max_length=2560
        )
        with torch.inference_mode():
            _ = llama_model.generate(
                **llama_warmup,
                max_new_tokens=16,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=llama_tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()

        progress_queue.put(("status", rank, "generating_llama", 0))

        # Generate
        results = []
        total_time = 0
        num_batches = math.ceil(len(prompts_subset) / GEN_BATCH_SIZE)

        for batch_idx, i in enumerate(range(0, len(prompts_subset), GEN_BATCH_SIZE)):
            batch_prompts = prompts_subset[i:i + GEN_BATCH_SIZE]
            batch_indices = indices_subset[i:i + GEN_BATCH_SIZE]

            inputs = tokenize_with_bos(
                llama_tokenizer,
                batch_prompts,
                device,
                padding=True,
                truncation=True,
                max_length=2560
            )

            prompt_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                output_ids = llama_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=llama_tokenizer.pad_token_id,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            total_time += batch_time

            generated_ids = output_ids[:, prompt_len:]
            generations = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, gen in zip(batch_indices, generations):
                results.append({"id": idx, "generation": gen})

            progress_queue.put(("progress", rank, "llama", batch_idx + 1, num_batches))

        del llama_model
        torch.cuda.empty_cache()

        progress_queue.put(("done", rank, None, 0))
        result_queue.put((rank, {
            "results": results,
            "time": total_time,
            "num_samples": len(prompts_subset),
        }))

    except Exception as e:
        import traceback
        error_msg = f"[GPU {rank}] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        progress_queue.put(("error", rank, str(e), 0))
        result_queue.put((rank, {"error": error_msg}))


# ============================================================================
# PROGRESS MONITOR FUNCTION
# ============================================================================

def progress_monitor(
    progress_queue: mp.Queue,
    num_gpus: int,
    batches_per_gpu: int,
    config_name: str,
    stop_event: mp.Event,
):
    """Monitor progress from all workers and display progress bars."""
    gpu_status = {i: "initializing" for i in range(num_gpus)}
    gpu_progress = {i: 0 for i in range(num_gpus)}
    completed_gpus = set()

    total_work = num_gpus * batches_per_gpu

    pbar = tqdm(
        total=total_work,
        desc=f"[{config_name}] Overall",
        unit="batch",
        position=0,
        leave=True,
        ncols=100,
    )

    gpu_bars = {}
    for i in range(num_gpus):
        gpu_bars[i] = tqdm(
            total=100,
            desc=f"GPU {i}",
            position=i + 1,
            leave=True,
            bar_format="{desc}: {postfix}",
            ncols=100,
        )
        gpu_bars[i].set_postfix_str("Initializing...")

    try:
        while len(completed_gpus) < num_gpus and not stop_event.is_set():
            try:
                msg = progress_queue.get(timeout=0.5)
            except:
                continue

            msg_type = msg[0]
            rank = msg[1]

            if msg_type == "status":
                status = msg[2]
                gpu_status[rank] = status
                status_display = status.replace("_", " ").title()
                gpu_bars[rank].set_postfix_str(status_display)

            elif msg_type == "progress":
                current = msg[3]
                total = msg[4]

                old_progress = gpu_progress[rank]
                gpu_progress[rank] = current

                delta = current - old_progress
                pbar.update(delta)

                percent = int(100 * current / total) if total > 0 else 0
                gpu_bars[rank].set_postfix_str(f"{current}/{total} ({percent}%)")

            elif msg_type == "done":
                completed_gpus.add(rank)
                gpu_bars[rank].set_postfix_str("✓ Completed")

            elif msg_type == "error":
                error_msg = msg[2]
                completed_gpus.add(rank)
                gpu_bars[rank].set_postfix_str(f"✗ Error: {error_msg[:30]}...")

    finally:
        pbar.close()
        for bar in gpu_bars.values():
            bar.close()


# ============================================================================
# RESULT AGGREGATION AND SAVING
# ============================================================================

def save_results(
    all_results: Dict[int, Dict],
    prompts: List[str],
    config_name: str,
    config: Optional[Dict],
    output_dir: str,
):
    """Aggregate results from all workers and save to files."""
    all_method_results = []
    total_time = 0
    total_samples = 0

    for rank, worker_result in all_results.items():
        if "error" in worker_result:
            print(f"Warning: GPU {rank} had error, skipping its results")
            continue

        all_method_results.extend(worker_result["results"])
        total_time += worker_result["time"]
        total_samples += worker_result["num_samples"]

    # Sort by index
    all_method_results.sort(key=lambda x: x["id"])

    tokens_per_second = (total_samples * MAX_NEW_TOKENS) / total_time if total_time > 0 else 0

    # Save generations
    gen_path = os.path.join(output_dir, f"{config_name}.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for result in all_method_results:
            idx = result["id"]
            entry = {
                "id": idx,
                "prompt": prompts[idx],
                "generation": result["generation"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save metadata
    if config_name == "llama":
        config_info = {"method": "autoregressive", "temperature": TEMPERATURE}
    else:
        config_info = config

    meta_path = os.path.join(output_dir, f"{config_name}_meta.json")
    meta = {
        "method_name": config_name,
        "config": config_info,
        "num_samples": len(all_method_results),
        "generation_time": total_time,
        "tokens_per_second": tokens_per_second,
        "timestamp": datetime.now().isoformat(),
        "num_gpus": len(all_results),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved {config_name}.jsonl ({len(all_method_results)} samples)")
    print(f"✓ Saved {config_name}_meta.json")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Tokens/second: {tokens_per_second:.2f}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate samples with diffusion or llama model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(DIFFUSION_CONFIGS.keys()) + ["llama", "all"],
        help="Configuration name to run"
    )
    args = parser.parse_args()

    config_name = args.config
    is_llama = (config_name == "llama")
    run_all = (config_name == "all")

    # Create experiment name based on config
    EXPERIMENT_NAME = f"hellaswag_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_output_dir = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
    os.makedirs(exp_output_dir, exist_ok=True)

    print("=" * 80)
    print(f"GENERATION EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Config: {config_name}")
    print(f"Using {NUM_GPUS} GPUs")
    print(f"Output directory: {exp_output_dir}")
    print("=" * 80)

    if not is_llama and not run_all:
        config = DIFFUSION_CONFIGS[config_name]
        print(f"\n📊 Configuration: block_size={config['block_size']}, steps={config['steps']}")

    # ==================== Load Dataset ====================
    print("\n📊 Loading HellaSwag dataset...")
    dataset = load_dataset(HELLASWAG_PATH)
    validation = dataset["validation"]

    total_samples = len(validation)
    print(f"   Total validation samples available: {total_samples}")

    if NUM_SAMPLES is None or NUM_SAMPLES > total_samples:
        prompts = list(validation["ctx"])
        print(f"   Using ALL {len(prompts)} samples")
    else:
        prompts = list(validation["ctx"][:NUM_SAMPLES])
        print(f"   Using {len(prompts)} samples")

    indices = list(range(len(prompts)))

    # Save experiment config
    experiment_config = {
        "config_name": config_name,
        "diffusion_model_path": DIFFUSION_MODEL_PATH if not is_llama else None,
        "llama_model_path": LLAMA_MODEL_PATH if is_llama else None,
        "hellaswag_path": HELLASWAG_PATH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_samples": len(prompts),
        "gen_batch_size": GEN_BATCH_SIZE,
        "temperature": TEMPERATURE,
        "num_gpus": NUM_GPUS,
        "timestamp": datetime.now().isoformat(),
    }

    if not is_llama and not run_all:
        experiment_config["diffusion_config"] = DIFFUSION_CONFIGS[config_name]
        experiment_config["use_gumbel"] = USE_GUMBEL
        experiment_config["gumbel_temperature"] = GUMBEL_TEMPERATURE

    config_path = os.path.join(exp_output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"✓ Saved experiment_config.json")

    # Save prompts
    prompts_path = os.path.join(exp_output_dir, "prompts.jsonl")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"id": i, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"✓ Saved prompts.jsonl ({len(prompts)} prompts)")

    # ==================== Split Data for Each GPU ====================
    print(f"\n📦 Splitting data across {NUM_GPUS} GPUs...")
    prompts_splits = split_data(prompts, NUM_GPUS)
    indices_splits = split_data(indices, NUM_GPUS)

    for i, split in enumerate(prompts_splits):
        print(f"   GPU {i}: {len(split)} samples (indices {indices_splits[i][0]}-{indices_splits[i][-1]})")

    # ==================== Launch Worker Processes ====================
    print(f"\n🚀 Launching {NUM_GPUS} worker processes for {config_name}...")

    mp.set_start_method('spawn', force=True)

    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    stop_event = mp.Event()

    batches_per_gpu = math.ceil(len(prompts_splits[0]) / GEN_BATCH_SIZE)

    # Start progress monitor
    monitor_process = mp.Process(
        target=progress_monitor,
        args=(progress_queue, NUM_GPUS, batches_per_gpu, config_name, stop_event)
    )
    monitor_process.start()

    # Start workers
    processes = []
    for rank in range(NUM_GPUS):
        if is_llama:
            p = mp.Process(
                target=llama_worker_process,
                args=(
                    rank,
                    NUM_GPUS,
                    prompts_splits[rank],
                    indices_splits[rank],
                    result_queue,
                    progress_queue,
                )
            )
        else:
            p = mp.Process(
                target=diffusion_worker_process,
                args=(
                    rank,
                    NUM_GPUS,
                    prompts_splits[rank],
                    indices_splits[rank],
                    config_name,
                    DIFFUSION_CONFIGS[config_name],
                    result_queue,
                    progress_queue,
                )
            )
        p.start()
        processes.append(p)

    print("\n")

    # Collect results
    all_results = {}
    for _ in range(NUM_GPUS):
        rank, results = result_queue.get()
        all_results[rank] = results

    # Wait for workers
    for p in processes:
        p.join()

    # Stop monitor
    stop_event.set()
    monitor_process.join(timeout=2)
    if monitor_process.is_alive():
        monitor_process.terminate()

    print("\n" * (NUM_GPUS + 2))
    print("✓ All workers completed!")

    # ==================== Save Results ====================
    print("\n📁 Saving results...")

    if is_llama:
        save_results(all_results, prompts, config_name, None, exp_output_dir)
    else:
        save_results(
            all_results, prompts, config_name,
            DIFFUSION_CONFIGS[config_name], exp_output_dir
        )

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(f"GENERATION COMPLETE: {config_name}")
    print("=" * 80)
    print(f"   Output directory: {exp_output_dir}")
    print(f"   Total samples: {len(prompts)}")
    print(f"   GPUs used: {NUM_GPUS}")
    print(f"\n   Files saved:")
    print(f"      - experiment_config.json")
    print(f"      - prompts.jsonl")
    print(f"      - {config_name}.jsonl")
    print(f"      - {config_name}_meta.json")
    print("\n   Next step: Run calculate_ppl.py to evaluate quality")
    print("=" * 80)


if __name__ == "__main__":
    main()