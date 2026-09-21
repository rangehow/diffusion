"""
Diffusion Generation Evaluation with PPL on Full HellaSwag Validation Set
Uses Qwen3-0.6B-Base to calculate perplexity of generated text.
Compares different sampling parameters and algorithms.
Saves all results to structured output folder.
"""

import torch
import torch.nn.functional as F
import time
import math
import os
import json
import csv
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# GENERATION METHODS (ALIGNED SAMPLING)
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
    temperature_schedule: str = "cosine",
    final_temperature: float = 0.1,
    use_confidence_masking: bool = False,
    confidence_threshold: float = 0.9,
    min_refinement_steps: int = 2,
    use_gumbel: bool = True,
    gumbel_temperature: float = 0.7,
    **kwargs
) -> torch.LongTensor:
    """Quality-optimized with temperature annealing and early stopping."""
    mask_token_id = mask_token_id or self.config.mask_token_id
    batch_size, prefix_len = input_ids.shape
    device, dtype = input_ids.device, self.lm_head.weight.dtype

    def get_scheduled_temperature(step: int, total_steps: int, base_temp: float, final_temp: float) -> float:
        if temperature_schedule == "constant" or total_steps <= 1:
            return base_temp
        progress = step / max(total_steps - 1, 1)
        if temperature_schedule == "linear":
            return base_temp + (final_temp - base_temp) * progress
        elif temperature_schedule == "cosine":
            return final_temp + (base_temp - final_temp) * (1 + math.cos(math.pi * progress)) / 2
        return base_temp

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

        block_confidence = torch.zeros(batch_size, curr_block_size, device=device)
        converged = False
        last_block_state = None

        effective_steps = min(steps, curr_block_size + min_refinement_steps)

        for step in range(effective_steps):
            ar_pointer = step

            if ar_pointer >= curr_block_size and step >= min_refinement_steps:
                converged = (block == last_block_state).all() if last_block_state is not None else False
                break

            current_gumbel_temp = get_scheduled_temperature(
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

            if use_confidence_masking:
                probs = F.softmax(relevant_logits.float(), dim=-1)
                new_confidence, _ = probs.max(dim=-1)
            else:
                new_confidence = torch.ones(batch_size, curr_block_size, device=device)

            old_block = block.clone()

            if use_confidence_masking and step > 0:
                update_mask = block_confidence < confidence_threshold
                update_mask[:, min(ar_pointer, curr_block_size - 1):] = True
            else:
                update_mask = torch.zeros(batch_size, curr_block_size, dtype=torch.bool, device=device)
                if ar_pointer < curr_block_size:
                    update_mask[:, ar_pointer:] = True
                else:
                    update_mask[:, -1:] = True

            block = torch.where(update_mask, candidate_tokens, old_block)
            block_confidence = torch.where(update_mask, new_confidence, block_confidence)
            last_block_state = old_block

            no_changes = (block == old_block).all()
            all_confident = (block_confidence >= confidence_threshold).all() if use_confidence_masking else False

            if step >= min_refinement_steps - 1:
                if no_changes:
                    converged = True
                    break
                if all_confident and use_confidence_masking:
                    converged = True
                    break

        if not converged:
            cache.set_seq_len(committed_len - 1)
            self.forward(
                input_ids=torch.cat([committed[:, -1:], block], dim=1),
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seq_lens(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

        cache.set_seq_len(committed_len + curr_block_size)
        committed_len += curr_block_size
        committed = torch.cat([committed, block], dim=1)
        generated.append(block)
        num_generated += curr_block_size

    return torch.cat([input_ids] + generated, dim=1)


def generate_diffusion_fast(
    self,
    input_ids,
    mask_token_id: int = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 50,
    steps: int = 8,
    temperature: float = 0.7,
    block_size: int = 16,
    tokenizer=None,
    use_gumbel: bool = True,
    gumbel_temperature: float = 0.7,
    final_gumbel_temperature: float = 0.1,
    **kwargs
) -> torch.LongTensor:
    """Fast version with larger blocks and fewer steps."""
    mask_token_id = mask_token_id or self.config.mask_token_id
    batch_size, prefix_len = input_ids.shape
    device, dtype = input_ids.device, self.lm_head.weight.dtype

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

        prev_block = None
        converged = False
        actual_steps = min(steps, curr_block_size)

        for step in range(actual_steps):
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

            progress = step / max(actual_steps - 1, 1)
            current_temp = gumbel_temperature + (final_gumbel_temperature - gumbel_temperature) * progress

            if use_gumbel and current_temp > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, current_temp)
                candidate_tokens = gumbel_logits.argmax(dim=-1).to(input_ids.dtype)
            elif temperature > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, temperature)
                candidate_tokens = gumbel_logits.argmax(dim=-1).to(input_ids.dtype)
            else:
                candidate_tokens = relevant_logits.argmax(dim=-1).to(input_ids.dtype)

            prev_block = block.clone()

            update_mask = torch.zeros(batch_size, curr_block_size, dtype=torch.bool, device=device)
            update_mask[:, step:] = True
            block = torch.where(update_mask, candidate_tokens, block)

            if step > 0 and (block == prev_block).all():
                converged = True
                break

        if not converged:
            cache.set_seq_len(committed_len - 1)
            self.forward(
                input_ids=torch.cat([committed[:, -1:], block], dim=1),
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seq_lens(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

        cache.set_seq_len(committed_len + curr_block_size)
        committed_len += curr_block_size
        committed = torch.cat([committed, block], dim=1)
        generated.append(block)
        num_generated += curr_block_size

    return torch.cat([input_ids] + generated, dim=1)


# ============================================================================
# PPL CALCULATION
# ============================================================================

def calculate_ppl_batch(
    eval_model,
    eval_tokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda"
) -> Tuple[float, List[float]]:
    """
    Calculate perplexity of texts using the evaluation model.
    Returns (average_ppl, list_of_individual_ppls).
    """
    eval_model.eval()
    individual_ppls = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Filter out empty texts
        valid_texts = [t for t in batch_texts if t and len(t.strip()) > 0]
        if not valid_texts:
            individual_ppls.extend([float('inf')] * len(batch_texts))
            continue

        inputs = eval_tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        ).to(device)

        with torch.inference_mode():
            outputs = eval_model(**inputs)
            logits = outputs.logits

        # Calculate per-sample PPL
        for j in range(len(valid_texts)):
            sample_ids = inputs["input_ids"][j]
            sample_mask = inputs["attention_mask"][j]
            sample_logits = logits[j]

            actual_len = sample_mask.sum().item()
            if actual_len <= 1:
                individual_ppls.append(float('inf'))
                continue

            shift_logits = sample_logits[:-1, :]
            shift_labels = sample_ids[1:]
            shift_mask = sample_mask[1:]

            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                reduction='none'
            )

            masked_loss = loss * shift_mask.float()
            avg_loss = masked_loss.sum() / shift_mask.sum()

            ppl = torch.exp(avg_loss).item()
            individual_ppls.append(ppl if not math.isnan(ppl) and not math.isinf(ppl) else float('inf'))

        num_filtered = len(batch_texts) - len(valid_texts)
        individual_ppls.extend([float('inf')] * num_filtered)

    valid_ppls = [p for p in individual_ppls if p != float('inf') and p < 1e6]
    avg_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float('inf')

    return avg_ppl, individual_ppls


def calculate_ppl_for_generations(
    eval_model,
    eval_tokenizer,
    prompts: List[str],
    generations: List[str],
    batch_size: int = 4,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Calculate PPL for generated text (continuation only) and full text (prompt + generation).
    """
    full_texts = [p + g for p, g in zip(prompts, generations)]
    full_ppl, full_individual = calculate_ppl_batch(
        eval_model, eval_tokenizer, full_texts, batch_size, device=device
    )

    gen_ppl, gen_individual = calculate_ppl_batch(
        eval_model, eval_tokenizer, generations, batch_size, device=device
    )

    return {
        "full_ppl": full_ppl,
        "gen_ppl": gen_ppl,
        "full_individual": full_individual,
        "gen_individual": gen_individual,
    }


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


# ============================================================================
# SAVE RESULTS TO FOLDER
# ============================================================================

def save_results_to_folder(
    results: Dict,
    prompts: List[str],
    output_dir: str = "generation_outputs",
    experiment_name: str = None
) -> str:
    """
    Save all generation results to a structured folder.

    Structure:
        output_dir/
        └── experiment_name/
            ├── summary.json
            ├── config.json
            ├── all_generations.csv
            ├── generations/
            │   ├── original.jsonl
            │   ├── quality_cosine.jsonl
            │   └── ...
            └── samples/
                ├── sample_0000.txt
                ├── sample_0001.txt
                └── ...
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(output_dir, experiment_name)
    gen_dir = os.path.join(exp_dir, "generations")
    sample_dir = os.path.join(exp_dir, "samples")

    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print(f"\n📁 Saving results to: {exp_dir}")

    # 1. Save summary statistics
    summary = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(prompts),
        "methods": {}
    }

    for method, r in results.items():
        summary["methods"][method] = {
            "config": r["config"],
            "time": r["time"],
            "tokens_per_second": r["tokens_per_second"],
            "full_ppl": r["full_ppl"],
            "gen_ppl": r["gen_ppl"],
        }

    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("   ✓ summary.json")

    # 2. Save generations per method (JSONL format)
    for method, r in results.items():
        jsonl_path = os.path.join(gen_dir, f"{method}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, (prompt, gen) in enumerate(zip(prompts, r["generations"])):
                entry = {
                    "id": i,
                    "prompt": prompt,
                    "generation": gen,
                    "full_ppl": r["full_individual"][i],
                    "gen_ppl": r["gen_individual"][i],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   ✓ generations/{method}.jsonl")

    # 3. Save side-by-side comparison samples (first 100)
    num_samples_to_save = min(100, len(prompts))
    for i in range(num_samples_to_save):
        sample_path = os.path.join(sample_dir, f"sample_{i:04d}.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"SAMPLE {i}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"PROMPT:\n{prompts[i]}\n\n")
            f.write(f"{'-'*80}\n")

            for method in sorted(results.keys()):
                r = results[method]
                gen = r["generations"][i]
                ppl = r["gen_individual"][i]
                f.write(f"\n[{method.upper()}] (Gen PPL: {ppl:.2f})\n")
                f.write(f"{gen}\n")
                f.write(f"{'-'*40}\n")
    print(f"   ✓ samples/ ({num_samples_to_save} files)")

    # 4. Save all generations as CSV for easy analysis
    csv_path = os.path.join(exp_dir, "all_generations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        methods = list(results.keys())
        fieldnames = ["id", "prompt"] + [f"{m}_generation" for m in methods] + [f"{m}_gen_ppl" for m in methods] + [f"{m}_full_ppl" for m in methods]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(prompts)):
            row = {"id": i, "prompt": prompts[i]}
            for m in methods:
                row[f"{m}_generation"] = results[m]["generations"][i]
                row[f"{m}_gen_ppl"] = results[m]["gen_individual"][i]
                row[f"{m}_full_ppl"] = results[m]["full_individual"][i]
            writer.writerow(row)
    print(f"   ✓ all_generations.csv")

    # 5. Save PPL statistics
    stats_path = os.path.join(exp_dir, "ppl_statistics.json")
    ppl_stats = {}
    for method, r in results.items():
        valid_gen_ppls = [p for p in r["gen_individual"] if p != float('inf') and p < 1e6]
        valid_full_ppls = [p for p in r["full_individual"] if p != float('inf') and p < 1e6]

        ppl_stats[method] = {
            "gen_ppl_mean": sum(valid_gen_ppls) / len(valid_gen_ppls) if valid_gen_ppls else float('inf'),
            "gen_ppl_median": sorted(valid_gen_ppls)[len(valid_gen_ppls)//2] if valid_gen_ppls else float('inf'),
            "gen_ppl_min": min(valid_gen_ppls) if valid_gen_ppls else float('inf'),
            "gen_ppl_max": max(valid_gen_ppls) if valid_gen_ppls else float('inf'),
            "full_ppl_mean": sum(valid_full_ppls) / len(valid_full_ppls) if valid_full_ppls else float('inf'),
            "full_ppl_median": sorted(valid_full_ppls)[len(valid_full_ppls)//2] if valid_full_ppls else float('inf'),
            "num_valid_samples": len(valid_gen_ppls),
            "num_inf_samples": len(r["gen_individual"]) - len(valid_gen_ppls),
        }

    with open(stats_path, "w") as f:
        json.dump(ppl_stats, f, indent=2)
    print(f"   ✓ ppl_statistics.json")

    return exp_dir


def save_checkpoint(
    results: Dict,
    prompts: List[str],
    checkpoint_dir: str,
    method_name: str
):
    """Save intermediate checkpoint for a single method."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{method_name}_checkpoint.json")

    checkpoint_data = {
        "method": method_name,
        "config": results[method_name]["config"],
        "time": results[method_name]["time"],
        "tokens_per_second": results[method_name]["tokens_per_second"],
        "full_ppl": results[method_name]["full_ppl"],
        "gen_ppl": results[method_name]["gen_ppl"],
        "num_samples": len(results[method_name]["generations"]),
    }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    # Also save generations
    gen_path = os.path.join(checkpoint_dir, f"{method_name}_generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for i, (prompt, gen) in enumerate(zip(prompts, results[method_name]["generations"])):
            entry = {
                "id": i,
                "prompt": prompt,
                "generation": gen,
                "full_ppl": results[method_name]["full_individual"][i],
                "gen_ppl": results[method_name]["gen_individual"][i],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    # ==================== Configuration ====================
    DIFFUSION_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335"
    LLAMA_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp"
    EVAL_MODEL_PATH = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/shxs/model/Qwen/Qwen3-0.6B-Base/main"
    HELLASWAG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main"

    DEVICE = "cuda"
    MAX_NEW_TOKENS = 64

    # ⭐ USE ALL VALIDATION SAMPLES
    NUM_SAMPLES = None  # None = use ALL samples (~10,042)

    EVAL_BATCH_SIZE = 16  # Increased for faster PPL calculation
    GEN_BATCH_SIZE = 8    # Batch size for generation

    # Output configuration
    OUTPUT_DIR = "generation_outputs"
    CHECKPOINT_DIR = "checkpoints"
    SAVE_CHECKPOINTS = True  # Save after each method completes

    # Shared sampling parameters
    USE_GUMBEL = True
    GUMBEL_TEMPERATURE = 0.7
    TEMPERATURE = 0.7
    CONFIDENCE_THRESHOLD = 0.9

    # Method configurations to test
    CONFIGS = {
        "original": {
            "method": "original",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
        },
        "quality_cosine": {
            "method": "quality",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
            "temperature_schedule": "cosine",
            "final_temperature": 0.2,
            "use_confidence_masking": False,
        },
        "quality_linear": {
            "method": "quality",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
            "temperature_schedule": "linear",
            "final_temperature": 0.2,
            "use_confidence_masking": False,
        },
        "quality_constant": {
            "method": "quality",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
            "temperature_schedule": "constant",
            "final_temperature": 0.7,
            "use_confidence_masking": False,
        },
        "fast_b32_s8": {
            "method": "fast",
            "block_size": 32,
            "steps": 8,
            "gumbel_temperature": 0.7,
            "final_gumbel_temperature": 0.2,
        },
        "fast_b16_s12": {
            "method": "fast",
            "block_size": 16,
            "steps": 12,
            "gumbel_temperature": 0.7,
            "final_gumbel_temperature": 0.2,
        },
        "low_temp": {
            "method": "original",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.3,
        },
        "high_temp": {
            "method": "original",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 1.0,
        },
    }

    # ==================== Load Dataset ====================
    print("=" * 80)
    print("Loading HellaSwag dataset...")
    print("=" * 80)

    dataset = load_dataset(HELLASWAG_PATH)
    validation = dataset["validation"]

    # Show actual dataset size
    total_samples = len(validation)
    print(f"📊 Total validation samples available: {total_samples}")

    # Select samples
    if NUM_SAMPLES is None or NUM_SAMPLES > total_samples:
        prompts = list(validation["ctx"])
        print(f"📊 Using ALL {len(prompts)} samples for evaluation")
    else:
        prompts = list(validation["ctx"][:NUM_SAMPLES])
        print(f"📊 Using {len(prompts)} samples for evaluation")

    print(f"Example prompt: {prompts[0][:100]}...")

    # Estimate time
    estimated_time_per_sample = 0.5  # seconds (rough estimate)
    num_methods = len(CONFIGS) + 1  # +1 for Llama
    estimated_total_time = len(prompts) * estimated_time_per_sample * num_methods / 60
    print(f"⏱️  Estimated total time: ~{estimated_total_time:.0f} minutes")

    # ==================== Load Evaluation Model ====================
    print("\n" + "=" * 80)
    print("Loading evaluation model (Qwen3-0.6B-Base)...")
    print("=" * 80)

    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_PATH, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        EVAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    print("✓ Evaluation model loaded.")

    # ==================== Load Diffusion Model ====================
    print("\n" + "=" * 80)
    print("Loading Diffusion model...")
    print("=" * 80)

    diffusion_tokenizer = AutoTokenizer.from_pretrained(DIFFUSION_MODEL_PATH)
    if diffusion_tokenizer.pad_token is None:
        diffusion_tokenizer.pad_token = diffusion_tokenizer.eos_token

    diffusion_model = AutoModel.from_pretrained(
        DIFFUSION_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(DEVICE).eval()

    # Monkey-patch new generation methods
    import types
    diffusion_model.generate_diffusion_quality = types.MethodType(generate_diffusion_quality, diffusion_model)
    diffusion_model.generate_diffusion_fast = types.MethodType(generate_diffusion_fast, diffusion_model)

    mask_token_id = diffusion_model.config.mask_token_id
    print(f"✓ Diffusion model loaded. Mask token ID: {mask_token_id}")

    # ==================== Load Llama Model ====================
    print("\n" + "=" * 80)
    print("Loading Llama model...")
    print("=" * 80)

    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(DEVICE).eval()

    print("✓ Llama model loaded.")

    # ==================== Warmup ====================
    print("\n" + "=" * 80)
    print("Warming up models...")
    print("=" * 80)

    warmup_inputs = diffusion_tokenizer(
        prompts[:2], return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(DEVICE)

    with torch.inference_mode():
        _ = diffusion_model.generate_diffusion_block_kvcache(
            input_ids=warmup_inputs["input_ids"],
            mask_token_id=mask_token_id,
            max_new_tokens=16,
            block_size=16,
            steps=8,
            use_gumbel=USE_GUMBEL,
            gumbel_temperature=GUMBEL_TEMPERATURE,
        )
        _ = diffusion_model.generate_diffusion_quality(
            input_ids=warmup_inputs["input_ids"],
            mask_token_id=mask_token_id,
            max_new_tokens=16,
            block_size=16,
            steps=8,
            use_gumbel=USE_GUMBEL,
            gumbel_temperature=GUMBEL_TEMPERATURE,
        )
        _ = diffusion_model.generate_diffusion_fast(
            input_ids=warmup_inputs["input_ids"],
            mask_token_id=mask_token_id,
            max_new_tokens=16,
            block_size=16,
            steps=8,
            use_gumbel=USE_GUMBEL,
            gumbel_temperature=GUMBEL_TEMPERATURE,
        )

    llama_warmup = tokenize_with_bos(
        llama_tokenizer,
        prompts[:2],
        DEVICE,
        padding=True,
        truncation=True,
        max_length=256
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
    print("✓ Warmup done.")

    # ==================== Results Storage ====================
    results = {}

    # ==================== Generate with Diffusion Configs ====================
    print("\n" + "=" * 80)
    print(f"Running Diffusion generation experiments ({len(CONFIGS)} configurations)...")
    print("=" * 80)

    for config_idx, (config_name, config) in enumerate(CONFIGS.items()):
        print(f"\n[{config_idx + 1}/{len(CONFIGS)}] --- {config_name} ---")
        print(f"    Config: {config}")

        all_generations = []
        total_time = 0

        # Process in batches with progress bar
        pbar = tqdm(
            range(0, len(prompts), GEN_BATCH_SIZE), 
            desc=f"Generating [{config_name}]",
            total=(len(prompts) + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE
        )

        for i in pbar:
            batch_prompts = prompts[i:i + GEN_BATCH_SIZE]

            inputs = diffusion_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(DEVICE)

            prompt_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                if config["method"] == "original":
                    output_ids = diffusion_model.generate_diffusion_block_kvcache(
                        input_ids=inputs["input_ids"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=TEMPERATURE,
                        use_gumbel=USE_GUMBEL,
                        gumbel_temperature=config["gumbel_temperature"],
                    )
                elif config["method"] == "quality":
                    output_ids = diffusion_model.generate_diffusion_quality(
                        input_ids=inputs["input_ids"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=TEMPERATURE,
                        use_gumbel=USE_GUMBEL,
                        gumbel_temperature=config["gumbel_temperature"],
                        temperature_schedule=config.get("temperature_schedule", "constant"),
                        final_temperature=config.get("final_temperature", 0.2),
                        use_confidence_masking=config.get("use_confidence_masking", False),
                    )
                elif config["method"] == "fast":
                    output_ids = diffusion_model.generate_diffusion_fast(
                        input_ids=inputs["input_ids"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=TEMPERATURE,
                        use_gumbel=USE_GUMBEL,
                        gumbel_temperature=config["gumbel_temperature"],
                        final_gumbel_temperature=config.get("final_gumbel_temperature", 0.2),
                    )

            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            total_time += batch_time

            # Decode generations only (not prompt)
            generated_ids = output_ids[:, prompt_len:]
            generations = diffusion_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generations.extend(generations)

            # Update progress bar with speed
            tokens_generated = len(all_generations) * MAX_NEW_TOKENS
            pbar.set_postfix({
                "tok/s": f"{tokens_generated / total_time:.1f}",
                "samples": len(all_generations)
            })

        # Calculate PPL
        print(f"    Calculating PPL for {len(all_generations)} generations...")
        ppl_results = calculate_ppl_for_generations(
            eval_model, eval_tokenizer, prompts, all_generations, EVAL_BATCH_SIZE, DEVICE
        )

        results[config_name] = {
            "config": config,
            "generations": all_generations,
            "time": total_time,
            "tokens_per_second": (len(prompts) * MAX_NEW_TOKENS) / total_time,
            **ppl_results,
        }

        print(f"    ✓ Time: {total_time:.2f}s | Tok/s: {results[config_name]['tokens_per_second']:.1f} | Full PPL: {ppl_results['full_ppl']:.2f} | Gen PPL: {ppl_results['gen_ppl']:.2f}")

        # Save checkpoint
        if SAVE_CHECKPOINTS:
            save_checkpoint(results, prompts, CHECKPOINT_DIR, config_name)
            print(f"    ✓ Checkpoint saved")

    # ==================== Generate with Llama ====================
    print("\n" + "=" * 80)
    print("Running Llama generation...")
    print("=" * 80)

    llama_generations = []
    llama_total_time = 0

    pbar = tqdm(
        range(0, len(prompts), GEN_BATCH_SIZE), 
        desc="Generating [llama]",
        total=(len(prompts) + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE
    )

    for i in pbar:
        batch_prompts = prompts[i:i + GEN_BATCH_SIZE]

        inputs = tokenize_with_bos(
            llama_tokenizer,
            batch_prompts,
            DEVICE,
            padding=True,
            truncation=True,
            max_length=256
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
        llama_total_time += batch_time

        generated_ids = output_ids[:, prompt_len:]
        generations = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        llama_generations.extend(generations)

        # Update progress bar
        tokens_generated = len(llama_generations) * MAX_NEW_TOKENS
        pbar.set_postfix({
            "tok/s": f"{tokens_generated / llama_total_time:.1f}",
            "samples": len(llama_generations)
        })

    print(f"    Calculating PPL for {len(llama_generations)} generations...")
    llama_ppl_results = calculate_ppl_for_generations(
        eval_model, eval_tokenizer, prompts, llama_generations, EVAL_BATCH_SIZE, DEVICE
    )

    results["llama"] = {
        "config": {"method": "autoregressive", "temperature": TEMPERATURE},
        "generations": llama_generations,
        "time": llama_total_time,
        "tokens_per_second": (len(prompts) * MAX_NEW_TOKENS) / llama_total_time,
        **llama_ppl_results,
    }

    print(f"    ✓ Time: {llama_total_time:.2f}s | Tok/s: {results['llama']['tokens_per_second']:.1f} | Full PPL: {llama_ppl_results['full_ppl']:.2f} | Gen PPL: {llama_ppl_results['gen_ppl']:.2f}")

    if SAVE_CHECKPOINTS:
        save_checkpoint(results, prompts, CHECKPOINT_DIR, "llama")
        print(f"    ✓ Checkpoint saved")

    # ==================== Results Summary ====================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print(f"Total samples evaluated: {len(prompts)}")
    print("=" * 80)

    # Sort by full PPL
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]["full_ppl"])

    print(f"\n{'Method':<20} {'Full PPL':>10} {'Gen PPL':>10} {'Time (s)':>10} {'Tok/s':>10} {'Speedup':>10}")
    print("-" * 75)

    llama_time = results["llama"]["time"]

    for method in sorted_methods:
        r = results[method]
        speedup = llama_time / r["time"] if r["time"] > 0 else 0
        print(f"{method:<20} {r['full_ppl']:>10.2f} {r['gen_ppl']:>10.2f} {r['time']:>10.2f} {r['tokens_per_second']:>10.1f} {speedup:>10.2f}x")

    # ==================== Best Configurations ====================
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    # Best PPL (excluding llama for fair comparison)
    diffusion_methods = [m for m in sorted_methods if m != "llama"]
    best_ppl = min(diffusion_methods, key=lambda x: results[x]["full_ppl"])
    print(f"\n🎯 Best PPL (Diffusion): {best_ppl}")
    print(f"   Full PPL: {results[best_ppl]['full_ppl']:.2f}")
    print(f"   Config: {results[best_ppl]['config']}")

    # Fastest
    fastest = min(diffusion_methods, key=lambda x: results[x]["time"])
    print(f"\n⚡ Fastest: {fastest}")
    print(f"   Time: {results[fastest]['time']:.2f}s")
    print(f"   Full PPL: {results[fastest]['full_ppl']:.2f}")

    # Best PPL-Speed tradeoff
    def tradeoff_score(method):
        ppl = results[method]["full_ppl"]
        speedup = llama_time / results[method]["time"]
        return ppl / speedup

    best_tradeoff = min(diffusion_methods, key=tradeoff_score)
    print(f"\n⚖️  Best PPL-Speed Tradeoff: {best_tradeoff}")
    print(f"   Full PPL: {results[best_tradeoff]['full_ppl']:.2f}")
    print(f"   Speedup: {llama_time / results[best_tradeoff]['time']:.2f}x")

    # ==================== Example Generations ====================
    print("\n" + "=" * 80)
    print("EXAMPLE GENERATIONS (first 3 prompts)")
    print("=" * 80)

    for i in range(min(3, len(prompts))):
        print(f"\n{'─' * 70}")
        print(f"Prompt [{i+1}]: {prompts[i][:80]}...")
        print(f"{'─' * 70}")

        for method in ["original", best_ppl, fastest, "llama"]:
            if method in results:
                gen = results[method]["generations"][i]
                ppl = results[method]["gen_individual"][i]
                label = f"{method.upper()}"
                if method == best_ppl and method != "original":
                    label += " (Best PPL)"
                if method == fastest and method != "original":
                    label += " (Fastest)"
                print(f"{label:<25} [PPL:{ppl:>7.1f}] {gen[:100]}{'...' if len(gen) > 100 else ''}")

    # ==================== Save All Results ====================
    print("\n" + "=" * 80)
    print("Saving all results to folder...")
    print("=" * 80)

    exp_name = f"hellaswag_full_n{len(prompts)}_tok{MAX_NEW_TOKENS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results_to_folder(results, prompts, OUTPUT_DIR, exp_name)

    # Also save quick summary JSON
    summary_results = {}
    for method, r in results.items():
        summary_results[method] = {
            "config": r["config"],
            "time": r["time"],
            "tokens_per_second": r["tokens_per_second"],
            "full_ppl": r["full_ppl"],
            "gen_ppl": r["gen_ppl"],
            "num_samples": len(r["generations"]),
        }

    with open("ppl_evaluation_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print(f"   Total samples: {len(prompts)}")
    print(f"   Results saved to: {OUTPUT_DIR}/{exp_name}/")
    print("=" * 80)


if __name__ == "__main__":
    main()