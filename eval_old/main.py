# main.py

import argparse
import json
from typing import Dict, List, Any
from itertools import cycle, combinations
from pathlib import Path
from datetime import datetime
import math
import time

from datetime import datetime, timezone
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer
)
import numpy as np
import ray


from .tasks import get_task, TASK_REGISTRY
from .tasks.base import TaskConfig # 或者也在 __init__.py 中暴露 TaskConfig

class LogProbCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list, continuation_masks, attention_masks = [], [], []
        group_ids, is_correct_list, continuation_len_list, continuation_char_len_list = [], [], [], []
        max_length = 0

        for item in batch:
            full_sequence = item['input_ids'] + item['continuation_ids']
            input_ids_list.append(full_sequence)
            continuation_masks.append([0] * len(item['input_ids']) + [1] * len(item['continuation_ids']))
            attention_masks.append([1] * len(full_sequence))

            group_ids.append(item['group_id'])
            is_correct_list.append(item['is_correct'])
            continuation_len_list.append(item['continuation_len'])
            continuation_char_len_list.append(item.get('continuation_char_len', 1))

            max_length = max(max_length, len(full_sequence))

        padded_input_ids, padded_continuation_masks, padded_attention_masks = [], [], []
        for seq, cont_mask, att_mask in zip(input_ids_list, continuation_masks, attention_masks):
            pad_length = max_length - len(seq)
            padded_input_ids.append(seq + [self.pad_token_id] * pad_length)
            padded_continuation_masks.append(cont_mask + [0] * pad_length)
            padded_attention_masks.append(att_mask + [0] * pad_length)

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'continuation_mask': torch.tensor(padded_continuation_masks, dtype=torch.bool),
            'group_ids': group_ids,
            'is_correct': is_correct_list,
            'continuation_len': continuation_len_list,
            'continuation_char_len': continuation_char_len_list
        }


def compute_logprobs_causal(model: PreTrainedModel, batch: Dict[str, torch.Tensor], **kwargs) -> List[float]:
    input_ids, attention_mask, continuation_mask = batch['input_ids'].to(model.device), batch['attention_mask'].to(model.device), batch['continuation_mask'].to(model.device)
    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        masked_logprobs = token_logprobs * continuation_mask[:, 1:]
        total_logprobs = masked_logprobs.sum(dim=1)

    return total_logprobs.cpu().tolist()

def _compute_nll_deterministic(
    model: PreTrainedModel,
    seq: torch.Tensor,
    prompt_len: int,
    target_len: int,
    mask_token_id: int,
    batch_size: int
) -> float:
    """
    Deterministically computes the exact NLL for a sequence by enumerating all possible masks.
    This is feasible only when target_len is small.
    The NLL is calculated as the expectation over all mask sizes k (from 1 to T).
    NLL = (1/T) * sum_{k=1 to T} [ (T/k) * E_{mask of size k}[loss] ]
    """
    device = model.device
    total_loss_terms = []
    
    with torch.inference_mode():
        for k in range(1, target_len + 1):
            target_indices = range(target_len)
            mask_positions_in_target = list(combinations(target_indices, k))
            num_combinations = len(mask_positions_in_target)
            
            k_total_loss = 0.0
            
            for i in range(0, num_combinations, batch_size):
                batch_combinations = mask_positions_in_target[i:i+batch_size]
                actual_batch_size = len(batch_combinations)
                
                seq_batch = seq.unsqueeze(0).repeat(actual_batch_size, 1)
                mask = torch.zeros_like(seq_batch, dtype=torch.bool, device=device)
                absolute_indices = torch.tensor(batch_combinations, dtype=torch.long, device=device) + prompt_len
                rows = torch.arange(actual_batch_size, device=device).unsqueeze(1)
                mask[rows, absolute_indices] = True
                
                perturbed_seq = torch.where(mask, mask_token_id, seq_batch)
                logits = model(perturbed_seq).logits
                loss = F.cross_entropy(logits[mask], seq_batch[mask], reduction='sum')
                k_total_loss += loss.item()

            avg_loss_for_k = k_total_loss / num_combinations
            weighted_loss = avg_loss_for_k * (target_len / k)
            total_loss_terms.append(weighted_loss)

    final_nll = sum(total_loss_terms) / target_len
    return final_nll

# NEW: Function to compute Pseudo-Log-Likelihood in a batched way
def _compute_pll(
    model: PreTrainedModel,
    seq: torch.Tensor,
    prompt_len: int,
    target_len: int,
    mask_token_id: int,
) -> float:
    """
    Computes the Pseudo-Log-Likelihood for the target part of a sequence.
    This is done by masking one token at a time in the target sequence and summing
    the log-probabilities of the original token at that position.
    The process is batched for efficiency.
    """
    device = model.device
    
    # Create a batch where each row corresponds to masking a different token
    seq_batch = seq.unsqueeze(0).repeat(target_len, 1)
    
    # Create a mask that masks one token per row, diagonally
    mask = torch.zeros_like(seq_batch, dtype=torch.bool, device=device)
    target_indices = torch.arange(target_len, device=device)
    mask[target_indices, prompt_len + target_indices] = True
    
    # Apply the mask
    perturbed_batch = torch.where(mask, mask_token_id, seq_batch)
    
    with torch.inference_mode():
        # Get logits for the entire batch
        logits_batch = model(perturbed_batch).logits
        
        # Extract logits only at the masked positions
        logits_at_masked_pos = logits_batch[mask]
        
        # Get the original token IDs that were masked
        original_token_ids = seq_batch[mask]
        
        # Calculate log-softmax over the vocabulary dimension
        log_probs = F.log_softmax(logits_at_masked_pos, dim=-1)
        
        # Gather the log-probabilities of the correct tokens
        token_log_probs = torch.gather(log_probs, dim=-1, index=original_token_ids.unsqueeze(-1)).squeeze(-1)
        
        # The final PLL is the sum of these log-probabilities
        total_log_prob = token_log_probs.sum().item()
        
    return total_log_prob


# MODIFIED: The main diffusion logprob computer now supports different evaluation modes
def compute_logprobs_diffusion(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    mc_num: int,
    mc_batch_size: int,
    eval_mode: str = 'mc' # MODIFIED: Added eval_mode parameter
) -> List[float]:
    """
    Computes the log-likelihood for a discrete diffusion model.
    Supports two modes:
    - 'mc': Monte Carlo estimation (or deterministic if target is short).
    - 'pll': Pseudo-Log-Likelihood estimation.
    """
    input_ids = batch['input_ids']
    continuation_mask = batch['continuation_mask']

    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    device = model.device

    if mask_token_id is None:
        raise ValueError("Tokenizer for Discrete Diffusion Model must have a `mask_token_id`.")

    final_log_probs = []

    for i in range(input_ids.shape[0]):
        sample_ids = input_ids[i]
        sample_cont_mask = continuation_mask[i]

        seq_len = (sample_ids != pad_token_id).sum().item() if pad_token_id is not None else len(sample_ids)
        seq = sample_ids[:seq_len].to(device)
        cont_mask_unpadded = sample_cont_mask[:seq_len]

        prompt_len = (~cont_mask_unpadded).sum().item()
        target_len = cont_mask_unpadded.sum().item()

        if target_len == 0:
            final_log_probs.append(0.0)
            continue
        
        # MODIFIED: Switch between evaluation modes
        if eval_mode == 'pll':
            log_prob = _compute_pll(
                model=model,
                seq=seq,
                prompt_len=prompt_len,
                target_len=target_len,
                mask_token_id=mask_token_id
            )
            final_log_probs.append(log_prob)
        
        elif eval_mode == 'mc':
            if target_len > 0 and (2**target_len) <= mc_num:
                nll = _compute_nll_deterministic(
                    model=model,
                    seq=seq,
                    prompt_len=prompt_len,
                    target_len=target_len,
                    mask_token_id=mask_token_id,
                    batch_size=mc_batch_size
                )
                final_log_probs.append(-nll)
            
            else:
                loss_accumulator = []
                # Ensure mc_num is a multiple of mc_batch_size for simplicity, or handle remainder
                if mc_num < mc_batch_size:
                    mc_batch_size = mc_num
                mc_loops = mc_num // mc_batch_size

                with torch.inference_mode():
                    for _ in range(mc_loops):
                        seq_batch = seq.unsqueeze(0).repeat(mc_batch_size, 1)
                        k = torch.randint(1, target_len + 1, (mc_batch_size,), device=device)
                        
                        # Generate random masks of size k for each item in the batch
                        target_indices = torch.arange(target_len, device=device)
                        shuffled_indices = torch.stack([torch.randperm(target_len, device=device) for _ in range(mc_batch_size)])
                        is_target_mask = shuffled_indices < k.unsqueeze(1)
                        
                        prompt_mask = torch.zeros(mc_batch_size, prompt_len, dtype=torch.bool, device=device)
                        full_mask = torch.cat((prompt_mask, is_target_mask), dim=1)
                        
                        perturbed_seq = torch.where(full_mask, mask_token_id, seq_batch)
                        
                        # The probability of a specific mask of size k is (1/T) * (1/C(T,k))
                        # The probability of a mask being chosen is p(k) * p(mask|k) = (1/T) * (1/C(T,k))
                        # We are estimating E[loss / p(mask)]. The loss is sum over masked tokens.
                        # The MC estimator for the NLL is avg( loss_i / p(mask_i) ) over samples i.
                        # NLL = E_k E_mask [ L(mask) * T/k ]. Our p(k) is uniform 1/T, so we get E_mask[ L(mask) / (k/T) ].
                        p_mask_ratio = (k / target_len).unsqueeze(1).repeat(1, seq_len)
                        
                        logits = model(perturbed_seq).logits
                        # Important: use reduction='none' to get per-token loss
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), seq_batch.view(-1), reduction='none').view_as(seq_batch)
                        masked_loss = loss * full_mask
                        
                        # Sum loss for each sequence in the batch and divide by the sampling probability correction factor
                        loss_per_sequence = masked_loss.sum(dim=1)
                        k_float = k.float()
                        # Avoid division by zero, although k is from 1 to target_len
                        k_float[k_float == 0] = 1.0
                        
                        weighted_loss_per_sequence = loss_per_sequence * (target_len / k_float)
                        
                        avg_loss_for_iter = weighted_loss_per_sequence.mean()
                        loss_accumulator.append(avg_loss_for_iter.item())

                nll = sum(loss_accumulator) / len(loss_accumulator) if loss_accumulator else 0.0
                final_log_probs.append(-nll)
        else:
            raise ValueError(f"Unknown diffusion_eval_mode: {eval_mode}")

    return final_log_probs


def calculate_metrics(group_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    correct_raw, correct_norm_char, correct_norm_token = 0, 0, 0
    total_groups = len(group_results)

    if total_groups == 0:
        return { "acc": 0.0, "acc_norm_char": 0.0, "acc_norm_token": 0.0, "acc_correct": 0, "acc_norm_char_correct": 0, "acc_norm_token_correct": 0, "total": 0 }

    for group_id, group_data in group_results.items():
        if max(group_data, key=lambda x: x['logprob'])['is_correct'] == 1:
            correct_raw += 1
        
        # Character-level normalized accuracy
        # The key change is using max(x['..._len'], 1) to prevent division by zero.
        # This robustly handles cases where an option string is empty.
        if max(group_data, key=lambda x: x['logprob'] / max(x['continuation_char_len'], 1))['is_correct'] == 1:
            correct_norm_char += 1
        
        # Token-level normalized accuracy
        if max(group_data, key=lambda x: x['logprob'] / max(x['continuation_len'], 1))['is_correct'] == 1:
            correct_norm_token += 1

    return {
        "acc": correct_raw / total_groups,
        "acc_norm_char": correct_norm_char / total_groups,
        "acc_norm_token": correct_norm_token / total_groups,
        "acc_correct": correct_raw,
        "acc_norm_char_correct": correct_norm_char,
        "acc_norm_token_correct": correct_norm_token,
        "total": total_groups
    }

def calculate_mc2_prob_mass(group_results: Dict[str, List[Dict]]) -> tuple:
    all_pm_true_scores, total_groups = [], len(group_results)
    if total_groups == 0: return 0.0, 0, 0
    for group_id, group_data in group_results.items():
        if not group_data: continue
        logprobs = np.array([item['logprob'] for item in group_data])
        labels = np.array([item['is_correct'] for item in group_data])
        probs_norm = np.exp(logprobs) / np.sum(np.exp(logprobs))
        all_pm_true_scores.append(np.sum(probs_norm[labels == 1]))
    return np.mean(all_pm_true_scores) if all_pm_true_scores else 0.0, len(all_pm_true_scores), total_groups

@ray.remote(num_gpus=1)
class Evaluator:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        trust_remote_code: bool,
        mc_num: int,
        batch_size: int,
        diffusion_eval_mode: str # NEW
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        if self.tokenizer.pad_token_id is None and model_type == 'causal':
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_type == 'causal':
            # Try AutoModelForCausalLM first, fallback to AutoModel if fails
            try:
                model_class = AutoModelForCausalLM
                self.model = model_class.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                ).to('cuda').eval()
                print(f"[INFO] Successfully loaded causal model using AutoModelForCausalLM for {model_name}")
            except (ValueError, OSError, ImportError) as e:
                print(f"[WARNING] AutoModelForCausalLM failed for {model_name}: {e}")
                print(f"[INFO] Falling back to AutoModel...")
                model_class = AutoModel
                self.model = model_class.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                ).to('cuda').eval()
                print(f"[INFO] Successfully loaded model using AutoModel for {model_name}")
        elif model_type == 'discrete_diffusion':
            model_class = AutoModel
            self.model = model_class.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
            ).to('cuda').eval()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        

        if model_type == 'causal':
            self.logprob_computer_fn = compute_logprobs_causal
            self.computer_kwargs = {}
        elif model_type == 'discrete_diffusion':
            self.logprob_computer_fn = compute_logprobs_diffusion
            self.computer_kwargs = {
                'mc_num': mc_num,
                'mc_batch_size': batch_size,
                'eval_mode': diffusion_eval_mode 
            }

    def evaluate_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch_logprobs = self.logprob_computer_fn(
            model=self.model, batch=batch, tokenizer=self.tokenizer, **self.computer_kwargs
        )
        return [
            {'group_id': gid, 'logprob': lp, 'is_correct': ic, 'continuation_len': cl, 'continuation_char_len': ccl}
            for gid, lp, ic, cl, ccl in zip(
                batch['group_ids'], batch_logprobs, batch['is_correct'], batch['continuation_len'], batch['continuation_char_len']
            )
        ]

def evaluate_dataset_ray(actors: List[Evaluator], tokenizer: PreTrainedTokenizer, dataset, batch_size: int, num_workers: int) -> tuple:
    print(f"Starting Ray evaluation on {len(dataset)} samples with {len(actors)} GPUs...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=LogProbCollator(tokenizer), num_workers=num_workers)
    result_futures = [actor.evaluate_batch.remote(batch) for actor, batch in zip(cycle(actors), dataloader)]
    all_results = [item for future in tqdm(result_futures, desc="Collecting results") for item in ray.get(future)]
    group_results = {}
    for res in all_results:
        group_results.setdefault(res['group_id'], []).append(res)
    return all_results, group_results
def get_model_short_name(model_path: str) -> str:
    path = Path(model_path)
    # Handle cases where path might end with a slash
    if path.name == '':
        path = path.parent
        
    # If it is a checkpoint folder, we combine the parent name (experiment name)
    # with the checkpoint name to create a unique identifier.
    # e.g., "my_run/checkpoint-100" -> "my_run_checkpoint-100"
    if path.name.startswith('checkpoint-'):
        return f"{path.parent.name}_{path.name}"
    
    # Otherwise just return the leaf name
    return path.name

def prepare_output_env(output_dir: str, model_path: str) -> Path:
    """
    创建模型输出目录。
    逻辑更新：如果模型路径包含 'checkpoint-', 则创建两级目录：'父目录名/checkpoint名'。
    这样可以将同一个实验的不同 checkpoint 收纳在一个文件夹下。
    """
    path = Path(model_path)
    # 处理路径末尾可能带斜杠的情况
    if path.name == '':
        path = path.parent
        
    # 判断是否为 checkpoint 文件夹
    if path.name.startswith('checkpoint-'):
        experiment_name = path.parent.name
        checkpoint_name = path.name
        # 新结构: output_dir / experiment_name / checkpoint_name
        model_dir = Path(output_dir) / experiment_name / checkpoint_name
    else:
        # 普通模型: output_dir / model_name
        model_dir = Path(output_dir) / path.name

    (model_dir / "runs").mkdir(parents=True, exist_ok=True)
    return model_dir



def save_evaluation_results(model_dir: Path, result_data: Dict):
    """
    以原子方式保存单次评估的完整结果，并更新全局摘要。
    现在会将不同任务的运行结果保存在各自的子目录中。
    """
    metadata = result_data['metadata']
    task_name = metadata['task_name']
    num_fewshot = metadata['run_args']['num_fewshot']
    model_type = metadata['model_type']
    
    # NEW: 为当前任务的运行结果创建一个专用的子目录
    task_runs_dir = model_dir / "runs" / task_name
    task_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建唯一的文件名和摘要键
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    
    # 用于 summary.json 的键，需要包含任务名以保证唯一性
    summary_specifiers = [task_name, f"{num_fewshot}shot", model_type]
    if model_type == 'discrete_diffusion':
        summary_specifiers.append(metadata['run_args']['diffusion_eval_mode'])
    summary_key = "_".join(summary_specifiers)

    # 用于 JSON 文件名。因为文件已在任务子目录中，文件名可以省略任务名，使其更简洁。
    file_specifiers = [f"{num_fewshot}shot", model_type]
    if model_type == 'discrete_diffusion':
        file_specifiers.append(metadata['run_args']['diffusion_eval_mode'])
    run_name_for_file = "_".join(file_specifiers)
    detailed_filename = f"{timestamp}_{run_name_for_file}.json"

    # 2. 保存详细的、自包含的运行结果
    # MODIFIED: 将详细结果保存到任务特定的子目录中
    detailed_filepath = task_runs_dir / detailed_filename
    try:
        with open(detailed_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed results for '{task_name}' saved to: {detailed_filepath}")
    except Exception as e:
        print(f"✗ Error saving detailed results to {detailed_filepath}: {e}")
        return

    # 3. 线程安全地更新 summary.json
    summary_filepath = model_dir / "summary.json"
    summary_update = {
        "timestamp_utc": metadata['timestamp_utc'],
        "metrics": {k: v for k, v in result_data['metrics'].items() if 'correct' not in k and k != 'total_questions'},
        "total_questions": result_data['metrics'].get('total', result_data['metrics'].get('total_questions', 0)),
        # `relative_to` 会自动处理新的子目录结构，生成正确的相对路径，如 "runs/csqa/..."
        "detailed_results_file": str(detailed_filepath.relative_to(model_dir))
    }

    try:
        # 简单的文件锁实现，适用于多进程但不适用于多节点
        lock_path = summary_filepath.with_suffix('.lock')
        for _ in range(10): # Try 10 times
            if not os.path.exists(lock_path):
                # 使用 os.open 创建原子性文件
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    break
                except FileExistsError:
                    time.sleep(0.1)
                    continue
            time.sleep(0.1)
        else:
            raise TimeoutError("Could not acquire lock for summary.json")

        summary_data = {}
        if summary_filepath.exists():
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
        
        # MODIFIED: 使用包含任务名的 summary_key 来更新摘要
        summary_data[summary_key] = summary_update
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Summary updated for '{summary_key}' in: {summary_filepath}")

    except Exception as e:
        print(f"✗ Error updating summary file {summary_filepath}: {e}")
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

def print_results_summary(result_data: Dict):
    """在控制台清晰地打印评估结果摘要。"""
    metadata = result_data['metadata']
    metrics = result_data['metrics']
    
    print(f"\n=== {metadata['task_name'].upper()} Evaluation Results ===")
    print(f"Model: {metadata['model_short_name']}")
    
    if metadata['task_name'] == 'truthfulqa_mc2':
        score = metrics.get('mc2_prob_mass_score', 0.0)
        total = metrics.get('total_questions', 0)
        scored = metrics.get('scored_count', 0)
        print(f"Metric: Normalized Probability Mass (MC2)")
        print(f"Score: {score:.4f} (Evaluated on {scored}/{total} questions)")
    else:
        total = metrics.get('total', 0)
        for metric_name, score in metrics.items():
            if metric_name.startswith('acc') and 'correct' not in metric_name:
                correct_count = metrics.get(f"{metric_name}_correct", 0)
                print(f"Metric: {metric_name}: {score:.4f} ({correct_count}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on log-probability tasks.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=['causal', 'discrete_diffusion'])
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="random", choices=['first_n', 'random', 'balanced'])
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mc_num", type=int, default=128, help="Total number of Monte Carlo iterations for diffusion NLL. Also acts as a threshold for switching to deterministic computation.")
    # NEW: Command line argument to choose the evaluation mode
    parser.add_argument("--diffusion_eval_mode", type=str, default="pll", choices=['mc', 'pll'], help="Evaluation mode for discrete diffusion models: 'mc' for Monte Carlo or 'pll' for Pseudo-Log-Likelihood.")
    
    args = parser.parse_args()
    print("Arguments:", args)

    # 使用新的准备函数
    model_dir = prepare_output_env(args.output_dir, args.model_name_or_path)
    model_short_name = get_model_short_name(args.model_name_or_path)
    print(f"Results will be saved to: {model_dir}")

    ray.init()
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: raise ValueError("No GPUs found.")
    print(f"Found {num_gpus} GPUs. Creating one actor per GPU.")

    actors = [
        Evaluator.remote(
            model_name=args.model_name_or_path,
            model_type=args.model_type,
            trust_remote_code=args.trust_remote_code,
            mc_num=args.mc_num,
            batch_size=args.batch_size,
            diffusion_eval_mode=args.diffusion_eval_mode # NEW
        ) for _ in range(num_gpus)
    ]

    tokenizer_for_collator = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer_for_collator.pad_token_id is None and args.model_type == 'causal':
        tokenizer_for_collator.pad_token_id = tokenizer_for_collator.eos_token_id

    all_task_results = []
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    for task_name in tasks:
        print(f"\n----- Starting evaluation for task: {task_name} ({args.num_fewshot}-shot) -----")
        if args.model_type == 'discrete_diffusion':
             print(f"----- Diffusion Eval Mode: {args.diffusion_eval_mode.upper()} -----")


        config = TaskConfig(
            tokenizer=tokenizer_for_collator, num_fewshot=args.num_fewshot,
            sampler_name=args.sampler, sampler_seed=args.sampler_seed,
            model_type=args.model_type  # <<< MODIFIED: 将 model_type 传递给 TaskConfig
        )


        task_instance = get_task(task_name, config)
        dataset = task_instance.process()
        

        if args.limit > 0:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

        # 在评估开始前打印一个数据样本，并附上 token ID
        if dataset:
            try:
                print("\n--- 准备阶段：打印数据样本 ---")
                print(f"任务: {task_name}")

                example_group_id = dataset[0]['group_id']
                
                prompt_info = None
                choices_info = []
                for item in dataset:
                    if item['group_id'] == example_group_id:
                        if prompt_info is None:
                            prompt_info = {
                                'text': tokenizer_for_collator.decode(
                                    item['input_ids'], skip_special_tokens=True
                                ),
                                'ids': item['input_ids']
                            }
                        
                        choices_info.append({
                            'text': tokenizer_for_collator.decode(
                                item['continuation_ids'], skip_special_tokens=True
                            ).strip(),
                            'ids': item['continuation_ids']
                        })
                
                if prompt_info and choices_info:
                    print(f"提示(Prompt):\n{prompt_info['text']}")
                    print(f"Token IDs (Prompt): {prompt_info['ids']}")
                    
                    print("\n选项(Choices):")
                    for i, choice in enumerate(choices_info):
                        print(f"  - 选项 {i+1}: '{choice['text']}'")
                        print(f"    Token IDs (Choice): {choice['ids']}")
                else:
                    print("未能从数据集中提取完整的样本。")
                
                print("------------------------\n")
            except Exception as e:
                print(f"\n[Warning] 无法打印数据样本: {e}")

        results, group_results = evaluate_dataset_ray(actors, tokenizer_for_collator, dataset, args.batch_size, args.num_workers)

        # NEW: Added debugging block to print average option lengths.
        all_option_lengths_token = []
        all_option_lengths_char = []
        if group_results:
            # Iterate through all options for all questions to collect lengths
            for group_data in group_results.values():
                for item in group_data:
                    all_option_lengths_token.append(item['continuation_len'])
                    all_option_lengths_char.append(item['continuation_char_len'])

        if all_option_lengths_token:
            avg_len_token = sum(all_option_lengths_token) / len(all_option_lengths_token)
            avg_len_char = sum(all_option_lengths_char) / len(all_option_lengths_char)
            min_len_token = min(all_option_lengths_token)
            max_len_token = max(all_option_lengths_token)
            print(f"\n[DEBUG] Task '{task_name}' Option Length Statistics:")
            print(f"  - Average (tokens): {avg_len_token:.2f}  (Min: {min_len_token}, Max: {max_len_token})")
            print(f"  - Average (chars):  {avg_len_char:.2f}")
        else:
            print("\n[DEBUG] No results to calculate option length statistics.")
        # END NEW BLOCK

        # 在评估后打印第一个评估样本的详细信息
        if group_results:
            try:
                example_group_id = next(iter(group_results))
                example_group_results = group_results[example_group_id]
                
                original_prompt_text = ""
                choices_with_results = []
                
                result_lookup = {
                    (res['continuation_len'], res['continuation_char_len']): res 
                    for res in example_group_results
                }
                
                for item in dataset:
                    if item['group_id'] == example_group_id:
                        if not original_prompt_text:
                            original_prompt_text = tokenizer_for_collator.decode(
                                item['input_ids'], skip_special_tokens=True
                            )
                        
                        key = (item['continuation_len'], item.get('continuation_char_len', 1))
                        if key in result_lookup:
                            res = result_lookup[key]
                            choices_with_results.append({
                                'text': tokenizer_for_collator.decode(item['continuation_ids'], skip_special_tokens=True).strip(),
                                'logprob': res['logprob'],
                                'is_correct': res['is_correct']
                            })
                            del result_lookup[key]

                if original_prompt_text and choices_with_results:
                    print("\n--- 评估样本示例 ---")
                    print(f"任务: {task_name}, 问题ID: {example_group_id}")
                    print(f"提示(Prompt):\n{original_prompt_text}")
                    print("\n选项(Choices):")

                    best_choice = max(choices_with_results, key=lambda x: x['logprob'])

                    for choice in sorted(choices_with_results, key=lambda x: x['logprob'], reverse=True):
                        correct_marker = " (正确答案)" if choice['is_correct'] else ""
                        prediction_marker = " <-- 模型预测" if choice['text'] == best_choice['text'] else ""
                        print(f"  - 选项: '{choice['text']}'")
                        print(f"    对数概率(LogProb): {choice['logprob']:.4f}{correct_marker}{prediction_marker}")

                    print(f"\n模型预测: '{best_choice['text']}'")
                    print(f"预测结果: {'正确' if best_choice['is_correct'] else '错误'}")
                    print("------------------------\n")

            except Exception as e:
                print(f"\n[Warning] 无法打印评估样本: {e}")

        print(f"\n=== {task_name.upper()} Evaluation Results ===")
        print(f"Model: {model_short_name}")

        if task_name == 'truthfulqa_mc2':
            score, scored_count, total = calculate_mc2_prob_mass(group_results)
            all_metrics = {
                'mc2_prob_mass_score': score, 
                'scored_count': scored_count, 
                'total_questions': total
            }
        else:
            all_metrics = calculate_metrics(group_results)

        # 2. 将所有信息打包到一个字典中
        final_result_package = {
            "metadata": {
                "model_name_or_path": args.model_name_or_path,
                "model_short_name": model_short_name,
                "model_type": args.model_type,
                "task_name": task_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_args": vars(args)
            },
            "metrics": all_metrics,
            "detailed_results": results
        }

        # 3. 打印和保存
        print_results_summary(final_result_package)
        save_evaluation_results(model_dir, final_result_package)

    ray.shutdown()

if __name__ == "__main__":
    main()