"""
Qwen3-8B Continued Pretraining — 5 experimental groups:
  1. ARM:           Pure next-token prediction (baseline)
  2. CARD-Naive:    Scattered mask + DAUM weighting (V1)
  3. CARD-V3:       Topological reorder + logical position_ids
  4. Adapter-Only:  Suffix mask + noise-gated adapter (adapter params only)
  5. Adapter-Full:  Suffix mask + noise-gated adapter (all params)
"""
import argparse
import json
import logging
import os
import ast
import math

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from ..collators import NTPCollator, CausalMLMCollator
from ..collators.v3 import CausalMLMCollatorV3
from ..collators.v5_suffix import SuffixMaskCollator
from ..modeling.noise_gated_adapter import insert_adapters
from ..trainer import EMATrainer
from ..utils.load_dataset import get_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_main_process():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except Exception:
        return True


def check_for_checkpoints(output_dir):
    import re
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, d)) and re.match(r"^checkpoint-", d)
        for d in os.listdir(output_dir)
    )


def setup_tokenizer(model_path):
    """Load Qwen3 tokenizer and add mask token if missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen3 has no BOS token — add one along with MASK
    special_tokens_to_add = {}
    if tokenizer.bos_token is None:
        special_tokens_to_add["bos_token"] = "<|begin_of_text|>"
    if tokenizer.mask_token is None:
        special_tokens_to_add["mask_token"] = "<|mask|>"
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|endoftext|>"

    if special_tokens_to_add:
        num_added = tokenizer.add_special_tokens(special_tokens_to_add)
        if is_main_process():
            logging.info(f"Added {num_added} special tokens: {special_tokens_to_add}")

    return tokenizer


def build_model_and_collator(args, tokenizer):
    """Build model + collator for each experimental mode."""
    mode = args.mode
    model_path = args.model_name_or_path

    # ──────── Load base Qwen3 model ────────
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Resize embeddings if we added special tokens
    model.resize_token_embeddings(len(tokenizer))

    # ──────── Choose collator and adapt model ────────
    eval_collator = NTPCollator(
        tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    if mode == "arm":
        # Group 1: Pure ARM
        collator = NTPCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
        )
        ddp_find_unused = False

    elif mode == "card_naive":
        # Group 2: CARD V1 (scattered mask + DAUM)
        collator = CausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=args.tail_bias_factor,
            use_daum=args.use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=False,
        )
        ddp_find_unused = False

    elif mode == "card_v3":
        # Group 3: Topological reorder + logical position_ids
        collator = CausalMLMCollatorV3(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=args.tail_bias_factor,
            use_daum=args.use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=False,
        )
        ddp_find_unused = False

    elif mode in ("adapter_only", "adapter_full"):
        # Groups 4 & 5: Suffix mask + noise-gated adapter
        collator = SuffixMaskCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
            alpha=args.schedule_alpha,
        )
        # Insert zero-initialized adapters
        model = insert_adapters(
            model,
            rank=args.adapter_rank,
            verbose=is_main_process(),
        )

        if mode == "adapter_only":
            # Freeze backbone, train only adapter
            for name, param in model.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False
            if is_main_process():
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                logging.info(
                    f"[adapter_only] Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
                    f"({trainable/total*100:.2f}%)"
                )
        else:
            # adapter_full: train everything
            if is_main_process():
                adapter_params = sum(
                    p.numel() for n, p in model.named_parameters() if "adapter" in n
                )
                total = sum(p.numel() for p in model.parameters())
                logging.info(
                    f"[adapter_full] Adapter: {adapter_params/1e6:.1f}M / Total: {total/1e6:.0f}M"
                )

        ddp_find_unused = True  # adapter_only has many frozen params
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model, collator, eval_collator, ddp_find_unused


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B Continued Pretraining (5 experiments)")

    # Paths
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--validation_dataset_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    # Experiment mode
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["arm", "card_naive", "card_v3", "adapter_only", "adapter_full"],
    )

    # Masking params (for card_naive, card_v3)
    parser.add_argument("--mlm_start_prob", type=float, default=1.0)
    parser.add_argument("--mlm_end_prob", type=float, default=0.0001)
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)
    parser.add_argument("--use_daum", type=ast.literal_eval, default=True)

    # Adapter params (for adapter_only, adapter_full)
    parser.add_argument("--adapter_rank", type=int, default=64)
    parser.add_argument("--schedule_alpha", type=float, default=1.0)

    # Data
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--pad_to_max_length", action="store_true")

    # Training
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--lr_scheduler_kwargs", type=ast.literal_eval, default="{}")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    # ──── Log config ────
    if is_main_process():
        logging.info("=" * 80)
        logging.info(f"Qwen3-8B CPT — mode={args.mode}")
        logging.info("=" * 80)
        for k, v in vars(args).items():
            logging.info(f"  {k:35s}: {v}")
        logging.info("=" * 80)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ──── Data ────
    train_dataset = get_dataset(args.dataset_name)
    eval_dataset = None
    if args.validation_dataset_name:
        eval_dataset = get_dataset(args.validation_dataset_name)

    # ──── Tokenizer ────
    tokenizer = setup_tokenizer(args.model_name_or_path)

    # ──── Model + Collator ────
    model, collator, eval_collator, ddp_find_unused = build_model_and_collator(args, tokenizer)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ──── Training args ────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        include_num_input_tokens_seen=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=ddp_find_unused,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ──── Trainer ────
    trainer = EMATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        eval_data_collator=eval_collator,
        ema_decay=args.ema_decay,
    )

    # ──── Train ────
    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ──── Save final ────
    if is_main_process():
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        logging.info("Training complete.")


if __name__ == "__main__":
    main()
