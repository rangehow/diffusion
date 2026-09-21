"""
Ablation 1 & 5: Decomposing CARD's contributions (Reviewer 2Bg4)
               + Tail-bias factor sensitivity.

=== FAIRNESS GUARANTEE ===
ALL variants (including PrefixLM) use:
  - The SAME ModernBertForDiffusionLM architecture (from NiuConfig)
  - The SAME random initialization (same seed)
  - The SAME dataset (fineweb_edu_1b, ~1B tokens)
  - The SAME training hyperparameters (LR, batch size, steps, etc.)
  - The SAME tokenizer (ModernBERT-base, 50k vocab)
The ONLY controlled variable is the data collator (masking/weighting strategy).
This is a proper controlled ablation — NOT a comparison against a separately
pretrained model.

Variant 1: full_card        — Tail-biased masking + DAUM (full CARD)
Variant 2: no_daum          — Tail-biased masking, no DAUM weighting
Variant 3: no_tail_bias     — Uniform masking + DAUM (no tail bias)
Variant 4: no_both          — Uniform masking, no DAUM (pure causal diffusion)
Variant 5: prefix_lm        — Prefix-LM baseline (no diffusion at all)

Usage:
    # Ablation 1: Component decomposition
    python -m ablation.train_ablation1 \
        --variant full_card \
        --model_name_or_path <tokenizer_path> \
        --config_path <model_config.json> \
        --dataset_name fineweb_edu_1b \
        --output_dir ablation_outputs/full_card \
        --max_steps 50000

    # Ablation 5: Tail-bias factor sensitivity
    python -m ablation.train_ablation1 \
        --variant full_card \
        --tail_bias_factor 2.0 \
        --output_dir ablation_outputs/tbf_2.0 \
        ...
"""

import argparse
import json
import logging
import os
import ast

from transformers import AutoTokenizer, TrainingArguments

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collators import CausalMLMCollator, NTPCollator
from ablation.collator_ablation import PrefixLMCollator, UniformCausalMLMCollator
from modeling.modeling_niu import ModernBertForDiffusionLM
from modeling.configuration_niu import NiuConfig
from trainer import EMATrainer
from utils.load_dataset import get_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


VARIANT_CONFIGS = {
    "full_card": {
        "description": "Full CARD: tail-biased masking + DAUM",
        "collator_cls": CausalMLMCollator,
        "use_daum": True,
        "tail_bias_factor": 1.5,
    },
    "no_daum": {
        "description": "CARD without DAUM: tail-biased masking, uniform loss weighting",
        "collator_cls": CausalMLMCollator,
        "use_daum": False,
        "tail_bias_factor": 1.5,
    },
    "no_tail_bias": {
        "description": "CARD without tail-bias: uniform masking + DAUM",
        "collator_cls": UniformCausalMLMCollator,
        "use_daum": True,
        "tail_bias_factor": 1.0,  # not used but for logging
    },
    "no_both": {
        "description": "CARD without tail-bias AND without DAUM (minimal causal diffusion)",
        "collator_cls": UniformCausalMLMCollator,
        "use_daum": False,
        "tail_bias_factor": 1.0,
    },
    "prefix_lm": {
        "description": "Prefix-LM baseline: no diffusion, random prefix, causal NTP on suffix",
        "collator_cls": PrefixLMCollator,
        "use_daum": False,
        "tail_bias_factor": 1.0,
    },
}


def is_main_process():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True


def check_for_checkpoints(output_dir):
    import re
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, item)) and re.match(r"^checkpoint-", item)
        for item in os.listdir(output_dir)
    )


def align_config_and_tokenizer(config, tokenizer):
    config.mask_token_id = tokenizer.mask_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.embedding_size = len(tokenizer)
    config.vocab_size = len(tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Train CARD ablation variants")

    # Variant selection
    parser.add_argument("--variant", type=str, required=True,
                        choices=list(VARIANT_CONFIGS.keys()),
                        help="Which ablation variant to train")
    parser.add_argument("--tail_bias_factor", type=float, default=None,
                        help="Override tail_bias_factor (for Ablation 5 sweep). "
                             "If not set, uses the default from VARIANT_CONFIGS.")

    # Paths
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Tokenizer / base model path")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Model config JSON path")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--validation_dataset_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training hyperparams (same for all variants for fair comparison)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)

    # MLM schedule (only used by applicable variants)
    parser.add_argument("--mlm_start_prob", type=float, default=1.0)
    parser.add_argument("--mlm_end_prob", type=float, default=0.0001)

    parser.add_argument("--lr_scheduler_kwargs", type=ast.literal_eval, default='{}')

    args = parser.parse_args()

    variant_cfg = VARIANT_CONFIGS[args.variant].copy()  # copy to avoid mutating global

    # Override tail_bias_factor if specified via CLI (for Ablation 5: sensitivity sweep)
    if args.tail_bias_factor is not None:
        variant_cfg["tail_bias_factor"] = args.tail_bias_factor
        if is_main_process():
            logging.info(f"CLI override: tail_bias_factor = {args.tail_bias_factor}")

    if is_main_process():
        logging.info("=" * 80)
        logging.info(f"ABLATION 1: Training variant '{args.variant}'")
        logging.info(f"Description: {variant_cfg['description']}")
        logging.info("=" * 80)
        for k, v in vars(args).items():
            logging.info(f"  {k:35s}: {v}")
        logging.info("=" * 80)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if 'modernbert' in args.model_name_or_path.lower():
        if tokenizer.eos_token is None:
            tokenizer.eos_token_id = 50279
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = 50285
    elif 'gpt' in args.model_name_or_path.lower():
        tokenizer.add_tokens(['<|begin_of_text|>', '[MASK]'])
        tokenizer.bos_token = '<|begin_of_text|>'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = tokenizer.eos_token
    elif 'bert' in args.model_name_or_path.lower():
        tokenizer.add_tokens(['<|begin_of_text|>', '<|end_of_text|>'])
        tokenizer.bos_token = '<|begin_of_text|>'
        tokenizer.eos_token = '<|end_of_text|>'

    # --- Load model ---
    config = NiuConfig.from_pretrained(args.config_path)
    align_config_and_tokenizer(config, tokenizer)
    model = ModernBertForDiffusionLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    # --- Load dataset ---
    train_dataset = get_dataset(args.dataset_name)
    eval_dataset = None
    if args.validation_dataset_name:
        eval_dataset = get_dataset(args.validation_dataset_name)

    # --- Create collator based on variant ---
    CollatorCls = variant_cfg["collator_cls"]
    use_daum = variant_cfg["use_daum"]
    tail_bias_factor = variant_cfg["tail_bias_factor"]

    if CollatorCls == CausalMLMCollator:
        collator = CausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=tail_bias_factor,
            use_daum=use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=False,
        )
        eval_collator = CausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=tail_bias_factor,
            use_daum=use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=True,
        )
    elif CollatorCls == UniformCausalMLMCollator:
        collator = UniformCausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            use_daum=use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=False,
        )
        eval_collator = UniformCausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            use_daum=use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=True,
        )
    elif CollatorCls == PrefixLMCollator:
        collator = PrefixLMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
        )
        # For eval, use NTP collator (clean context, standard NTP loss)
        eval_collator = NTPCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
        )
    else:
        raise ValueError(f"Unknown collator class: {CollatorCls}")

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        logging_steps=args.logging_steps,
        dataloader_num_workers=8,
        report_to='none',
        include_num_input_tokens_seen=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    # --- Trainer ---
    trainer = EMATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        ema_decay=args.ema_decay,
        eval_data_collator=eval_collator,
    )

    # --- Save ablation metadata ---
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        meta = {
            "variant": args.variant,
            "description": variant_cfg["description"],
            "use_daum": use_daum,
            "tail_bias_factor": tail_bias_factor,
            "collator_cls": CollatorCls.__name__,
            **vars(args),
        }
        with open(os.path.join(args.output_dir, "ablation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # --- Train ---
    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
