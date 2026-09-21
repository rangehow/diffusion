import argparse
import datasets
import json
import logging
import os
import re
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    LlamaForCausalLM,
    ModernBertForMaskedLM,
    AutoModel,
    AutoModelForCausalLM
)
from .collator import MLMCollator,NTPCollator,LLaDACollator,CausalLMCollator
from .mlm_schedule import LazyScheduledMLMProbProvider,LazyMLMProbSchedulerCallback
from .modeling_niu import ModernBertForDiffusionLM
from .configuration_niu import NiuConfig
from .llada.modeling_llada import LLaDAModelLM
from .llada.configuration_llada import LLaDAConfig
from .trainer import MultipleLossTrainer
from .utils.debug_func import analyze_weights,debug_data
import accelerate
# 在 main 函数的开始部分
import torch.multiprocessing as mp
from .utils.load_dataset import get_dataset
# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
import ast

def is_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True



def check_for_checkpoints(output_dir):
    """
    检查指定的输出目录下是否存在类似 checkpoint- 的文件夹（更简练的版本）。
    """
    import re
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, item)) and re.match(r"^checkpoint-", item)
        for item in os.listdir(output_dir)
    )



def main():
    # --- 1. 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="使用可配置参数训练一个MLM模型")

    # 路径参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型或本地模型/分词器的路径。")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="数据集的名称（如：ultra_fineweb, wikipedia, common_crawl）。")
    parser.add_argument("--config_path", type=str, required=True,
                        help="模型配置文件的路径。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型 checkpoints 和输出的保存路径。")

    parser.add_argument("--mode",default="niu")
    # MLM Schedule 参数，消融实验
    parser.add_argument("--mlm_start_prob", type=float, default=0.25,
                        help="Lazy MLM provider 的初始 masking 概率。")
    parser.add_argument("--mlm_end_prob", type=float, default=0.15,
                        help="Lazy MLM provider 的最终 masking 概率。")
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine',
                        help="Lazy MLM provider 的概率调度类型 (e.g., 'cosine', 'linear')。")
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)
    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512,
                        help="输入序列的最大长度。")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="训练的总轮数。")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数。")
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help="学习率预热的比例。")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                        help="数据加载器使用的工作进程数。")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="最多保存的 checkpoint 数量。")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="每隔多少步记录一次日志。")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每隔多少步保存一次 checkpoint。")
    parser.add_argument("--seed", type=int, default=42,
                        help="用于复现的随机种子。")
    parser.add_argument("--bf16", action='store_true',
                        help="如果设置，则使用 bfloat16 混合精度训练。")

    args = parser.parse_args()

    # --- 2. 打印和保存参数配置 ---
    # 只在主进程打印和保存参数配置
    if is_main_process():
        logging.info("=" * 80)
        logging.info("训练参数配置:")
        logging.info("=" * 80)
        args_dict = vars(args)
        for key, value in args_dict.items():
            logging.info(f"{key:30}: {value}")
        logging.info("=" * 80)
        
        os.makedirs(args.output_dir, exist_ok=True)
        args_json_path = os.path.join(args.output_dir, "training_args.json")
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"所有参数已保存至: {args_json_path}")



    # --- 4. 加载数据集和分词器 ---
    if is_main_process():
        logging.info(f"加载数据集 '{args.dataset_name}'...")
    dataset = get_dataset(args.dataset_name)

    model_path =  args.model_name_or_path
    if is_main_process():
        logging.info(f"从路径 '{model_path}' 加载分词器和模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # 确保 eos_token 设置正确
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 50279

    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 50285



    shared_step = mp.Value('i', 0)

    if args.mode == 'llada':
        config = LLaDAConfig.from_pretrained(args.config_path)
        model = LLaDAModelLM(config,init_params=True)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    elif args.mode == 'llama':
        config = AutoConfig.from_pretrained(args.config_path)
        model = LlamaForCausalLM(config)
    else:
        config = NiuConfig.from_pretrained(args.config_path)
        model = ModernBertForDiffusionLM(config)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
        # model.register_for_auto_class("AutoModelForCausalLM")
    # analyze_weights(model)

 

    if args.mode == 'llama':
        collator = NTPCollator(tokenizer, max_length=args.max_length)
    
    elif args.mode == 'llada':
        collator = LLaDACollator(tokenizer,max_length=args.max_length)
    else:
        lazy_prob_provider = LazyScheduledMLMProbProvider(
            shared_step=shared_step,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            schedule_type=args.mlm_schedule_type,
            
        )
        
        lazy_prob_scheduler_callback = LazyMLMProbSchedulerCallback(lazy_prob_provider,shared_step=shared_step)
        collator = CausalLMCollator(
            tokenizer, 
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor = args.tail_bias_factor
        )



    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps", # 明确保存策略
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        data_seed=args.seed,
        seed=args.seed,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,


    )


    # --- 7. 初始化并开始训练 ---
    trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=None if args.mode == 'llama' or args.mode == 'llada' else [lazy_prob_scheduler_callback],
        keys_you_want_to_log = ['lm_loss','current_mlm_prob','corrector_loss','masked_lm_loss','non_masked_lm_loss']
    )


    # if is_main_process() and args.mode!='llama':
    #     debug_data(trainer, tokenizer, collator)

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



    



if __name__ == "__main__":
    main()