# main.py
import argparse
import json
import logging
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    LlamaForCausalLM,
    TrainerCallback
)



from ..collators import (
    NTPCollator,
    MDLMCollator,
    CausalMLMCollator, 
    BD3LMCollator,
    PrefixLMCollator,
    ReasoningCARDCollator,
    ReasoningARCollator,
    ReasoningMDLMCollator,
)
from ..collators.reasoning import ReasoningMDMCollator


import math
from ..modeling.modeling_niu import ModernBertForDiffusionLM
from ..modeling.configuration_niu import NiuConfig
from ..llada.modeling_llada import LLaDAModelLM
from ..llada.configuration_llada import LLaDAConfig
from ..bd3lm.modeling_bd3lm import BD3LM
from ..bd3lm.configuration_bd3lm import BD3LMConfig


# --- 修改导入 ---
# 假设 EMATrainer 在 ..trainer 模块中
from ..trainer import EMATrainer # <--- 导入 EMATrainer
from ..utils.debug_func import analyze_weights,debug_data

# 在 main 函数的开始部分
import torch.multiprocessing as mp
from ..utils.load_dataset import get_dataset
from ..reasoning_tasks.char_tokenizer import ReasoningCharTokenizer
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
    except Exception as e:
        print(f"检测主进程代码失效{e}")
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



def align_config_and_tokenizer(config,tokenizer):
    config.mask_token_id = tokenizer.mask_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id 
    config.embedding_size = len(tokenizer)
    config.vocab_size = len(tokenizer)




def main():
    # --- 1. 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="使用可配置参数训练一个MLM模型")

    # 路径参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型或本地模型/分词器的路径。")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="训练数据集的名称（如：finefineweb）。")

    parser.add_argument("--validation_dataset_name", type=str, default=None,
                        help="验证数据集的名称（如：finefineweb_validation）。")
    parser.add_argument("--config_path", type=str, required=True,
                        help="模型配置文件的路径。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型 checkpoints 和输出的保存路径。")
    parser.add_argument("--mode",default="niu")
    
    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=1)
    parser.add_argument("--mlm_end_prob", type=float, default=0.0001)
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512, help="输入序列的最大长度。")
    parser.add_argument("--pad_to_max_length", action="store_true", 
                        help="如果设置，将所有样本填充到 max_length，而不是 Batch 内最大长度。")
    # loss参数
    parser.add_argument("--use_daum", 
                        type=ast.literal_eval, 
                        default=True, 
                        help="Enable DAUM loss. Pass True or False.")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps",type=int,default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_kwargs", 
        type=ast.literal_eval, 
        default='{}',
        help='LR scheduler keyword arguments as a Python dict string. Example: \'{"min_lr_rate": 0.01}\''
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="每个设备的评估批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
    # --- 修改 1: 添加 save_strategy 参数，确认 save_total_limit 存在 ---
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Checkpoints 保存策略 ('no', 'steps', 'epoch')。")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="最多保存多少个 Checkpoints。")
                        
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler_type",type=str,default="cosine_with_min_lr")
    parser.add_argument("--warmup_steps",type=int,default=0)
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="评估策略 ('no', 'steps', 'epoch')。")
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="每隔多少步进行一次评估。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action='store_true')
    
    # --- 新增 EMA 参数 ---
    parser.add_argument("--ema_decay", type=float, default=None, 
                        help="Exponential Moving Average decay rate. 如果为 None 则禁用 EMA。")

    args = parser.parse_args()
    if args.ema_decay is not None and args.ema_decay <= 0:
        args.ema_decay = None  # 允许用户传 0 来关闭
        
    # --- 2. 打印和保存参数配置 ---
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
        logging.info(f"加载训练数据集 '{args.dataset_name}'...")
    train_dataset = get_dataset(args.dataset_name)
    
    if is_main_process():
        print(train_dataset)
        print(train_dataset[0])

    eval_dataset = None
    if args.validation_dataset_name and args.validation_dataset_name != "":
        if is_main_process():
            logging.info(f"加载验证数据集 '{args.validation_dataset_name}'...")
        eval_dataset = get_dataset(args.validation_dataset_name)

        print(eval_dataset)
        eval_on_start = False
    else:
        args.evaluation_strategy = "no"
        eval_on_start = False

    model_path =  args.model_name_or_path
    if is_main_process():
        logging.info(f"从路径 '{model_path}' 加载分词器...")
    
    # Use custom character-level tokenizer for reasoning tasks
    if args.mode in ('reasoning_card', 'reasoning_ar', 'reasoning_mdlm', 'reasoning_mdm'):
        tokenizer = ReasoningCharTokenizer()
        if is_main_process():
            logging.info(f"Using ReasoningCharTokenizer (vocab_size={tokenizer.vocab_size})")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if 'modernbert' in model_path.lower():
            if tokenizer.eos_token is None:
                tokenizer.eos_token_id = 50279
            if tokenizer.bos_token is None:
                tokenizer.bos_token_id = 50285
        elif 'gpt' in model_path.lower():
            tokenizer.add_tokens(['<|begin_of_text|>','[MASK]'])
            tokenizer.bos_token = '<|begin_of_text|>'
            tokenizer.mask_token = '[MASK]'
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.eos_token is None:
                tokenizer.eos_token_id = 50279
            if tokenizer.bos_token is None:
                tokenizer.bos_token_id = 50285
        elif 'bert' in model_path.lower():
            tokenizer.add_tokens(['<|begin_of_text|>','<|end_of_text|>'])
            tokenizer.bos_token = '<|begin_of_text|>'
            tokenizer.eos_token = '<|end_of_text|>'
        else:
            assert False,f"unsupported tokenizer"

    

    if args.mode in ('llada', 'mdlm', 'prefixlm', 'reasoning_mdlm', 'reasoning_mdm'):
        config = LLaDAConfig.from_pretrained(args.config_path)
        align_config_and_tokenizer(config,tokenizer)

        model = LLaDAModelLM(config,init_params=True)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    elif args.mode in ('llama', 'reasoning_ar'):
        config = AutoConfig.from_pretrained(args.config_path)
        align_config_and_tokenizer(config,tokenizer)
        model = LlamaForCausalLM(config)
    elif args.mode == "bd3lm":
        config = BD3LMConfig.from_pretrained(args.config_path)
        align_config_and_tokenizer(config,tokenizer)
        model = BD3LM(config)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    elif args.mode == "reasoning_card":
        config = NiuConfig.from_pretrained(args.config_path)
        align_config_and_tokenizer(config,tokenizer)
        model = ModernBertForDiffusionLM(config)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    else:
        config = NiuConfig.from_pretrained(args.config_path)
        align_config_and_tokenizer(config,tokenizer)
        model = ModernBertForDiffusionLM(config)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")


    eval_collator = None
    if args.mode == 'llama' or args.mode == 'reasoning_ar':
        collator = NTPCollator(
            tokenizer, 
            max_length=args.max_length, 
            pad_to_max_length=args.pad_to_max_length
        ) if args.mode == 'llama' else ReasoningARCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == "mdlm":
        collator = MDLMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == "reasoning_mdlm":
        collator = ReasoningMDLMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == "reasoning_mdm":
        collator = ReasoningMDMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
            diffusion_steps=20,
            time_reweighting='linear',
            token_reweighting=True,
            focal_alpha=0.25,
            focal_gamma=2.0,
        )
    elif args.mode == "bd3lm":
        collator = BD3LMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == "prefixlm":
        collator = PrefixLMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == "reasoning_card":
        collator = ReasoningCARDCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=args.tail_bias_factor,
            use_daum=args.use_daum,
            pad_to_max_length=args.pad_to_max_length,
        )
    else:
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
        eval_collator = CausalMLMCollator(
            tokenizer, 
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=args.tail_bias_factor,
            use_daum=args.use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=True, 
        )

    # --- 修改 2: 在 TrainingArguments 中使用这些参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        
        # 这里的 save_strategy 以前是 "steps"，现在改为 args.save_strategy
        save_strategy=args.save_strategy,
        # save_steps 只在 save_strategy="steps" 时生效
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        data_seed=args.seed,
        seed=args.seed,
        bf16=True,
        # adam_beta2 = 0.95,
        # weight_decay = 0.1,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        eval_strategy=args.evaluation_strategy, 
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        eval_on_start = eval_on_start,
    )

    # --- 7. 初始化并开始训练 ---
    trainer = EMATrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=collator,
        ema_decay=args.ema_decay,
        eval_data_collator = eval_collator,
    )
    

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

if __name__ == "__main__":
    main()