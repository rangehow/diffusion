import argparse
import json
import logging
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig
)
from .collator import NTPCollator,LLaDACollator,CausalLMCollator

from ..modeling.modeling_niu import ModernBertForDiffusionLM
from ..modeling.configuration_niu import NiuConfig
from ..llada.modeling_llada import LLaDAModelLM
from ..llada.configuration_llada import LLaDAConfig
from ..trainer import MultipleLossTrainer
from ..utils.debug_func import analyze_weights,debug_data

# 在 main 函数的开始部分
import torch.multiprocessing as mp
from ..utils.load_dataset import get_dataset
import datasets
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
                        help="训练数据集的名称（如：finefineweb）。")



    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型 checkpoints 和输出的保存路径。")
    parser.add_argument("--mode",default="niu")
    
    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=0.25)
    parser.add_argument("--mlm_end_prob", type=float, default=0.15)
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine')
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512, help="输入序列的最大长度。")

    # loss参数
    parser.add_argument("--use_daum", 
                        type=ast.literal_eval, 
                        default=False, 
                        help="Enable DAUM loss. Pass True or False.")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action='store_true')
    
    args = parser.parse_args()

    



    # --- 4. 加载数据集和分词器 ---
    if is_main_process():
        logging.info(f"加载训练数据集 '{args.dataset_name}'...")

    dataset_name_list = args.dataset_name.split(',')
    dataset_list=[]
    for dataset_name in dataset_name_list:
        dataset_list.append(get_dataset(dataset_name))

    train_dataset = datasets.concatenate_datasets(dataset_list).shuffle()


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

    model_path =  args.model_name_or_path
    if is_main_process():
        logging.info(f"从路径 '{model_path}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    
    tokenizer.eos_token_id = 50279
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 50285
    

    if args.mode == 'llama':
        model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_path,trust_remote_code=True)
    

    



    if args.mode == 'llama':
        collator = NTPCollator(tokenizer, max_length=args.max_length)
    elif args.mode == 'llada':
        collator = LLaDACollator(tokenizer,max_length=args.max_length)
    else:
        # lazy_prob_provider = LazyScheduledMLMProbProvider(
        #     shared_step=shared_step,
        #     start_prob=args.mlm_start_prob,
        #     end_prob=args.mlm_end_prob,
        #     schedule_type=args.mlm_schedule_type,
        # )
        # lazy_prob_scheduler_callback = LazyMLMProbSchedulerCallback(lazy_prob_provider,shared_step=shared_step)
        collator = CausalLMCollator(
            tokenizer, 
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor = args.tail_bias_factor,
            use_daum = args.use_daum
        )

    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        data_seed=args.seed,
        seed=args.seed,
        bf16=True,
        adam_beta2 = 0.95,
        weight_decay = 0.1,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # eval_on_start = True,
    )


    # --- 7. 初始化并开始训练 ---
    trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # <-- 变量名更新
        data_collator=collator,
        # callbacks=None if args.mode == 'llama' or args.mode == 'llada' else [lazy_prob_scheduler_callback],
        keys_you_want_to_log = ['unweighted_total_loss','lm_loss','current_mlm_prob','masked_lm_loss','non_masked_lm_loss',]
    )


    # if is_main_process() and args.mode!='llama':
    #     debug_data(trainer, tokenizer, collator)

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



if __name__ == "__main__":
    main()