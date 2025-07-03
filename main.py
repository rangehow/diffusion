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
    LlamaForCausalLM
)
from .collator import MLMCollator,NTPCollator
from .mlm_schedule import LazyScheduledMLMProbProvider,LazyMLMProbSchedulerCallback
from .modeling import ModernBertForDiffusionLM
from .llada.modeling_llada import LLaDAModelLM
from .trainer import MultipleLossTrainer
import accelerate
# 在 main 函数的开始部分
import torch.multiprocessing as mp
from .utils.load_dataset import get_dataset
# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def is_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True


def debug_data(trainer,tokenizer,collator):
    dataloader = trainer.get_train_dataloader()

    # 检查collator的结果
    logging.info("开始检查collator的结果...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:  # 只检查前1个batch
            break
            
        logging.info(f"\n=== Batch {batch_idx + 1} ===")
        logging.info(f"Batch大小: {batch['input_ids'].shape[0]}")
        logging.info(f"序列长度: {batch['input_ids'].shape[1]}")
        
        # 检查是否存在token_change_labels
        has_token_change_labels = 'token_change_labels' in batch
        
        # 检查前1个样本
        for sample_idx in range(min(1, batch['input_ids'].shape[0])):
            logging.info(f"\n--- 样本 {sample_idx + 1} ---")
            
            input_ids = batch['input_ids'][sample_idx]
            labels = batch['labels'][sample_idx]
            token_change_labels = batch['token_change_labels'][sample_idx] if has_token_change_labels else None
            attention_mask = batch['attention_mask'][sample_idx]
            
            # 转换为tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 找到有效长度（排除padding）
            valid_length = attention_mask.sum().item()
            total_length = len(input_ids)
            
            logging.info(f"总token数量: {total_length}")
            logging.info(f"非pad token数量: {valid_length}")
            logging.info(f"padding token数量: {total_length - valid_length}")
            
            current_mlm_prob = collator._get_mlm_probability()
            logging.info(f"当前MLM概率: {current_mlm_prob:.3f}")
            
            if has_token_change_labels:
                logging.info("\n详细Token信息:")
                logging.info("格式: [位置] Token名称 (ID) | 输入状态 | MLM标签 | 变化标签 | 注意力")
            else:
                logging.info("\n详细Token信息 (无变化标签):")
                logging.info("格式: [位置] Token名称 (ID) | 输入状态 | MLM标签 | 注意力")
            logging.info("-" * 100)
            
            # 统计计数器
            masked_count = 0
            random_count = 0
            keep_count = 0
            original_count = 0
            pad_count = 0
            
            # 显示所有token，包括pad
            for i in range(total_length):
                token_id = input_ids[i].item()
                token = tokens[i]
                label = labels[i].item()
                change_label = token_change_labels[i].item() if has_token_change_labels else None
                attention = attention_mask[i].item()
                
                # 判断是否为padding
                is_padding = (attention == 0)
                
                if is_padding:
                    # padding token
                    status = "PAD"
                    mlm_label_info = f"标签:{label}"
                    change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    pad_count += 1
                else:
               
                    if token == tokenizer.mask_token:
                        status = f"MASK"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        masked_count += 1
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    elif token == label:
                        status = f"KEEP"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        keep_count +=1 
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    elif label!=-100:
                        status = f"随机替换"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        random_count += 1
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    else:
                        status = "无需计算loss"
                        mlm_label_info = "标签:无(-100)"
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                        original_count += 1

                
                # 显示token信息
                attention_info = f"注意力:{attention}"
                if has_token_change_labels:
                    logging.info(f"[{i:2d}] {token:6}  | {status:10} | {mlm_label_info:20} | {change_info:6} | {attention_info}")
                else:
                    logging.info(f"[{i:2d}] {token:6}  | {status:10} | {mlm_label_info:20} | {attention_info}")
            
            # 详细统计信息
            logging.info(f"\n=== 统计信息 ===")
            logging.info(f"总token数量: {total_length}")
            logging.info(f"有效token数量: {valid_length}")
            logging.info(f"padding token数量: {pad_count}")
            logging.info(f"")
            logging.info(f"MLM处理统计:")
            logging.info(f"  - 原始未处理: {original_count}")
            logging.info(f"  - MASK替换: {masked_count}")
            if has_token_change_labels:
                logging.info(f"  - 随机替换: {random_count}")
                logging.info(f"  - 保持原样: {keep_count}")
            logging.info(f"  - 总MLM处理: {masked_count + random_count + keep_count}")
            logging.info(f"")
            logging.info(f"比例统计:")
            logging.info(f"  - 当前MLM概率设置: {current_mlm_prob:.3f}")
            if valid_length > 0:
                actual_mlm_ratio = (masked_count + random_count + keep_count) / valid_length
                mask_ratio = masked_count / valid_length
                
                logging.info(f"  - 实际MLM处理比例: {actual_mlm_ratio:.3f}")
                logging.info(f"  - MASK比例: {mask_ratio:.3f}")
                
                if has_token_change_labels:
                    random_ratio = random_count / valid_length
                    keep_ratio = keep_count / valid_length
                    logging.info(f"  - 随机替换比例: {random_ratio:.3f}")
                    logging.info(f"  - 保持原样比例: {keep_ratio:.3f}")
            
            # 显示标签分布
            label_distribution = {}
            change_label_distribution = {} if has_token_change_labels else None
            
            for i in range(total_length):
                label = labels[i].item()
                label_distribution[label] = label_distribution.get(label, 0) + 1
                
                if has_token_change_labels:
                    change_label = token_change_labels[i].item()
                    change_label_distribution[change_label] = change_label_distribution.get(change_label, 0) + 1
            
            logging.info(f"\n=== 标签分布 ===")
            logging.info(f"MLM标签分布: {label_distribution}")
            if has_token_change_labels:
                logging.info(f"变化标签分布: {change_label_distribution}")
            
            # 显示原始文本和处理后文本的对比
            try:
                # 重建原始文本
                original_ids = input_ids.clone()
                for i in range(total_length):
                    if labels[i].item() != -100:
                        original_ids[i] = labels[i]
                
                # 只显示有效部分的文本
                original_text = tokenizer.decode(original_ids[:valid_length], skip_special_tokens=False)
                current_text = tokenizer.decode(input_ids[:valid_length], skip_special_tokens=False)
                
                logging.info(f"\n=== 文本对比 ===")
                logging.info(f"原始文本: {original_text}")
                logging.info(f"处理后文本: {current_text}")
                
                # 显示差异
                original_tokens = tokenizer.convert_ids_to_tokens(original_ids[:valid_length])
                current_tokens = tokenizer.convert_ids_to_tokens(input_ids[:valid_length])
                
                differences = []
                for i, (orig, curr) in enumerate(zip(original_tokens, current_tokens)):
                    if orig != curr:
                        differences.append(f"位置{i}: {orig} → {curr}")
                
                if differences:
                    logging.info(f"Token差异: {'; '.join(differences)}")
                else:
                    logging.info("没有Token差异")
                    
            except Exception as e:
                logging.warning(f"无法重建文本对比: {e}")


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
    parser.add_argument("--use_all_tokens_for_loss", type=bool, default=False,
                    help="是否对所有token计算loss，而不仅仅是被mask的token")
    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=0.25,
                        help="Lazy MLM provider 的初始 masking 概率。")
    parser.add_argument("--mlm_end_prob", type=float, default=0.15,
                        help="Lazy MLM provider 的最终 masking 概率。")
    parser.add_argument("--random_probability", type=float)
    parser.add_argument("--mask_probability", type=float)
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine',
                        help="Lazy MLM provider 的概率调度类型 (e.g., 'cosine', 'linear')。")

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

    # --- 2. 保存参数配置 ---
    # 只在主进程保存参数配置
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        args_dict = vars(args)
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

    
    config = AutoConfig.from_pretrained(args.config_path)
    shared_step = mp.Value('i', 0)

    if args.mode == 'llada':
        # 随机加噪，吸收态, 模型结构
        args.mlm_schedule_type = 'random'
        args.mlm_start_prob = 1
        args.mlm_end_prob = 0
        args.mask_probability = 1
        args.random_probability = 0
        args.use_all_tokens_for_loss = False
        model = LLaDAModelLM(config)
        
    elif args.mode == 'llama':
        model = LlamaForCausalLM(config)
    else:
        model = ModernBertForDiffusionLM(config)


    if args.mode == 'llama':
        collator = NTPCollator(tokenizer, max_length=args.max_length)
    else:
        lazy_prob_provider = LazyScheduledMLMProbProvider(
            shared_step=shared_step,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            schedule_type=args.mlm_schedule_type,
            
        )
        lazy_prob_scheduler_callback = LazyMLMProbSchedulerCallback(lazy_prob_provider,shared_step=shared_step)
        collator = MLMCollator(
            tokenizer, 
            mlm_probability=lazy_prob_provider,
            max_length=args.max_length,
            mask_probability=args.mask_probability,
            random_probability=args.random_probability,
            mode=args.mode,
            use_all_tokens_for_loss=args.use_all_tokens_for_loss
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
        # dataloader_num_workers=args.dataloader_num_workers,
        dataloader_num_workers=0,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


    # --- 7. 初始化并开始训练 ---
    trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        # callbacks=[lazy_prob_scheduler_callback,additional_callbacks],
        callbacks=[lazy_prob_scheduler_callback] if args.mode != 'llama' else None,
        keys_you_want_to_log = ['mlm_loss','token_change_loss','current_mlm_prob']
    )


    if is_main_process() and args.mode!='llama':
        debug_data(trainer, tokenizer, collator)

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



    



if __name__ == "__main__":
    main()