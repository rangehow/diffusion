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
    TrainingArguments,
    Trainer,
)
# 假设这些是你自定义的模块
from collator import MLMCollator
from mlm_schedule import LazyScheduledMLMProbProvider
from callbacks import LazyMLMProbSchedulerCallback
from modeling import ModernBertForDiffusionLM

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)



def debug_data(trainer):
    dataloader = trainer.get_train_dataloader()

    # 检查collator的结果
    logging.info("开始检查collator的结果...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:  # 只检查前1个batch
            break
            
        logging.info(f"\n=== Batch {batch_idx + 1} ===")
        logging.info(f"Batch大小: {batch['input_ids'].shape[0]}")
        logging.info(f"序列长度: {batch['input_ids'].shape[1]}")
        
        # 检查前1个样本
        for sample_idx in range(min(1, batch['input_ids'].shape[0])):
            logging.info(f"\n--- 样本 {sample_idx + 1} ---")
            
            input_ids = batch['input_ids'][sample_idx]
            labels = batch['labels'][sample_idx]
            token_change_labels = batch['token_change_labels'][sample_idx]
            attention_mask = batch['attention_mask'][sample_idx]
            
            # 转换为tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 找到有效长度（排除padding）
            valid_length = attention_mask.sum().item()
            total_length = len(input_ids)
            
            logging.info(f"总token数量: {total_length}")
            logging.info(f"有效token数量: {valid_length}")
            logging.info(f"padding token数量: {total_length - valid_length}")
            
            current_mlm_prob = collator._get_mlm_probability()
            logging.info(f"当前MLM概率: {current_mlm_prob:.3f}")
            
            logging.info("\n详细Token信息:")
            logging.info("格式: [位置] Token名称 (ID) | 输入状态 | MLM标签 | 变化标签 | 注意力")
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
                change_label = token_change_labels[i].item()
                attention = attention_mask[i].item()
                
                # 判断是否为padding
                is_padding = (attention == 0)
                
                if is_padding:
                    # padding token
                    status = "PAD"
                    mlm_label_info = f"标签:{label}"
                    change_info = f"变化:{change_label}"
                    pad_count += 1
                else:
                    # 有效token
                    if label == -100:  # 未被选中进行MLM的位置
                        status = "原始"
                        mlm_label_info = "标签:无(-100)"
                        change_info = f"变化:{change_label}"
                        original_count += 1
                    else:  # 被选中进行MLM的位置
                        original_token = tokenizer.convert_ids_to_tokens([label])[0]
                        mlm_label_info = f"标签:{original_token}({label})"
                        
                        if change_label == -100:  # 被mask的位置
                            status = f"MASK"
                            change_info = "变化:MASK(-100)"
                            masked_count += 1
                        elif change_label == 1:  # 被随机替换的位置
                            status = f"随机替换"
                            change_info = "变化:随机(1)"
                            random_count += 1
                        else:  # 保持不变的位置
                            status = "保持原样"
                            change_info = "变化:保持(0)"
                            keep_count += 1
                
                # 显示token信息
                attention_info = f"注意力:{attention}"
                logging.info(f"[{i:2d}] {token:15} ({token_id:5d}) | {status:10} | {mlm_label_info:20} | {change_info:15} | {attention_info}")
            
            # 详细统计信息
            logging.info(f"\n=== 统计信息 ===")
            logging.info(f"总token数量: {total_length}")
            logging.info(f"有效token数量: {valid_length}")
            logging.info(f"padding token数量: {pad_count}")
            logging.info(f"")
            logging.info(f"MLM处理统计:")
            logging.info(f"  - 原始未处理: {original_count}")
            logging.info(f"  - MASK替换: {masked_count}")
            logging.info(f"  - 随机替换: {random_count}")
            logging.info(f"  - 保持原样: {keep_count}")
            logging.info(f"  - 总MLM处理: {masked_count + random_count + keep_count}")
            logging.info(f"")
            logging.info(f"比例统计:")
            logging.info(f"  - 当前MLM概率设置: {current_mlm_prob:.3f}")
            if valid_length > 0:
                actual_mlm_ratio = (masked_count + random_count + keep_count) / valid_length
                mask_ratio = masked_count / valid_length
                random_ratio = random_count / valid_length
                keep_ratio = keep_count / valid_length
                
                logging.info(f"  - 实际MLM处理比例: {actual_mlm_ratio:.3f}")
                logging.info(f"  - MASK比例: {mask_ratio:.3f}")
                logging.info(f"  - 随机替换比例: {random_ratio:.3f}")
                logging.info(f"  - 保持原样比例: {keep_ratio:.3f}")
            
            # 显示标签分布
            label_distribution = {}
            change_label_distribution = {}
            
            for i in range(total_length):
                label = labels[i].item()
                change_label = token_change_labels[i].item()
                
                label_distribution[label] = label_distribution.get(label, 0) + 1
                change_label_distribution[change_label] = change_label_distribution.get(change_label, 0) + 1
            
            logging.info(f"\n=== 标签分布 ===")
            logging.info(f"MLM标签分布: {label_distribution}")
            logging.info(f"变化标签分布: {change_label_distribution}")
            
            # 显示原始文本和处理后文本的对比
            try:
                # 重建原始文本
                original_ids = input_ids.clone()
                for i in range(total_length):
                    if labels[i].item() != -100:
                        original_ids[i] = labels[i]
                
                # 只显示有效部分的文本
                original_text = tokenizer.decode(original_ids[:valid_length], skip_special_tokens=True)
                current_text = tokenizer.decode(input_ids[:valid_length], skip_special_tokens=True)
                
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
    检查指定的输出目录下是否存在类似 checkpoint- 的文件夹。
    """
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, item)) and re.match(r"^checkpoint-", item)
        for item in os.listdir(output_dir)
    )

def main():
    # --- 1. 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="使用可配置参数和分块策略训练一个MLM模型")

    # 路径参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="预训练模型或本地模型/分词器的路径。")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="模型 checkpoints 和输出的保存路径。")

    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=0.25, help="Lazy MLM provider 的初始 masking 概率。")
    parser.add_argument("--mlm_end_prob", type=float, default=0.15, help="Lazy MLM provider 的最终 masking 概率。")
    parser.add_argument("--random_probability", type=float, default=0.1, help="在被选中的token中，替换为随机token的比例")
    parser.add_argument("--mask_probability", type=float, default=0.8, help="在被选中的token中，替换为[MASK]的比例")
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine', help="Lazy MLM provider 的概率调度类型 (e.g., 'cosine', 'linear')。")

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512, help="输入序列的最大长度，也是分块的目标长度。")
    # 新增参数
    parser.add_argument("--chunk_stride", type=int, default=400, help="滑动窗口分块的步长 (stride)。重叠大小为 max_length - chunk_stride。")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练的总轮数。")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数。")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="学习率预热的比例。")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="数据加载器使用的工作进程数。")
    parser.add_argument("--save_total_limit", type=int, default=1, help="最多保存的 checkpoint 数量。")
    parser.add_argument("--logging_steps", type=int, default=10, help="每隔多少步记录一次日志。")
    parser.add_argument("--save_steps", type=int, default=500, help="每隔多少步保存一次 checkpoint。")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子。")
    parser.add_argument("--bf16", action='store_true', help="如果设置，则使用 bfloat16 混合精度训练。")
    
    args = parser.parse_args()

    # --- 2. 保存参数配置 ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    logging.info(f"所有参数已保存至: {os.path.join(args.output_dir, 'training_args.json')}")

    # --- 3. 加载模型和分词器 ---
    logging.info(f"从路径 '{args.model_name_or_path}' 加载分词器和模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 你的模型加载逻辑
    config = AutoConfig.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json")
    config.attn_implementation="flash_attention_2"
    model = ModernBertForDiffusionLM(config)

    # 确保 eos_token 设置正确，这对分块很重要
    if tokenizer.eos_token is None:
        if tokenizer.pad_token is not None:
             tokenizer.eos_token = tokenizer.pad_token # 常见做法
             logging.warning(f"Tokenizer没有eos_token，已将其设置为pad_token: {tokenizer.eos_token}")
        else:
             # 如果连pad_token都没有，则添加一个
             tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
             model.resize_token_embeddings(len(tokenizer))
             logging.info(f"Tokenizer没有eos_token，已添加新token并设置为: {tokenizer.eos_token}")


    # --- 4. 加载和处理数据集 ---
    logging.info(f"从路径 '{args.dataset_path}' 加载数据集...")
    dataset = datasets.load_dataset(args.dataset_path, split='train')

    logging.info("开始对数据集进行 Tokenization 和分块...")

    def tokenize_and_chunk(examples):
        tokenized_outputs = tokenizer(
            examples['text'], add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )

        all_chunked_input_ids = []
        for input_ids in tokenized_outputs['input_ids']:
            if not input_ids: continue
            
            chunk_size = args.max_length - 1
            for i in range(0, len(input_ids), args.chunk_stride):
                chunk = input_ids[i : i + chunk_size]
                chunk.append(tokenizer.eos_token_id)
                all_chunked_input_ids.append(chunk)

        return {"input_ids": all_chunked_input_ids}

    processed_dataset = dataset.map(
        tokenize_and_chunk, batched=True, remove_columns=dataset.column_names,
        num_proc=args.dataloader_num_workers,
    )
    
    logging.info(f"分块完成。原始样本数: {len(dataset)}, 分块后样本数: {len(processed_dataset)}")

    # --- 5. 设置 MLM Collator 和 Callback ---
    lazy_prob_provider = LazyScheduledMLMProbProvider(
        start_prob=args.mlm_start_prob, end_prob=args.mlm_end_prob,
        schedule_type=args.mlm_schedule_type
    )
    lazy_prob_scheduler_callback = LazyMLMProbSchedulerCallback(lazy_prob_provider)
    collator = MLMCollator(
        tokenizer, mlm_probability=lazy_prob_provider, max_length=args.max_length,
        mask_probability=args.mask_probability, random_probability=args.random_probability
    )

    # --- 6. 设置训练参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False, # 设为False，配合resume_from_checkpoint更安全
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': 0.1},
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        per_device_train_batch_size=args.per_device_train_batch_size,
        remove_unused_columns=False, # Collator会处理，设为False
        ddp_find_unused_parameters=False,
    )

    # --- 7. 初始化并开始训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset, # 使用处理后的数据集
        data_collator=collator,
        callbacks=[lazy_prob_scheduler_callback]
    )

    # debug_data(trainer)

    logging.info("\n准备开始正式训练...\n")

    # 检查是否有checkpoint可以恢复
    resume_from_checkpoint = True if check_for_checkpoints(args.output_dir) else None
    if resume_from_checkpoint:
        logging.info(f"发现checkpoint，将从 {args.output_dir} 恢复训练。")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    logging.info("训练完成，最终模型已保存。")


if __name__ == "__main__":
    main()