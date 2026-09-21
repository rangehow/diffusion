# eval_loss.py
import argparse
import json
import logging
import os
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    LlamaForCausalLM,
)

from ..collators import (
    NTPCollator,
    MDLMCollator,
    CausalMLMCollator, 
    BD3LMCollator
)

from ..modeling.modeling_niu import ModernBertForDiffusionLM
from ..modeling.configuration_niu import NiuConfig
from ..llada.modeling_llada import LLaDAModelLM
from ..llada.configuration_llada import LLaDAConfig
from ..bd3lm.modeling_bd3lm import BD3LM
from ..bd3lm.configuration_bd3lm import BD3LMConfig
from ..trainer import EMATrainer
from ..utils.load_dataset import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

from datasets import disable_caching

# 禁用缓存，之后的所有数据处理操作将不再保存中间结果到磁盘
disable_caching()
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


def load_model(checkpoint_path, mode):
    """Load model from checkpoint based on mode."""
    if mode == 'llada' or mode == 'mdlm':
        config = LLaDAConfig.from_pretrained(checkpoint_path)
        model = LLaDAModelLM.from_pretrained(checkpoint_path, config=config)
    elif mode == 'llama':
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = LlamaForCausalLM.from_pretrained(checkpoint_path, config=config)
    elif mode == 'bd3lm':
        config = BD3LMConfig.from_pretrained(checkpoint_path)
        model = BD3LM.from_pretrained(checkpoint_path, config=config)
    else:  # niu (default)
        config = NiuConfig.from_pretrained(checkpoint_path)
        model = ModernBertForDiffusionLM.from_pretrained(checkpoint_path, config=config)
    
    return model, config


def load_ema_weights_into_model(model, checkpoint_path, device):
    """
    Load EMA weights from ema_state.pt and copy them into the model.
    
    Returns:
        bool: True if EMA weights were loaded, False otherwise
    """
    ema_path = os.path.join(checkpoint_path, "ema_state.pt")
    
    if not os.path.exists(ema_path):
        return False
    
    if is_main_process():
        logging.info(f"发现 EMA 状态文件: {ema_path}")
        logging.info("正在加载 EMA 权重...")
    
    # Load EMA state
    map_location = device if isinstance(device, str) else str(device)
    ema_state = torch.load(ema_path, map_location=map_location)
    
    # Get the shadow params from EMA state
    shadow_params = ema_state['shadow_params']
    
    # Copy EMA weights to model
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(shadow_params) != len(model_params):
        logging.warning(f"EMA 参数数量 ({len(shadow_params)}) 与模型参数数量 ({len(model_params)}) 不匹配!")
        logging.warning("将使用原始模型权重进行评估")
        return False
    
    with torch.no_grad():
        for shadow_param, model_param in zip(shadow_params, model_params):
            model_param.data.copy_(shadow_param.data)
    
    if is_main_process():
        logging.info(f"EMA 权重加载成功 (decay={ema_state.get('decay', 'unknown')}, "
                     f"num_updates={ema_state.get('num_updates', 'unknown')})")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="使用 Trainer 评估预训练模型的 loss")

    # 路径参数
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="预训练模型 checkpoint 的路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径（默认使用 checkpoint_path）")
    parser.add_argument("--dataset_name", type=str, default="sh_openwebtext",
                        help="评估数据集的名称")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="评估结果输出路径")
    parser.add_argument("--mode", type=str, default="niu",
                        choices=["niu", "llada", "mdlm", "llama", "bd3lm"],
                        help="模型类型")

    # EMA 参数
    parser.add_argument("--use_ema", type=lambda x: x.lower() == 'true', default=True,
                        help="是否使用 EMA 权重进行评估（如果存在）。默认 True")
    parser.add_argument("--force_ema", action="store_true",
                        help="如果设置，当 EMA 权重不存在时报错退出")

    # MLM Schedule 参数 (用于 collator)
    parser.add_argument("--mlm_start_prob", type=float, default=1.0)
    parser.add_argument("--mlm_end_prob", type=float, default=0.0001)
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)
    parser.add_argument("--use_daum", type=lambda x: x.lower() == 'true', default=True)

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--pad_to_max_length", action="store_true")

    # 评估参数
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="最大评估样本数（None 表示全部）")

    parser.add_argument("--bf16", action='store_true')

    args = parser.parse_args()

    # 打印配置
    if is_main_process():
        logging.info("=" * 80)
        logging.info("评估参数配置:")
        logging.info("=" * 80)
        for key, value in vars(args).items():
            logging.info(f"{key:30}: {value}")
        logging.info("=" * 80)

    # 加载分词器
    tokenizer_path = args.checkpoint_path
    if is_main_process():
        logging.info(f"从路径 '{tokenizer_path}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 处理特殊 token
    if 'modernbert' in tokenizer_path.lower():
        if tokenizer.eos_token is None:
            tokenizer.eos_token_id = 50279
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = 50285
    elif 'gpt' in tokenizer_path.lower():
        tokenizer.add_tokens(['<|begin_of_text|>', '[MASK]'])
        tokenizer.bos_token = '<|begin_of_text|>'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token is None:
            tokenizer.eos_token_id = 50279
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = 50285
    elif 'bert' in tokenizer_path.lower():
        tokenizer.add_tokens(['<|begin_of_text|>', '<|end_of_text|>'])
        tokenizer.bos_token = '<|begin_of_text|>'
        tokenizer.eos_token = '<|end_of_text|>'

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    if is_main_process():
        logging.info(f"从路径 '{args.checkpoint_path}' 加载模型...")
    model, config = load_model(args.checkpoint_path, args.mode)
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"模型加载成功。参数量: {total_params:,}")

    # 尝试加载 EMA 权重
    ema_loaded = False
    if args.use_ema:
        ema_loaded = load_ema_weights_into_model(model, args.checkpoint_path, device)
        
        if not ema_loaded:
            if args.force_ema:
                raise FileNotFoundError(
                    f"EMA 权重文件不存在: {os.path.join(args.checkpoint_path, 'ema_state.pt')}\n"
                    "使用 --force_ema 时必须存在 EMA 权重文件"
                )
            elif is_main_process():
                logging.warning("未找到 EMA 权重文件，将使用原始模型权重进行评估")
    else:
        if is_main_process():
            logging.info("--use_ema=False，使用原始模型权重进行评估")

    # 加载数据集
    if is_main_process():
        logging.info(f"加载评估数据集 '{args.dataset_name}'...")
    eval_dataset = get_dataset(args.dataset_name)
    
    # 如果指定了最大样本数，进行截断
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    
    if is_main_process():
        logging.info(f"评估数据集大小: {len(eval_dataset)}")

    # 创建 collator
    if args.mode == 'llama':
        collator = NTPCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == 'mdlm':
        collator = MDLMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    elif args.mode == 'bd3lm':
        collator = BD3LMCollator(
            tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length
        )
    else:  # niu, llada
        collator = CausalMLMCollator(
            tokenizer,
            max_length=args.max_length,
            start_prob=args.mlm_start_prob,
            end_prob=args.mlm_end_prob,
            tail_bias_factor=args.tail_bias_factor,
            use_daum=args.use_daum,
            pad_to_max_length=args.pad_to_max_length,
            is_eval=True,  # 重要：使用 eval 模式
        )

    # 创建 TrainingArguments (仅用于评估)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        report_to='none',
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    # 创建 Trainer (不启用 EMA，因为我们已经手动加载了 EMA 权重)
    trainer = EMATrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        ema_decay=None,  # 禁用 EMA，因为我们已经手动处理了
    )

    # 运行评估
    if is_main_process():
        logging.info("开始评估...")
        logging.info(f"使用权重类型: {'EMA' if ema_loaded else '原始模型'}")
    
    metrics = trainer.evaluate()

    # 添加额外信息到 metrics
    metrics['weights_type'] = 'ema' if ema_loaded else 'original'
    metrics['checkpoint_path'] = args.checkpoint_path
    metrics['dataset_name'] = args.dataset_name

    # 打印结果
    if is_main_process():
        logging.info("=" * 80)
        logging.info("评估结果:")
        logging.info("=" * 80)
        for key, value in metrics.items():
            logging.info(f"{key:30}: {value}")
        logging.info("=" * 80)

        # 保存结果到文件
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        logging.info(f"评估结果已保存至: {results_path}")


if __name__ == "__main__":
    main()