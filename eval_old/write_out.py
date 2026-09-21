# write_out_text.py

import argparse
import json
from pathlib import Path
from tqdm import tqdm

# 从新的 task.py 导入工厂函数和配置类
from task import get_task, TaskConfig

def get_model_short_name(model_path: str) -> str:
    """复用 main.py 中的函数，用于创建目录"""
    path = Path(model_path)
    model_name = path.parent.name if path.name.startswith('checkpoint-') else path.name
    return model_name.split('/')[-1]

def create_output_structure(output_dir: str, model_path: str) -> Path:
    """创建输出目录结构"""
    model_short_name = get_model_short_name(model_path)
    model_dir = Path(output_dir) / f"write_out_text_{model_short_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Process and write out evaluation data as plain text to a JSONL file.")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name used for naming the output directory.")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of tasks to process.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for testing. 0 for no limit.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples to use.")
    parser.add_argument("--sampler", type=str, default="random", choices=['first_n', 'random', 'balanced'], help="Few-shot sampler to use.")
    parser.add_argument("--sampler_seed", type=int, default=42, help="Seed for the few-shot sampler.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the output JSONL file.")
    
    args = parser.parse_args()
    print("Arguments:", args)
    
    output_dir = create_output_structure(args.output_dir, args.model_name_or_path)
    print(f"Output data will be saved to: {output_dir}")

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    for task_name in tasks:
        print(f"\n----- Processing data for task: {task_name} ({args.num_fewshot}-shot) -----")
        
        # 1. 配置任务
        config = TaskConfig(
            num_fewshot=args.num_fewshot,
            sampler_name=args.sampler,
            sampler_seed=args.sampler_seed,
            text_only=True  # <-- 启用纯文本模式
        )

        try:
            # ******************** 这是修改的核心 ********************
            # 使用新的工厂函数获取任务实例，然后调用 .process() 方法
            task_instance = get_task(task_name, config)
            dataset = task_instance.process()
            # ******************************************************
        except ValueError as e:
            print(f"Error: {e}. Skipping task '{task_name}'.")
            continue
            
        if args.limit > 0:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

        # 写入文件的逻辑保持不变
        output_filename = f"{task_name}_{args.num_fewshot}shot_text.jsonl"
        output_file_path = output_dir / output_filename

        count = 0
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Writing {task_name} text data"):
                context = item.get("context_text", "")
                continuation = item.get("continuation_text", "")
                full_prompt = context + continuation
                output_record = {"prompt": full_prompt}
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                count += 1
        
        print(f"✓ Successfully wrote {count} records for task '{task_name}' to: {output_file_path}")

    print("\nAll tasks processed and text data written out.")

if __name__ == "__main__":
    main()