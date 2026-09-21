import argparse
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

def parse_run_key(key: str) -> Dict[str, Any]:
    """
    从 summary.json 的键中解析出任务名、few-shot数等信息。
    例如: "hellaswag_0shot_causal" -> {'task': 'hellaswag', 'fewshot': 0, 'type': 'causal'}
    """
    parts = key.split('_')
    parsed = {}
    try:
        parsed['task'] = parts[0]
        # 寻找包含 "shot" 的部分
        shot_part = next((p for p in parts if "shot" in p), "0shot")
        parsed['fewshot'] = int(shot_part.replace("shot", ""))
        return parsed
    except (IndexError, ValueError) as e:
        print(f"[Warning] Could not parse run key: '{key}'. Skipping. Error: {e}")
        return None

def gather_data(results_dir: Path) -> List[Dict[str, Any]]:
    """
    遍历 results 目录，收集所有 summary.json 文件中的数据。
    """
    all_runs_data = []
    summary_files = list(results_dir.rglob('summary.json'))

    if not summary_files:
        print(f"No 'summary.json' files found in '{results_dir}'.")
        return []

    print(f"Found {len(summary_files)} summary files. Processing...")

    for summary_file in summary_files:
        # model_name 是 summary.json 所在的目录名
        model_name = summary_file.parent.name
        print(f"  - Processing model: {model_name}")

        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for run_key, run_details in data.items():
                parsed_key = parse_run_key(run_key)
                if not parsed_key:
                    continue
                
                # 创建一个扁平化的记录
                flat_record = {
                    'model_name': model_name,
                    'task_name': parsed_key['task'],
                    'num_fewshot': parsed_key['fewshot'],
                }
                
                # 添加所有指标
                flat_record.update(run_details.get('metrics', {}))
                
                all_runs_data.append(flat_record)

        except json.JSONDecodeError:
            print(f"[Error] Failed to decode JSON from {summary_file}. Skipping.")
        except Exception as e:
            print(f"[Error] An unexpected error occurred while processing {summary_file}: {e}")
            
    return all_runs_data

def create_pivot_table(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将扁平化的数据转换成一个适合阅读的透视表。
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # 创建一个用于多级列索引的显示名称
    df['task_display'] = df.apply(
        lambda row: f"{row['task_name']} ({row['num_fewshot']}-shot)", axis=1
    )
    
    # 确定所有出现过的指标列
    metric_columns = [
        col for col in df.columns 
        if col not in ['model_name', 'task_name', 'num_fewshot', 'task_display']
    ]

    if not metric_columns:
        print("[Warning] No metric columns found to create a pivot table.")
        return pd.DataFrame()

    # 创建数据透视表
    # index 是行, columns 是列, values 是填充单元格的值
    pivot_df = df.pivot_table(
        index='model_name',
        columns='task_display',
        values=metric_columns
    )
    
    # 交换列的层级，使得任务在第一层，指标在第二层，更易读
    # (metric, task_display) -> (task_display, metric)
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)
    
    # 按列名排序，让相同任务的指标聚在一起
    pivot_df.sort_index(axis=1, level=0, inplace=True)
    
    return pivot_df

def main():
    parser = argparse.ArgumentParser(
        description="Summarize model evaluation results from the 'results' directory into an Excel file."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="The base directory where evaluation results are stored."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_summary.xlsx",
        help="Path to the output Excel file."
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_file)

    if not results_path.is_dir():
        print(f"Error: The specified results directory '{results_path}' does not exist.")
        return

    # 1. 收集数据
    flat_data = gather_data(results_path)

    if not flat_data:
        print("No data collected. Exiting.")
        return

    # 2. 创建透视表
    summary_table = create_pivot_table(flat_data)

    if summary_table.empty:
        print("Failed to create a summary table. Exiting.")
        return

    # 3. 保存到 Excel
    try:
        summary_table.to_excel(output_path)
        print(f"\n✓ Successfully generated summary table at: {output_path.absolute()}")
    except Exception as e:
        print(f"\n✗ Error saving Excel file: {e}")


if __name__ == "__main__":
    main()