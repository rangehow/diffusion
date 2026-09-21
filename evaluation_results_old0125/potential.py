import os
import json
import pandas as pd
import re

# 1. Configuration
BASE_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/evaluation_results/"
# 定义模型前缀
MODEL_PREFIXES = ["arm", "bd3lm", "mdlm", "niu"]
# 定义文件夹后缀
DIR_SUFFIX = "_1B_fineweb_edu_100b_potential"
OUTPUT_FILE = "potential.xlsx"

def format_score(value):
    """Converts 0.3427 -> 34.28"""
    if value is None:
        return None
    try:
        return round(float(value) * 100, 2)
    except:
        return None

def get_checkpoint_step(path_str):
    """
    从路径中提取 checkpoint 的步数用于排序。
    例如: ".../checkpoint-5000/..." -> 5000
    如果没有找到，返回 -1
    """
    match = re.search(r'checkpoint-(\d+)', path_str)
    if match:
        return int(match.group(1))
    return -1

def extract_all_data():
    raw_rows = []
    
    # 遍历定义的模型前缀
    for prefix in MODEL_PREFIXES:
        # 拼接完整文件夹名称
        folder_name = f"{prefix}{DIR_SUFFIX}"
        model_dir = os.path.join(BASE_PATH, folder_name)
        
        # 显示名称仅保留前缀 (如 arm, bd3lm) 以便图表整洁，也可以用 folder_name
        display_name = prefix 
        
        if not os.path.exists(model_dir):
            print(f"Skipping missing directory: {model_dir}")
            continue

        # 遍历目录
        for root, dirs, files in os.walk(model_dir):
            if "index.json" in files:
                json_path = os.path.join(root, "index.json")
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        
                    if isinstance(content, list) and len(content) > 0:
                        data = content[0]
                    else:
                        continue

                    params = data.get("params", {})
                    metrics = data.get("metrics", {})
                    
                    # 1. 过滤任务：只保留 hellaswag
                    task_name = params.get("task", "")
                    if "hellaswag" not in task_name.lower():
                        continue

                    # 2. 提取 Checkpoint 步数
                    # 从 root 路径中查找 checkpoint-xxx
                    step = get_checkpoint_step(root)
                    if step == -1:
                        # 如果是在根目录或者没有checkpoint文件夹，标记为 Final 或 0
                        checkpoint_label = "Final"
                        step_sort_key = 999999999 # 放最后
                    else:
                        checkpoint_label = f"step-{step}"
                        step_sort_key = step

                    # 3. 提取分数 (HellaSwag 通常关注 acc 和 acc_norm)
                    # acc_norm 对应 acc_norm, acc_norm_token, 或 acc_norm_char
                    # 优先取 acc_norm, 其次 acc_norm_token/char
                    val_acc = metrics.get("acc")
                    
                    val_acc_norm = metrics.get("acc_norm")
                    if val_acc_norm is None:
                        val_acc_norm = metrics.get("acc_norm_token")
                    if val_acc_norm is None:
                        val_acc_norm = metrics.get("acc_norm_char")

                    raw_rows.append({
                        "Model": display_name,         # 行索引
                        "Step": step_sort_key,         # 用于列排序的数字
                        "Checkpoint": checkpoint_label, # 备用显示
                        "acc": format_score(val_acc),
                        "acc_norm": format_score(val_acc_norm)
                    })

                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
    
    return raw_rows

def create_pivot_df(raw_data, value_column):
    """
    创建透视表:
    Rows (Index): Model
    Cols: Step (Checkpoint number)
    Values: Score
    """
    df = pd.DataFrame(raw_data)
    if df.empty:
        return pd.DataFrame()
    
    # 使用 Step 数字作为列，保证 Excel 中列是按 1000, 2000, 3000... 排序的
    pivot_df = df.pivot_table(
        index="Model", 
        columns="Step", 
        values=value_column,
        aggfunc='first'
    )
    
    # 也可以选择把列名重命名回 "checkpoint-xxx" 格式，或者保留数字更简洁
    # 这里保留数字，并在Excel输出时稍微美化一下
    return pivot_df

def main():
    print(f"Scanning base path: {BASE_PATH}")
    data = extract_all_data()
    
    if not data:
        print("No 'hellaswag' data found matching the criteria.")
        return

    # 创建 HellaSwag 的 acc 和 acc_norm 两个 sheet
    df_acc = create_pivot_df(data, "acc")
    df_acc_norm = create_pivot_df(data, "acc_norm")

    # 保存
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_acc.to_excel(writer, sheet_name='Hellaswag_ACC')
        df_acc_norm.to_excel(writer, sheet_name='Hellaswag_ACC_Norm')

    print(f"Done! File saved to: {OUTPUT_FILE}")
    print(f"Processed {len(data)} records.")
    
    # 打印一下预览
    if not df_acc_norm.empty:
        print("\nPreview (ACC Norm):")
        print(df_acc_norm)

if __name__ == "__main__":
    main()