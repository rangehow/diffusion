import os
import json
import math
import pandas as pd

def generate_pivot_excel(root_path, output_excel_name="diffusion_benchmark_v2.xlsx"):
    all_data = []
    
    print(f"正在扫描路径: {root_path} ...")
    
    if not os.path.exists(root_path):
        print(f"错误: 路径不存在 -> {root_path}")
        return

    # 1. 遍历并收集数据
    for folder_name in os.listdir(root_path):
        folder_full_path = os.path.join(root_path, folder_name)
        
        if os.path.isdir(folder_full_path):
            json_file_path = os.path.join(folder_full_path, "eval_results.json")
            
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # --- 1. 获取 Dataset 名称 (作为列) ---
                    dataset = data.get("dataset_name", "unknown_dataset")
                    
                    # --- 2. 获取 Model 名称 (作为行) [核心修改部分] ---
                    ckpt_path = data.get("checkpoint_path", "")
                    
                    if ckpt_path:
                        # 去除末尾可能存在的斜杠
                        ckpt_path = ckpt_path.rstrip('/')
                        base_name = os.path.basename(ckpt_path)
                        
                        # 判断逻辑：如果最后一部分是 checkpoint-xxxx，则取上一级目录名
                        if base_name.startswith("checkpoint-"):
                            # os.path.dirname 获取父目录路径
                            # os.path.basename 获取父目录的名字 (即模型名)
                            parent_path = os.path.dirname(ckpt_path)
                            model_name = os.path.basename(parent_path)
                        else:
                            # 否则直接用最后一部分 (针对 arm_main_exp 这种情况)
                            model_name = base_name
                    else:
                        # 如果没路径，用外层文件夹名兜底
                        model_name = folder_name 

                    # --- 3. 计算 PPL ---
                    eval_loss = data.get("eval_loss")
                    ppl = None
                    if eval_loss is not None:
                        try:
                            ppl = math.exp(eval_loss)
                        except OverflowError:
                            ppl = float('inf')

                    # --- 4. 获取耗时 ---
                    runtime = data.get("eval_runtime")

                    # 添加到列表
                    all_data.append({
                        "Model": model_name,
                        "Dataset": dataset,
                        "PPL": ppl,
                        "Runtime": runtime
                    })
                    
                except Exception as e:
                    print(f"处理文件失败: {folder_name}, 错误: {e}")

    # 2. 生成表格
    if not all_data:
        print("未找到有效数据。")
        return

    df = pd.DataFrame(all_data)

    # 3. 数据透视 (Pivot)
    
    # Sheet 1: PPL
    # index=模型(行), columns=数据集(列), values=PPL
    df_ppl = df.pivot_table(index="Model", columns="Dataset", values="PPL", aggfunc='first')
    df_ppl = df_ppl.round(2) # 保留两位小数

    # Sheet 2: Runtime
    df_runtime = df.pivot_table(index="Model", columns="Dataset", values="Runtime", aggfunc='first')
    df_runtime = df_runtime.round(2) # 保留两位小数

    # 4. 写入 Excel
    output_path = os.path.join(os.getcwd(), output_excel_name)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_ppl.to_excel(writer, sheet_name='PPL')
            df_runtime.to_excel(writer, sheet_name='Runtime')
            
        print(f"\n成功生成表格: {output_path}")
        print(f"包含 {len(df)} 条实验记录。")
        print("已自动处理 'checkpoint-xxxx' 路径，提取上一级作为模型名。")
        
    except Exception as e:
        print(f"写入Excel失败: {e}")

if __name__ == "__main__":
    TARGET_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/eval_output"
    generate_pivot_excel(TARGET_PATH)