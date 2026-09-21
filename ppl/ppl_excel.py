import os
import json
import glob
import pandas as pd

# 设置你的目标文件夹路径
TARGET_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/ppl/generation_outputs/hellaswag_quality_cosine_merged"
OUTPUT_FILE = "ppl_summary_matrix.xlsx"

def main():
    # 查找文件
    file_pattern = os.path.join(TARGET_DIR, "ppl*.json")
    files = glob.glob(file_pattern)

    if not files:
        print(f"❌ 在路径 {TARGET_DIR} 下未找到 'ppl' 开头的 JSON 文件。")
        return

    all_data = []
    print(f"📂 找到 {len(files)} 个文件，开始解析...")

    # 1. 提取数据
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

                model_name = content.get('ppl_model_name', 'Unknown')
                results = content.get('results', {})

                for method_name, metrics in results.items():
                    all_data.append({
                        'Model': model_name,
                        'Method': method_name,
                        'PPL Mean': metrics.get('ppl_mean'),
                        'PPL Std': metrics.get('ppl_std'),
                        'Throughput': metrics.get('tokens_per_second')
                    })
        except Exception as e:
            print(f"⚠️ 跳过损坏文件 {os.path.basename(file_path)}: {e}")

    if not all_data:
        print("❌ 没有提取到有效数据。")
        return

    # 2. 转换为 DataFrame
    df = pd.DataFrame(all_data)

    # 3. 创建三个不同的透视表 (Rows=Model, Cols=Method)
    # pivot_table 会自动处理行列对齐，如果某个模型缺少某个方法的数据，会填充 NaN

    # Sheet 1: PPL Mean
    pivot_mean = df.pivot_table(index='Model', columns='Method', values='PPL Mean')

    # Sheet 2: PPL Std
    pivot_std = df.pivot_table(index='Model', columns='Method', values='PPL Std')

    # Sheet 3: Throughput (Tokens/sec)
    pivot_tps = df.pivot_table(index='Model', columns='Method', values='Throughput')

    # 4. 写入同一个 Excel 的不同 Sheet
    print(f"💾 正在写入 Excel 文件: {OUTPUT_FILE} ...")

    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            pivot_mean.to_excel(writer, sheet_name='PPL_Mean')
            pivot_std.to_excel(writer, sheet_name='PPL_Std')
            pivot_tps.to_excel(writer, sheet_name='Throughput')

        print(f"✅ 成功！文件已保存至当前目录: {os.path.abspath(OUTPUT_FILE)}")
        print("   Sheet 1: PPL_Mean (均值)")
        print("   Sheet 2: PPL_Std (标准差)")
        print("   Sheet 3: Throughput (吞吐速度)")

    except PermissionError:
        print("❌ 写入失败：请检查文件是否已被打开，关闭后重试。")

if __name__ == "__main__":
    main()