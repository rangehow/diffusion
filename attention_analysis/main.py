import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import warnings

# 忽略一些不影响结果的警告
warnings.filterwarnings("ignore")

# --- 1. 配置 ---
# 请确保路径正确
model_path = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-1.7B-Base/main"
dataset_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/datasets/filtered_finefineweb"

# --- 2. 实验参数 ---
MAX_LENGTH = 1024
NUM_SAMPLES = 1000 # 为了快速演示，可以减少样本量，例如 100

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, _attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model and tokenizer loaded.")

print("Loading dataset...")
dataset = datasets.load_from_disk(dataset_path).select(range(NUM_SAMPLES))
print("Dataset loaded.")

# 用于存储所有样本中，不同相对距离的注意力值
relative_position_attentions = defaultdict(list)

# --- 3. 提取和聚合注意力 ---
print(f"Processing {NUM_SAMPLES} samples with max length {MAX_LENGTH}...")
model.eval()
with torch.no_grad():
    for sample in tqdm(dataset, total=NUM_SAMPLES):
        if 'text' not in sample: continue
        
        text = sample['text']
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        if inputs['input_ids'].shape[0] != 1: continue
            
        stacked_attentions = torch.stack(attentions)
        avg_attention = stacked_attentions.mean(dim=(0, 2)).squeeze(0)
        avg_attention = avg_attention.cpu().float().numpy()
        seq_len = avg_attention.shape[0]
        
        for i_pos in range(seq_len):
            for j_pos in range(seq_len):
                if j_pos <= i_pos:
                    relative_dist = i_pos - j_pos
                    relative_position_attentions[relative_dist].append(avg_attention[i_pos, j_pos])

print("Attention scores aggregated.")

# --- 4. 数据处理和拟合 ---
print("Fitting the models to the aggregated data...")

distances = sorted(relative_position_attentions.keys())
avg_attentions = [np.mean(relative_position_attentions[d]) for d in distances]

x_data = np.array(distances)
y_data = np.array(avg_attentions)

# --- 模型定义 ---
# 模型1: 指数衰减 P(d) = λ * exp(-α * d) + C
def attention_model_exponential(d, lambda_p, alpha, C):
    return lambda_p * np.exp(-alpha * d) + C

# 模型2: 几何衰减 P(d) = λ * r^d + C
def attention_model_geometric(d, lambda_p, r, C):
    return lambda_p * (r ** d) + C

# --- 拟合与误差计算 ---
results = {}

try:
    # --- 拟合指数模型 ---
    print("\nFitting Exponential Model...")
    initial_guess_exp = [0.8, 0.1, 1e-4]
    bounds_exp = ([0, 0, 0], [1, np.inf, 1])
    params_exp, _ = curve_fit(attention_model_exponential, x_data, y_data, p0=initial_guess_exp, bounds=bounds_exp)
    
    # 计算误差
    y_pred_exp = attention_model_exponential(x_data, *params_exp)
    rmse_exp = np.sqrt(np.mean((y_data - y_pred_exp)**2))
    
    results['exponential'] = {'params': params_exp, 'rmse': rmse_exp, 'y_pred': y_pred_exp}
    
    # --- 拟合几何模型 ---
    print("Fitting Geometric Model...")
    initial_guess_geom = [0.8, 0.9, 1e-4] # r 的初值设为接近1
    bounds_geom = ([0, 0, 0], [1, 1, 1]) # r 的范围是 (0, 1)
    params_geom, _ = curve_fit(attention_model_geometric, x_data, y_data, p0=initial_guess_geom, bounds=bounds_geom)

    # 计算误差
    y_pred_geom = attention_model_geometric(x_data, *params_geom)
    rmse_geom = np.sqrt(np.mean((y_data - y_pred_geom)**2))
    
    results['geometric'] = {'params': params_geom, 'rmse': rmse_geom, 'y_pred': y_pred_geom}
    
    # --- 打印结果 ---
    # 指数模型结果
    lambda_fit_exp, alpha_fit_exp, C_fit_exp = results['exponential']['params']
    U_fit_exp = C_fit_exp / (1 - lambda_fit_exp) if (1 - lambda_fit_exp) != 0 else 0
    print("\n--- Fit Results (Exponential) ---")
    print(f"Fitted λ (Lambda): {lambda_fit_exp:.4f}")
    print(f"Fitted α (Alpha):  {alpha_fit_exp:.4f}")
    print(f"Fitted C (Constant): {C_fit_exp:.6f}")
    print(f"Implied U:           {U_fit_exp:.6f}")
    print("-----------------------------------")
    
    # 几何模型结果
    lambda_fit_geom, r_fit_geom, C_fit_geom = results['geometric']['params']
    U_fit_geom = C_fit_geom / (1 - lambda_fit_geom) if (1 - lambda_fit_geom) != 0 else 0
    # 为了方便比较，我们可以从 r 计算出等效的 alpha
    implied_alpha_from_r = -np.log(r_fit_geom)
    print("\n--- Fit Results (Geometric) ---")
    print(f"Fitted λ (Lambda): {lambda_fit_geom:.4f}")
    print(f"Fitted r (Ratio):  {r_fit_geom:.4f}")
    print(f"Fitted C (Constant): {C_fit_geom:.6f}")
    print(f"Implied U:           {U_fit_geom:.6f}")
    print(f"Implied α (-log(r)): {implied_alpha_from_r:.4f} (for comparison with exponential model)")
    print("---------------------------------")
    
    # 误差比较
    print("\n--- Model Fit Error (RMSE) ---")
    print(f"Exponential Model RMSE: {results['exponential']['rmse']:.8f}")
    print(f"Geometric Model RMSE:   {results['geometric']['rmse']:.8f}")
    if results['exponential']['rmse'] < results['geometric']['rmse']:
        print("=> Exponential model provides a slightly better fit.")
    else:
        print("=> Geometric model provides a slightly better fit.")
    print("--------------------------------\n")
    

    # --- 5. 可视化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    # 绘制经验数据
    ax1.scatter(x_data, y_data, label='Empirical Data (Model Avg)', s=15, alpha=0.7, color='royalblue', zorder=3)
    ax2.scatter(x_data, y_data, label='Empirical Data (Model Avg)', s=15, alpha=0.7, color='royalblue', zorder=3)

    # 绘制拟合曲线
    if 'exponential' in results:
        ax1.plot(x_data, results['exponential']['y_pred'], label=f'Exponential Fit (RMSE={results["exponential"]["rmse"]:.4g})', color='darkorange', linewidth=2.5)
        ax2.plot(x_data, results['exponential']['y_pred'], label=f'Exponential Fit', color='darkorange', linewidth=2.5)
    if 'geometric' in results:
        ax1.plot(x_data, results['geometric']['y_pred'], label=f'Geometric Fit (RMSE={results["geometric"]["rmse"]:.4g})', color='forestgreen', linewidth=2.5, linestyle='--')
        ax2.plot(x_data, results['geometric']['y_pred'], label=f'Geometric Fit', color='forestgreen', linewidth=2.5, linestyle='--')

    # 线性坐标图
    ax1.set_title('Attention Probability vs. Relative Position (Linear Scale)', fontsize=16)
    ax1.set_ylabel('Average Attention Probability', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 对数坐标图
    ax2.set_yscale('log')
    ax2.set_title('Attention Probability vs. Relative Position (Log Scale)', fontsize=16)
    ax2.set_xlabel('Relative Distance d = |i-j|', fontsize=12)
    ax2.set_ylabel('Average Attention Probability (log scale)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("attention_pattern_comparison.png")
    print("Plot saved as attention_pattern_comparison.png")
    plt.show()

except RuntimeError as e:
    print(f"\nCould not fit one or both of the curves. Error: {e}")
    print("This might happen if the data pattern is too complex for the models.")
    print("Plotting the raw data anyway.")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, s=10)
    plt.title('Empirical Attention Probability vs. Relative Position')
    plt.xlabel('Relative Distance |i-j|')
    plt.ylabel('Average Attention Probability')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()