import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 检查依赖库 ---
try:
    from xformers.ops import SwiGLU
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xformers is not installed. Will only run the PyTorch version. Speed comparison is not possible.")

# --- 模型定义 (与之前相同) ---
class ModernBertConfig:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_activation = "silu"
        self.mlp_dropout = 0.0
        self.mlp_bias = False

ACT2FN = {"silu": nn.functional.silu}

class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        gated_output = self.act(input) * gate
        return self.Wo(self.drop(gated_output))


# --- 基准测试函数 ---
def benchmark(model, input_tensor, warmup_iter=10, test_iter=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iter):
            _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(test_iter):
            _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / test_iter) * 1000
    return avg_time_ms

# --- 一致性检查函数 ---
def check_output_consistency(model_pytorch, model_xformers, input_tensor):
    """检查两个模型输出的一致性 - 使用L2距离"""
    model_pytorch.eval()
    
    with torch.no_grad():
        output_pytorch = model_pytorch(input_tensor)
        output_xformers = model_xformers(input_tensor)
        
        # 计算L2距离
        l2_distance = torch.norm(output_pytorch - output_xformers, p=2).item()
        
        return {
            'l2_distance': l2_distance
        }

# --- 可视化函数 ---
def plot_results(df):
    """使用 seaborn 和 matplotlib 生成对比图表"""
    
    # 准备用于绘图的数据
    df_melted = df.melt(
        id_vars=['Config Label'], 
        value_vars=['PyTorch (ms)', 'xformers (ms)'], 
        var_name='Implementation', 
        value_name='Latency (ms)'
    )
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.barplot(
        x='Config Label', 
        y='Latency (ms)', 
        hue='Implementation', 
        data=df_melted, 
        ax=ax,
        palette={'PyTorch (ms)': 'cornflowerblue', 'xformers (ms)': 'salmon'}
    )
    
    # 在条形图上添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)

    ax.set_title('Performance Comparison: PyTorch MLP vs. xformers SwiGLU', fontsize=18, pad=20)
    ax.set_xlabel('Configuration (Batch-SeqLen-Hidden)', fontsize=12)
    ax.set_ylabel('Average Latency per Forward Pass (ms)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.legend(title='Implementation', fontsize=11)
    
    # 调整Y轴上限，留出空间给标签
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("Running on CPU. Performance differences will be less pronounced.")
        
    DTYPE = torch.float16

    # 定义多种测试配置
    test_configs = [
        # Small model (like BERT-base)
        {'batch': 16, 'seq': 512,  'hidden': 768,  'inter': 3072},
        # Medium model
        {'batch': 8,  'seq': 1024, 'hidden': 2048, 'inter': 5504},
        # Large model (like Llama-7B) with shorter sequence
        {'batch': 4,  'seq': 1024, 'hidden': 4096, 'inter': 11008},
        # Large model with longer sequence (more demanding)
        {'batch': 2,  'seq': 4096, 'hidden': 4096, 'inter': 11008},
        # Very large batch for inference
        {'batch': 32, 'seq': 128,  'hidden': 4096, 'inter': 11008},
    ]

    results = []
    consistency_results = []

    print(f"Starting benchmark on device: {device} with dtype: {DTYPE}")
    print("=" * 60)

    for i, cfg in enumerate(test_configs):
        B, S, H, I = cfg['batch'], cfg['seq'], cfg['hidden'], cfg['inter']
        config_label = f"B{B}-S{S}-H{H}"
        
        print(f"Running test {i+1}/{len(test_configs)}: {config_label}")

        input_tensor = torch.randn(B, S, H, device=device, dtype=DTYPE)
        
        # 1. PyTorch 版本
        config_pt = ModernBertConfig(hidden_size=H, intermediate_size=I)
        model_pytorch = ModernBertMLP(config_pt).to(device).to(DTYPE)
        pytorch_time = benchmark(model_pytorch, input_tensor)

        # 2. xformers 版本
        xformers_time = float('nan') # 默认为 NaN
        consistency_info = None
        if XFORMERS_AVAILABLE:
            model_xformers = SwiGLU(H, I, H, bias=False).to(device).to(DTYPE)
            xformers_time = benchmark(model_xformers, input_tensor)
            
            # 检查输出一致性
            consistency_info = check_output_consistency(model_pytorch, model_xformers, input_tensor)
            consistency_results.append({
                'Config Label': config_label,
                'L2 Distance': consistency_info['l2_distance']
            })

        # 计算速度提升
        speedup = pytorch_time / xformers_time if XFORMERS_AVAILABLE and xformers_time > 0 else 1.0

        results.append({
            'Config Label': config_label,
            'PyTorch (ms)': pytorch_time,
            'xformers (ms)': xformers_time,
            'Speedup (x)': speedup
        })
        
        # 清理内存
        del model_pytorch, input_tensor
        if XFORMERS_AVAILABLE:
            del model_xformers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 将结果转换为 DataFrame 并打印
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("=" * 60)

    # 打印一致性检查结果
    if XFORMERS_AVAILABLE and consistency_results:
        consistency_df = pd.DataFrame(consistency_results)
        print("\nOutput Consistency Check (L2 Distance)")
        print("=" * 60)
        print(consistency_df.to_string(index=False, float_format="%.6f"))
        print("=" * 60)
        
        # 总结一致性检查
        max_l2_distance = max(result['L2 Distance'] for result in consistency_results)
        print(f"\nConsistency Summary:")
        print(f"Maximum L2 distance across all configurations: {max_l2_distance:.6f}")

    # 可视化结果
    if XFORMERS_AVAILABLE:
        print("\nGenerating performance plot...")
        plot_results(results_df)
        print("Done.")