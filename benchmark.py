# benchmark_aligned.py

import torch
import time
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers import ModernBertConfig, ModernBertForMaskedLM

# --- 1. 常量和配置 ---
BATCH_SIZE = 1
# ModernBert 默认支持到 8192，这是一个很好的长序列测试点
SEQ_LENGTH = 8192 
WARMUP_STEPS = 2
TEST_STEPS = 10 # 增加测试步数以获得更稳定的结果

# --- 2. 检查GPU和通用设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"检测到 {num_gpus} 个GPU")
print(f"主设备: {device}")
print(f"测试参数: Batch Size={BATCH_SIZE}, Sequence Length={SEQ_LENGTH}")
print("-" * 30)

# --- 3. 模型配置（参数量对齐）---

# 目标参数量: ~130M

# 3.1 ModernBERT 配置 (约 130M)
# 这是我们的基准，来自你提供的 config 文档
ModernBERT_130M_config = ModernBertConfig(
    vocab_size=50368,
    hidden_size=768,
    intermediate_size=1152, # ModernBERT 的 MLP 尺寸较小
    num_hidden_layers=22,
    num_attention_heads=12,
    max_position_embeddings=8192,
    torch_dtype="bfloat16",
    reference_compile = True,
    global_attn_every_n_layers = 100,
    local_attention= 1,
    # ModernBERT 可能有自己的优化实现，这里不强制指定 attn_implementation
)

# 3.2 Qwen2 全局注意力配置 (定制以达到 ~130M)
# 为了参数对齐，我们必须减小层数和隐藏层大小
Qwen2_130M_Global_config = Qwen2Config(
    vocab_size=50368,
    hidden_size=768,
    intermediate_size=1152, # Qwen2-0.5B 的比例
    num_key_value_heads=12, # 使用 MHA 以简化对比
    num_hidden_layers=22,    # 显著减少层数
    num_attention_heads=12,
    max_position_embeddings=8192,
    use_sliding_window=False, # 明确禁用SWA，使用全局注意力
    torch_dtype="bfloat16",
    use_cache=False,
    # attn_implementation='flash_attention_2',
)

# 3.3 Qwen2 滑动窗口注意力配置 (与全局版参数完全相同)
Qwen2_130M_SWA_config = Qwen2Config(
    vocab_size=50368,
    hidden_size=768,
    intermediate_size=1152,
    num_key_value_heads=12,
    num_attention_heads=12,
    num_hidden_layers=22,
    max_position_embeddings=8192,
    use_sliding_window=True, # 启用 SWA
    sliding_window=4096,     # 设置窗口大小
    max_window_layers=22,     # 所有层都使用 SWA
    torch_dtype="bfloat16",
    use_cache=False,
    # attn_implementation='flash_attention_2',
)

# --- 4. 模型实例化和字典 ---
models = {
    "ModernBERT_130M": ModernBertForMaskedLM(ModernBERT_130M_config),
    "Qwen2_130M_Global": Qwen2ForCausalLM(Qwen2_130M_Global_config),
    "Qwen2_130M_SWA": Qwen2ForCausalLM(Qwen2_130M_SWA_config),
}

# --- 5. GPU 分配和模型准备 ---
# (这部分代码与之前相同，无需修改)
gpu_assignments = {}
if num_gpus > 0:
    # 如果GPU够用，每个模型分配一个GPU
    if num_gpus >= len(models):
        for i, (name, model) in enumerate(models.items()):
            gpu_id = i
            model_device = f"cuda:{gpu_id}"
            model.to(model_device).to(torch.bfloat16)
            gpu_assignments[name] = gpu_id
            print(f"'{name}' 模型已分配到 GPU {gpu_id} 并转换为 bfloat16")
    else: # 如果GPU不够，循环分配
        for i, (name, model) in enumerate(models.items()):
            gpu_id = i % num_gpus
            model_device = f"cuda:{gpu_id}"
            model.to(model_device).to(torch.bfloat16)
            gpu_assignments[name] = gpu_id
            print(f"'{name}' 模型已分配到 GPU {gpu_id} (循环) 并转换为 bfloat16")
else:
    print("警告: 未检测到GPU，将在CPU上运行")
    for name, model in models.items():
        model.to(device) # CPU 不支持 bfloat16
        gpu_assignments[name] = "cpu"

print("-" * 30)
for name, model in models.items():
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_id = gpu_assignments.get(name, "cpu")
    print(f"'{name}' 模型参数量: {num_params / 1_000_000:.2f} M (设备: {gpu_id})")
print("-" * 30)


# --- 6. 基准测试函数 ---
# (这部分代码与之前相同，无需修改)
def benchmark_forward_pass(model, model_name, device_id):
    """测量模型前向传递的延迟和吞吐量。"""
    model.eval()
    
    vocab_size = model.config.vocab_size
    model_device = device_id if isinstance(device_id, str) and device_id == "cpu" else f"cuda:{device_id}"
    
    print(f"正在为 '{model_name}' (设备: {model_device}) 创建输入... Vocab: {vocab_size}, SeqLen: {SEQ_LENGTH}")
    dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LENGTH), device=model_device, dtype=torch.long)
    
    print(f"正在预热 '{model_name}'...")
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(dummy_input)
    
    if "cuda" in str(model_device):
        torch.cuda.synchronize(device=model_device)

    print(f"正在运行 '{model_name}' 的基准测试...")
    total_time = 0.0
    with torch.no_grad():
        for i in range(TEST_STEPS):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if "cuda" in str(model_device):
                torch.cuda.synchronize(device=model_device)
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
            
    avg_latency = (total_time / TEST_STEPS) * 1000
    throughput = (BATCH_SIZE * SEQ_LENGTH) / (total_time / TEST_STEPS)
    
    return avg_latency, throughput

# --- 7. 运行基准测试 ---
results = {}
for name, model in models.items():
    device_id = gpu_assignments.get(name, "cpu")
    try:
        latency, throughput = benchmark_forward_pass(model, name, device_id)
        results[name] = {"latency_ms": latency, "throughput_tokens_sec": throughput, "device_id": device_id}
    except Exception as e:
        print(f"测试 '{name}' 时出错: {e}")
        results[name] = {"error": str(e), "device_id": device_id}

# --- 8. 打印结果 ---
print("\n" + "=" * 50)
print("基准测试最终结果 (参数量对齐)")
print("=" * 50)
for name, res in sorted(results.items(), key=lambda item: item[1].get('latency_ms', float('inf'))):
    print(f"模型: {name} (设备 {res['device_id']})")
    if "error" in res:
        print(f"  -> 测试失败: {res['error']}")
    else:
        print(f"  -> 平均延迟 (Avg. Latency): {res['latency_ms']:.2f} ms")
        print(f"  -> 吞吐量 (Throughput): {res['throughput_tokens_sec']:,.0f} tokens/sec")
    print("-" * 20)