import torch
import time
import json
from transformers import ModernBertForMaskedLM, ModernBertConfig

# 确保你的自定义 LLaDA 文件在 Python 路径中
# 如果文件在当前目录，则无需任何操作
try:
    from modeling_llada import LLaDAModelLM
    from configuration_llada import LLaDAConfig
except ImportError as e:
    print("错误: 无法导入 LLaDA 模型。请确保 'modeling_llada.py' 和 'configuration_llada.py' 文件在当前工作目录中。")
    print(f"原始错误: {e}")
    exit()

# --- 评测配置 ---
BATCH_SIZE = 8
SEQ_LENGTH = 256
NUM_WARMUP_RUNS = 10  # 预热运行次数
NUM_BENCHMARK_RUNS = 50 # 正式评测运行次数

# --- 模型配置文件路径 ---
LLADA_CONFIG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llada_large.json"
MODERNBERT_CONFIG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json"

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
print("-" * 30)

def benchmark_forward_pass(model, model_name, dummy_input, num_warmup, num_runs):
    """
    一个通用的函数，用于评测模型的前向推理速度。

    Args:
        model (torch.nn.Module): 要评测的模型.
        model_name (str): 模型的名称，用于打印.
        dummy_input (dict): 喂给模型的输入数据.
        num_warmup (int): 预热运行的次数.
        num_runs (int): 正式评测的次数.
    """
    print(f"开始评测模型: {model_name}")
    
    # 将模型和数据移动到指定设备
    model.to(device)
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
    
    # 设置为评估模式并禁用梯度
    model.eval()
    with torch.no_grad():
        # 1. 预热 (Warm-up)
        print(f"正在进行 {num_warmup} 次预热...")
        for _ in range(num_warmup):
            _ = model(**dummy_input)
        
        # 确保预热操作完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("预热完成。")

        # 2. 正式评测
        timings = []
        print(f"正在进行 {num_runs} 次正式评测...")
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize() # 等待之前的GPU操作完成
            
            start_time = time.perf_counter()
            _ = model(**dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() # 确保本次前向推理完成
            
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
        
        # 3. 计算并打印结果
        avg_time_s = sum(timings) / len(timings)
        avg_time_ms = avg_time_s * 1000
        # 计算吞吐量 (Throughput), 即每秒可以处理多少个样本
        throughput = BATCH_SIZE / avg_time_s
        
        print(f"--- {model_name} 评测结果 ---")
        print(f"批处理大小 (Batch Size): {BATCH_SIZE}")
        print(f"序列长度 (Sequence Length): {SEQ_LENGTH}")
        print(f"平均前向推理时间: {avg_time_ms:.3f} ms")
        print(f"吞吐量 (Throughput): {throughput:.2f} samples/sec")
        print("-" * 30)


# --- 1. 加载和评测 LLaDA ---
try:
    print("正在加载 LLaDA 模型...")
    llada_config = LLaDAConfig.from_pretrained(LLADA_CONFIG_PATH)
    llada_model = LLaDAModelLM(llada_config)
    print("LLaDA 模型加载成功。")

    # 创建虚拟输入数据
    # 假设 LLaDA 也接受 input_ids 和 attention_mask
    dummy_input_ids = torch.randint(0, llada_config.vocab_size, (BATCH_SIZE, SEQ_LENGTH))
    dummy_attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH)
    llada_dummy_input = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
    }

    # 运行评测
    benchmark_forward_pass(llada_model, "LLaDA Large", llada_dummy_input, NUM_WARMUP_RUNS, NUM_BENCHMARK_RUNS)

except Exception as e:
    print(f"加载或评测 LLaDA 模型时出错: {e}")
    print("-" * 30)


# --- 2. 加载和评测 ModernBERT ---
try:
    print("正在加载 ModernBERT 模型...")
    # ModernBERT 需要从 config 初始化，而不是 from_pretrained
    modernbert_config = ModernBertConfig.from_json_file(MODERNBERT_CONFIG_PATH)
    modernbert_model = ModernBertForMaskedLM(modernbert_config)
    print("ModernBERT 模型加载成功。")

    # 创建虚拟输入数据
    dummy_input_ids = torch.randint(0, modernbert_config.vocab_size, (BATCH_SIZE, SEQ_LENGTH))
    dummy_attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH)
    modernbert_dummy_input = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
    }

    # 运行评测
    benchmark_forward_pass(modernbert_model, "ModernBERT Large", modernbert_dummy_input, NUM_WARMUP_RUNS, NUM_BENCHMARK_RUNS)

except Exception as e:
    print(f"加载或评测 ModernBERT 模型时出错: {e}")
    print("-" * 30)