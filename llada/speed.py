import torch
import torch.nn.functional as F
import time
import random

# --- 1. 配置参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 32
MAX_SEQ_LEN = 1024
MIN_SEQ_LEN = 100
EMBED_DIM = 768
NUM_HEADS = 12
HEAD_DIM = EMBED_DIM // NUM_HEADS
assert EMBED_DIM % NUM_HEADS == 0, "EMBED_DIM must be divisible by NUM_HEADS"

WARMUP_RUNS = 5
TIMED_RUNS = 20

# --- 2. 构造测试数据 ---
def setup_data(batch_size, min_len, max_len, embed_dim, device):
    sequences = []
    lengths = []
    for _ in range(batch_size):
        seq_len = random.randint(min_len, max_len)
        seq = torch.randn(1, seq_len, embed_dim, device=device)
        sequences.append(seq)
        lengths.append(seq_len)
    return sequences, lengths

# --- 3. 方法一：批处理与填充 (Batched with Padding) - 掩码逻辑修正 ---
def run_batched_padded(sequences, lengths):
    max_len = max(lengths)
    
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        [s.squeeze(0) for s in sequences], 
        batch_first=True, 
        padding_value=0.0
    )
    
    # !!!!!!!!!!!!!!!!!!
    # --- 修正之处 ---
    # 根据官方文档，布尔掩码中 True 代表 "参与计算"
    # 所以有效token的位置应该是True，padding的位置是False
    attention_mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    # !!!!!!!!!!!!!!!!!!
    
    attention_mask = attention_mask[:, None, None, :]
    
    b, s, e = padded_sequences.shape
    qkv = padded_sequences.view(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    start_time = time.perf_counter()
    attn_output = F.scaled_dot_product_attention(
        qkv, qkv, qkv,
        attn_mask=attention_mask,
        is_causal=False
    )
    duration = time.perf_counter() - start_time
    
    output_padded = attn_output.transpose(1, 2).contiguous().view(b, s, e)

    return output_padded, duration

# --- 4. 方法二：序列打包 (Packing) - 掩码逻辑修正 ---
def run_packed(sequences, lengths):
    packed_sequence = torch.cat(sequences, dim=1)
    total_seq_len = packed_sequence.shape[1]
    
    # !!!!!!!!!!!!!!!!!!
    # --- 修正之处 ---
    # 同样，True 代表 "参与计算"。我们从一个全False的矩阵开始。
    packed_mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool, device=device)
    
    current_pos = 0
    for length in lengths:
        # 将允许相互attend的对角块区域设置为 True
        packed_mask[current_pos:current_pos+length, current_pos:current_pos+length] = True
        current_pos += length
    # !!!!!!!!!!!!!!!!!!
        
    b, s, e = packed_sequence.shape
    qkv = packed_sequence.view(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    start_time = time.perf_counter()
    attn_output = F.scaled_dot_product_attention(
        qkv, qkv, qkv,
        attn_mask=packed_mask,
        is_causal=False
    )
    duration = time.perf_counter() - start_time
    
    output_packed = attn_output.transpose(1, 2).contiguous().view(b, s, e)
    
    return output_packed, duration

# --- 5. 主执行逻辑 (不变) ---
if __name__ == "__main__":
    sequences, lengths = setup_data(BATCH_SIZE, MIN_SEQ_LEN, MAX_SEQ_LEN, EMBED_DIM, device)
    
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        def measure_time(func, *args):
            # 将返回值解包，防止可能的异步问题
            output, duration = func(*args)
            starter.record()
            output, duration = func(*args)
            ender.record()
            torch.cuda.synchronize()
            return starter.elapsed_time(ender) / 1000.0
    else:
        def measure_time(func, *args):
            start = time.perf_counter()
            _, _ = func(*args)
            return time.perf_counter() - start

    print("--- Running Warmup ---")
    for _ in range(WARMUP_RUNS):
        _, _ = run_batched_padded(sequences, lengths)
        _, _ = run_packed(sequences, lengths)
    if device.type == 'cuda': torch.cuda.synchronize()
    print("Warmup complete.\n")
    
    print("--- Testing Batched Padded Method ---")
    padded_times = []
    output_padded, _ = run_batched_padded(sequences, lengths)
    for _ in range(TIMED_RUNS):
        padded_times.append(measure_time(run_batched_padded, sequences, lengths))
    avg_padded_time = sum(padded_times) / len(padded_times)
    print(f"Average forward time: {avg_padded_time * 1000:.4f} ms")
    
    print("\n--- Testing Packed Method ---")
    packed_times = []
    output_packed, _ = run_packed(sequences, lengths)
    for _ in range(TIMED_RUNS):
        packed_times.append(measure_time(run_packed, sequences, lengths))
    avg_packed_time = sum(packed_times) / len(packed_times)
    print(f"Average forward time: {avg_packed_time * 1000:.4f} ms")
    
    print("\n--- Verifying Accuracy ---")
    unpadded_outputs = [output_padded[i, :length, :] for i, length in enumerate(lengths)]
    unpacked_outputs = []
    current_pos = 0
    for length in lengths:
        unpacked_outputs.append(output_packed[0, current_pos:current_pos+length, :])
        current_pos += length
    
    all_correct = True
    max_diff = 0.0
    # 降低一点精度要求，因为底层实现不同可能导致微小差异
    for i in range(BATCH_SIZE):
        is_close = torch.allclose(unpadded_outputs[i], unpacked_outputs[i], atol=1e-5, rtol=1e-4) 
        if not is_close:
            all_correct = False
            diff = torch.max(torch.abs(unpadded_outputs[i] - unpacked_outputs[i]))
            if diff > max_diff: max_diff = diff
    
    if all_correct:
        print("✅ Accuracy check passed! Outputs are numerically identical.")
    else:
        print(f"❌ Accuracy check failed! Maximum difference: {max_diff.item()}")

    print("\n--- Conclusion ---")
    speedup_factor = avg_padded_time / avg_packed_time
    print(f"Padding method average time: {avg_padded_time * 1000:.4f} ms")
    print(f"Packing method average time: {avg_packed_time * 1000:.4f} ms")
    if speedup_factor > 1:
        print(f"Packing is {speedup_factor:.2f}x faster than Padding for this configuration.")
    else:
        print(f"Padding is {1/speedup_factor:.2f}x faster than Packing for this configuration.")

    total_tokens = sum(lengths)
    padded_tokens = BATCH_SIZE * max(lengths)
    waste_ratio = (padded_tokens - total_tokens) / padded_tokens
    print(f"\nAnalysis:")
    print(f"  Total actual tokens: {total_tokens}")
    print(f"  Total tokens after padding: {padded_tokens}")
    print(f"  Computational waste ratio in padding method: {waste_ratio:.2%}")