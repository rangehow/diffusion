#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩散语言模型 vs LLaMA 模型生成对比测试脚本
测试后继生成效果和吞吐速度
"""

import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    except:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    model = model.to(device).eval()
    return model, tokenizer


def test_diffusion_generation(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    block_size: int = 8,
    steps: int = 8,
    temperature: float = 1.0,
    confidence_threshold: float = 0.9,
    device: str = "cuda",
    num_warmup: int = 0,
    num_runs: int = 1,
):
    """测试扩散模型生成效果和吞吐速度"""

    bos_token_id = tokenizer.bos_token_id
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        if mask_token_id == tokenizer.unk_token_id:
            raise ValueError("无法找到 MASK token")

    print("=" * 80)
    print("🌀 扩散模型测试配置")
    print("=" * 80)
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  block_size: {block_size}")
    print(f"  steps: {steps}")
    print(f"  temperature: {temperature}")
    print(f"  confidence_threshold: {confidence_threshold}")
    print(f"  bos_token_id: {bos_token_id}")
    print(f"  mask_token_id: {mask_token_id}")
    print()

    # === 生成效果测试 ===
    print("=" * 80)
    print("🔤 扩散模型生成效果测试")
    print("=" * 80)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}] Prompt: {prompt}")
        print("-" * 40)

        # 扩散模型：开头加 BOS
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if bos_token_id is not None:
            tokens = [bos_token_id] + tokens

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                steps=steps,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                debug=False,
                use_gumbel=True
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_text = tokenizer.decode(output_ids[0, len(tokens):], skip_special_tokens=True)

        print(f"生成续写: {new_text}")
        results.append({"prompt": prompt, "generated": new_text, "full": generated_text})

    # === 吞吐速度测试 ===
    print("\n" + "=" * 80)
    print("⚡ 扩散模型吞吐速度测试")
    print("=" * 80)

    test_prompt = prompts[0]
    tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    if bos_token_id is not None:
        tokens = [bos_token_id] + tokens
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Warmup
    print(f"\n🔥 Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                steps=steps,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                debug=False,
            )

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"⏱️  Benchmark ({num_runs} runs)...")

    latencies = []
    for run in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                steps=steps,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                debug=False,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        num_generated = output_ids.shape[1] - len(tokens)
        print(f"  Run {run + 1}: {latency:.3f}s, {num_generated} tokens, {num_generated / latency:.1f} tokens/s")

    avg_latency = sum(latencies) / len(latencies)
    throughput = max_new_tokens / avg_latency

    print(f"\n📊 扩散模型统计结果:")
    print(f"  平均延迟: {avg_latency:.3f}s")
    print(f"  平均吞吐: {throughput:.1f} tokens/s")

    # Batch 测试
    batch_stats = {}
    print(f"\n📦 Batch 吞吐测试...")
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        batch_input_ids = input_ids.repeat(bs, 1)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=batch_input_ids,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                steps=steps,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                debug=False,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start_time
        total_tokens = bs * max_new_tokens
        bs_throughput = total_tokens / latency
        batch_stats[bs] = bs_throughput

        print(f"  Batch {bs}: {latency:.3f}s, {bs_throughput:.1f} tokens/s")

    return results, {"avg_latency": avg_latency, "throughput": throughput, "batch_stats": batch_stats}


def test_llama_generation(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
    num_warmup: int = 0,
    num_runs: int = 1,
    add_bos: bool = True,  # 新增参数：是否加BOS
):
    """测试 LLaMA 模型生成效果和吞吐速度"""

    bos_token_id = tokenizer.bos_token_id

    print("=" * 80)
    print("🦙 LLaMA 模型测试配置")
    print("=" * 80)
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  do_sample: True")
    print(f"  top_p: 1.0 (不截断)")
    print(f"  top_k: 0 (不截断)")
    print(f"  add_bos: {add_bos}")  # 显示是否加BOS
    print(f"  bos_token_id: {bos_token_id}")
    print()

    # 设置 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # === 生成效果测试 ===
    print("=" * 80)
    print("🔤 LLaMA 模型生成效果测试")
    print("=" * 80)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}] Prompt: {prompt}")
        print("-" * 40)

        # 编码
        tokens = tokenizer.encode(prompt, add_special_tokens=False)

        # 根据参数决定是否加BOS
        if add_bos and bos_token_id is not None:
            tokens = [bos_token_id] + tokens

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

        print(f"生成续写: {new_text}")
        results.append({"prompt": prompt, "generated": new_text, "full": generated_text})

    # === 吞吐速度测试 ===
    print("\n" + "=" * 80)
    print("⚡ LLaMA 模型吞吐速度测试")
    print("=" * 80)

    test_prompt = prompts[0]
    tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    if add_bos and bos_token_id is not None:
        tokens = [bos_token_id] + tokens
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    input_len = input_ids.shape[1]

    # Warmup
    print(f"\n🔥 Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"⏱️  Benchmark ({num_runs} runs)...")

    latencies = []
    for run in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        num_generated = output_ids.shape[1] - input_len
        print(f"  Run {run + 1}: {latency:.3f}s, {num_generated} tokens, {num_generated / latency:.1f} tokens/s")

    avg_latency = sum(latencies) / len(latencies)
    throughput = max_new_tokens / avg_latency

    print(f"\n📊 LLaMA 模型统计结果:")
    print(f"  平均延迟: {avg_latency:.3f}s")
    print(f"  平均吞吐: {throughput:.1f} tokens/s")

    # Batch 测试
    batch_stats = {}
    print(f"\n📦 Batch 吞吐测试...")
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        batch_input_ids = input_ids.repeat(bs, 1)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=batch_input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start_time
        total_tokens = bs * max_new_tokens
        bs_throughput = total_tokens / latency
        batch_stats[bs] = bs_throughput

        print(f"  Batch {bs}: {latency:.3f}s, {bs_throughput:.1f} tokens/s")

    return results, {"avg_latency": avg_latency, "throughput": throughput, "batch_stats": batch_stats}


def compare_results(diffusion_results, llama_results, diffusion_stats, llama_stats):
    """对比两个模型的结果"""
    print("\n" + "=" * 80)
    print("📊 模型对比总结")
    print("=" * 80)

    print("\n【生成效果对比】")
    print("-" * 80)
    for i, (d_res, l_res) in enumerate(zip(diffusion_results, llama_results)):
        print(f"\n[{i + 1}] Prompt: {d_res['prompt']}")
        d_text = d_res['generated'][:80] + "..." if len(d_res['generated']) > 80 else d_res['generated']
        l_text = l_res['generated'][:80] + "..." if len(l_res['generated']) > 80 else l_res['generated']
        print(f"  🌀 扩散: {d_text}")
        print(f"  🦙 LLaMA: {l_text}")

    print("\n" + "-" * 80)
    print("【吞吐性能对比】")
    print("-" * 80)

    d_thr = diffusion_stats['throughput']
    l_thr = llama_stats['throughput']
    speedup = d_thr / l_thr if l_thr > 0 else float('inf')

    print(f"\n  单样本测试:")
    print(f"    🌀 扩散模型:  延迟 {diffusion_stats['avg_latency']:.3f}s | 吞吐 {d_thr:.1f} tokens/s")
    print(f"    🦙 LLaMA模型: 延迟 {llama_stats['avg_latency']:.3f}s | 吞吐 {l_thr:.1f} tokens/s")
    print(f"    📈 扩散/LLaMA 速度比: {speedup:.2f}x")

    print(f"\n  Batch 吞吐对比 (tokens/s):")
    print(f"    {'Batch':<8} {'扩散模型':<15} {'LLaMA模型':<15} {'速度比':<10}")
    print(f"    {'-'*48}")

    for bs in diffusion_stats['batch_stats']:
        d_bs = diffusion_stats['batch_stats'][bs]
        l_bs = llama_stats['batch_stats'].get(bs, 0)
        ratio = d_bs / l_bs if l_bs > 0 else float('inf')
        print(f"    {bs:<8} {d_bs:<15.1f} {l_bs:<15.1f} {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="扩散语言模型 vs LLaMA 对比测试")
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="扩散模型路径")
    parser.add_argument("--llama_model_path", type=str, default=None, help="LLaMA模型路径（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="最大生成 token 数")
    parser.add_argument("--block_size", type=int, default=1, help="块大小（扩散模型专用）")
    parser.add_argument("--steps", type=int, default=8, help="每块去噪步数（扩散模型专用）")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度（两模型共用）")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="置信度阈值（扩散模型专用）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_warmup", type=int, default=1, help="预热次数")
    parser.add_argument("--num_runs", type=int, default=1, help="测试次数")

    args = parser.parse_args()

    test_prompts = [
        "What's your hobby?",
        "I love reading books.",
        "The weather is",
        "My favorite food is",
        "How do you",
        "Yesterday I went to",
        "The most important thing in life is",
        "Technology has changed",
        "When I was young",
        "In the future"
    ]

    # ========== 测试扩散模型 ==========
    print("🚀 加载扩散模型...")
    diffusion_model, diffusion_tokenizer = load_model_and_tokenizer(args.diffusion_model_path, args.device)
    print(f"✅ 扩散模型加载完成: {type(diffusion_model).__name__}")
    print(f"   Vocab size: {diffusion_tokenizer.vocab_size}")

    diffusion_results, diffusion_stats = test_diffusion_generation(
        model=diffusion_model,
        tokenizer=diffusion_tokenizer,
        prompts=test_prompts,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        steps=args.steps,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    # 释放扩散模型显存（如果需要）
    if args.llama_model_path:
        del diffusion_model
        torch.cuda.empty_cache()

    # ========== 测试 LLaMA 模型 ==========
    if args.llama_model_path:
        print("\n" + "=" * 80)
        print("🚀 加载 LLaMA 模型...")
        llama_model, llama_tokenizer = load_model_and_tokenizer(args.llama_model_path, args.device)
        print(f"✅ LLaMA 模型加载完成: {type(llama_model).__name__}")
        print(f"   Vocab size: {llama_tokenizer.vocab_size}")

        llama_results, llama_stats = test_llama_generation(
            model=llama_model,
            tokenizer=llama_tokenizer,
            prompts=test_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )

        # 对比结果
        compare_results(diffusion_results, llama_results, diffusion_stats, llama_stats)

    print("\n" + "=" * 80)
    print("✅ 测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()