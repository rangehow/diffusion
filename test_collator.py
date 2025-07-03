import torch
from transformers import AutoTokenizer
from collator import MLMCollator

def test_mlm_collator():
    """测试MLMCollator的主要功能"""
    
    # 1. 初始化tokenizer和collator
    print("=== 初始化tokenizer和collator ===")
    tokenizer = AutoTokenizer.from_pretrained('/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main')
    tokenizer.eos_token = "<|endoftext|>"
    collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_probability=1,  # 15%的token被mask
        mask_probability=0.4,   # 80%替换为[MASK]
        random_probability=0.1, # 10%替换为随机token
        max_length=128
    )
    
    print(f"词汇表大小: {collator.vocab_size}")
    print(f"特殊token: MASK={collator.mask_token_id}, PAD={collator.pad_token_id}")
    print(f"可用随机token数量: {len(collator.valid_token_ids)}")
    
    # 2. 准备测试数据
    print("\n=== 准备测试数据 ===")
    test_texts = [
        "Today is a good day.",
        "This is a smiple test sentence",
        "AI is fundamentally changing the world."
    ]
    
    # 分词处理
    examples = []
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        examples.append({'input_ids': tokens})
        print(f"原始文本: {text}")
        print(f"Token IDs: {tokens}")
        print(f"解码验证: {tokenizer.decode(tokens)}")
        print()
    
    # 3. 测试collator
    print("=== 测试collator处理 ===")
    batch_data = collator(examples)
    
    print("Batch数据结构:")
    for key, value in batch_data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
    # 4. 详细分析每个样本
    print("\n=== 详细分析处理结果 ===")
    for i in range(len(examples)):
        print(f"\n--- 样本 {i+1} ---")
        input_ids = batch_data['input_ids'][i]
        attention_mask = batch_data['attention_mask'][i]
        labels = batch_data['labels'][i]
        token_change_labels = batch_data['token_change_labels'][i]
        
        # 找到有效长度（去掉padding）
        valid_length = attention_mask.sum().item()
        
        print(f"有效长度: {valid_length}")
        print(f"原始文本: {test_texts[i]}")
        
        # 显示处理前后的对比
        original_tokens = examples[i]['input_ids'] + [tokenizer.eos_token_id]
        if len(original_tokens) > collator.max_length:
            original_tokens = original_tokens[:collator.max_length-1] + [tokenizer.eos_token_id]
            
        print("Token对比:")
        
        for j in range(valid_length):
            orig_token = original_tokens[j] if j < len(original_tokens) else tokenizer.pad_token_id
            curr_token = input_ids[j].item()
            label = labels[j].item()
            changed = token_change_labels[j].item()
            
            orig_text = tokenizer.decode([orig_token]) if orig_token != tokenizer.pad_token_id else "[PAD]"
            curr_text = tokenizer.decode([curr_token]) if curr_token != tokenizer.pad_token_id else "[PAD]"
            
            status = ""
            if changed == 1:
                if curr_token == tokenizer.mask_token_id:
                    status = " [MASKED]"
                elif curr_token != orig_token:
                    status = " [RANDOM]"
                else:
                    status = " [UNCHANGED]"
            
            print(f"  位置{j}: {orig_text} -> {curr_text}{status} (label: {label})")
        breakpoint()
    # 5. 验证masking统计
    print("\n=== Masking统计验证 ===")
    for i in range(len(examples)):
        token_change_labels = batch_data['token_change_labels'][i]
        attention_mask = batch_data['attention_mask'][i]
        
        # 计算有效token数量（排除padding和EOS）
        valid_positions = []
        input_ids = batch_data['input_ids'][i]
        for j in range(attention_mask.sum().item()):
            if input_ids[j].item() not in [tokenizer.cls_token_id, tokenizer.sep_token_id, 
                                          tokenizer.pad_token_id, tokenizer.eos_token_id]:
                valid_positions.append(j)
        
        total_valid_tokens = len(valid_positions)
        masked_tokens = token_change_labels.sum().item()
        
        print(f"样本{i+1}: 有效token数={total_valid_tokens}, 被处理token数={masked_tokens}, "
              f"处理比例={masked_tokens/total_valid_tokens:.2%}")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    tokenizer = AutoTokenizer.from_pretrained('/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main')
    tokenizer.eos_token = "<|endoftext|>"
    collator = MLMCollator(tokenizer, max_length=10)
    
    # 测试空序列
    try:
        empty_examples = [{'input_ids': []}]
        result = collator(empty_examples)
        print("空序列测试通过")
    except Exception as e:
        print(f"空序列测试失败: {e}")
    
    # 测试超长序列
    long_text = "This is a very long sentence," * 20
    long_tokens = tokenizer.encode(long_text, add_special_tokens=False)
    long_examples = [{'input_ids': long_tokens}]
    
    result = collator(long_examples)
    print(f"超长序列测试: 原长度={len(long_tokens)}, 处理后长度={result['attention_mask'][0].sum().item()}")
    
    # 测试只有特殊token的情况
    special_examples = [{'input_ids': [tokenizer.cls_token_id, tokenizer.sep_token_id]}]
    result = collator(special_examples)
    print("特殊token序列测试通过")

if __name__ == "__main__":
    test_mlm_collator()
    test_edge_cases()