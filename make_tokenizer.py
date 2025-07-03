from transformers import AutoTokenizer, Qwen2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main", trust_remote_code=True)


tokenizer.eos_token = "<|endoftext|>"
print(tokenizer)
# # 创建Qwen2Tokenizer，使用相同的参数
# qwen2_tokenizer = Qwen2Tokenizer.from_pretrained("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main", trust_remote_code=True)
# qwen2_tokenizer.eos_token_id = 50279

breakpoint()