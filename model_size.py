from transformers import AutoConfig,LlamaForCausalLM
# from .modeling import ModernBertForDiffusionLM
# from .modeling_llada import LLaDAModelLM


# config = AutoConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json')
# model = ModernBertForDiffusionLM(config)

# config = AutoConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llada_large.json')
# model = LLaDAModelLM(config)

config = AutoConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_400M.json')
model = LlamaForCausalLM(config)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

# 格式化输出
def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

print(f"总参数量: {format_params(total_params)} ({total_params:,})")
print(f"可训练参数量: {format_params(trainable_params)} ({trainable_params:,})")
print(f"不可训练参数量: {format_params(non_trainable_params)} ({non_trainable_params:,})")