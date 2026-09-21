from transformers import AutoConfig,LlamaForCausalLM
from ..modeling.modeling_niu import ModernBertForDiffusionLM
from ..modeling.configuration_niu import NiuConfig
from ..llada.modeling_llada import LLaDAModelLM
from ..bd3lm.modeling_bd3lm import BD3LM
from ..bd3lm.configuration_bd3lm import BD3LMConfig

# 格式化输出函数
def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def calculate_model_params(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{model_name} 参数统计:")
    print(f"总参数量: {format_params(total_params)} ({total_params:,})")
    print(f"可训练参数量: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"不可训练参数量: {format_params(non_trainable_params)} ({non_trainable_params:,})")

# 加载第一个模型
# config = NiuConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_2b.json')
# model = ModernBertForDiffusionLM(config)

# 加载第二个模型
# config1 = AutoConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llada_110m.json')
# model1 = LLaDAModelLM(config1)

# config2 = AutoConfig.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/llama_110m.json')
# model2 = LlamaForCausalLM(config2)

config3 = BD3LMConfig.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/bd3lm_220m.json")
model3 = BD3LM(config3)

# print(model1)
print(model3)
# 计算并输出两个模型的参数量
# calculate_model_params(model, "ModernBertForDiffusionLM")
# calculate_model_params(model1, "LLaDAModelLM")
# calculate_model_params(model2, "LlamaForCausalLM")
calculate_model_params(model3, "BD3LM")
breakpoint()