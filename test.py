import torch
from modeling import ModernBertForDiffusionLM
from transformers import AutoTokenizer
import accelerate
from accelerate import Accelerator

# 初始化accelerator
accelerator = Accelerator()

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if accelerator.is_main_process:
    print(f"Using device: {device}")

model_path = "diffusion/model_output/random_m1_not_all_token/checkpoint-18572"

# 加载模型和tokenizer
if accelerator.is_main_process:
    print("Loading model...")
model = ModernBertForDiffusionLM.from_pretrained(model_path)
model.to(device)  # 移动模型到GPU
model.eval()  # 设置为评估模式

if accelerator.is_main_process:
    print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试输入
test_input = "What's your hobby?"
if accelerator.is_main_process:
    print(f"Input text: {test_input}")

# 手动encode输入，不添加特殊tokens
input_ids = tokenizer.encode(test_input, add_special_tokens=False, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids).to(device)  # 手动创建attention mask

if accelerator.is_main_process:
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

# 获取mask token id
mask_token_id = tokenizer.mask_token_id
if mask_token_id is None:
    # 如果tokenizer没有mask token，使用[MASK]或者vocab中的特殊token
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if mask_token_id == tokenizer.unk_token_id:
        # 如果还是没有，使用pad token作为替代
        mask_token_id = tokenizer.pad_token_id

if accelerator.is_main_process:
    print(f"Mask token ID: {mask_token_id}")

# 生成文本
if accelerator.is_main_process:
    print("\nGenerating text...")
try:
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            max_new_tokens=100,  
            num_diffusion_steps=100, 
            temperature_mlm=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            debug=accelerator.is_main_process,  # 只在主进程debug
            tokenizer=tokenizer,
            use_token_change_classifier=False,
        )
    
    # 解码生成的文本
    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    # print(f"Generated text: {generated_text}")
    
    # 显示新生成的部分
    original_length = input_ids.shape[1]
    new_tokens = generated_ids[0][original_length:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if accelerator.is_main_process:
        print(f"New generated part: {new_text}")
    
except Exception as e:
    if accelerator.is_main_process:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()