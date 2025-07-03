import torch

from .llada_generate import generate
from transformers import AutoTokenizer
from .modeling_llada import LLaDAModelLM

def test_base_model():
    """
    使用 'prompt + mask' 的方式测试基座模型，不使用聊天模板。
    """
    device = 'cuda'
    # 模型和分词器加载（与原代码相同）
    model_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/llada_500m/checkpoint-18572'
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # --- 关键参数 ---
    # 需要生成的长度
    gen_length = 128
    # LLaDA 生成过程的步数
    steps = 128

    print('*' * 66)
    print(f'**  模式: Base Model Test (Prompt + Mask)  **')
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    # 验证特殊 token 是否存在
    # 对于 GLM 类型的模型，通常使用 gmask_token_id 和 bos_token_id
    # 如果您的分词器中没有这些，可能需要换成 mask_token_id 和 cls_token_id 等
    # if not hasattr(tokenizer, 'gmask_token_id'):
    #     raise ValueError("Tokenizer a'gmask_token_id' 属性, 请检查分词器配置。可能需要使用 'mask_token_id'。")
    # if not hasattr(tokenizer, 'bos_token_id'):
    #     raise ValueError("Tokenizer a'bos_token_id' 属性, 请检查分词器配置。可能需要使用 'cls_token_id'。")

    mask_token_id = tokenizer.mask_token_id
    bos_token_id = tokenizer.bos_token_id

    while True:
        question = input("Enter your question: ")
        if question.strip().lower() == 'exit':
            break

        # 1. 【核心修改】手动构建输入
        # 不再使用 apply_chat_template
        # 格式为：[BOS] + 问题tokens + [gMASK] + [MASK] * gen_length
        
        # 将问题文本编码成 token ids
        question_ids = tokenizer.encode(question, add_special_tokens=False)

        # 构建完整的 input_ids
        # 注意：这里我们添加了一个 gMASK token 来触发生成，然后填充普通 MASK token 作为生成占位符
        # 有些实现可能只需要 [BOS] + 问题 + [MASK]*N，这取决于模型的具体训练方式。
        # GLM/LLaDA 常用 [BOS] + 问题 + [gMASK] + [MASK]*N 的形式。
        # 我们假设生成长度 gen_length 已经包含了 gMASK。
        mask_tokens = [mask_token_id] + [tokenizer.mask_token_id] * (gen_length -1)
        input_ids = question_ids + mask_tokens
        
        # 转换为 PyTorch Tensor
        prompt = torch.tensor([input_ids]).to(device)

        # 2. 【核心修改】调用生成函数
        # 这里的 prompt 就是我们手动构建好的张量，不再需要处理多轮对话历史
        out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence',mask_id=mask_token_id)

        # 3. 【核心修改】解码输出
        # 我们需要从输出中解码生成的部分
        # 生成的部分从问题 token 之后开始
        prompt_part_len = len(question_ids) + 1  # +1 是因为开头的 bos_token_id
        
        # 从完整输出 out 中，切片出生成内容的部分
        # 注意：out 的形状通常是 (1, total_length)，包含了 prompt 和生成内容
        generated_tokens = out[:, prompt_part_len : prompt_part_len + gen_length]
        
        answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Bot's reply: {answer}")
        print('-----------------------------------------------------------------------')

# 原有的 chat 函数，可以保留用于对比
def chat():
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/model_output/llada_500m/checkpoint-20000', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/model_output/llada_500m/checkpoint-20000', trust_remote_code=True)

    gen_length = 128
    steps = 128
    print('*' * 66)
    print(f'**  模式: Chat Template  **')
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    # 选择要运行的函数
    test_base_model()
    # chat()