import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel

def check_bos_eos_training(model_path, tokenizer_path=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,trust_remote_code=True)
        
    except:
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,trust_remote_code=True)
    model.eval().cuda()

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"BOS: '{tokenizer.bos_token}' (ID: {bos_id})")
    print(f"EOS: '{tokenizer.eos_token}' (ID: {eos_id})")
    print("-" * 50)

    with torch.no_grad():
        # === BOS Check ===
        bos_input = torch.tensor([[bos_id]]).cuda()
        logits_after_bos = model(bos_input).logits[0, 0].float()  # convert to float32
        probs_after_bos = torch.softmax(logits_after_bos, dim=-1)
        top_prob_after_bos = probs_after_bos.max().item()
        # Fix nan issue
        probs_clamped = probs_after_bos.clamp(min=1e-10)
        entropy_after_bos = -(probs_clamped * torch.log(probs_clamped)).sum().item()

        print(f"[BOS Test] Top prob after BOS: {top_prob_after_bos:.4f}")
        print(f"[BOS Test] Entropy after BOS: {entropy_after_bos:.2f}")

        # === EOS Check ===
        complete_sentence = "This is a complete sentence."
        tokens = tokenizer(complete_sentence, add_special_tokens=False, return_tensors="pt")
        input_ids = tokens.input_ids.cuda()  # Fix: move input_ids to cuda, not the whole dict
        logits = model(input_ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        eos_prob = probs[eos_id].item()

        print(f"[EOS Test] P(EOS) after complete sentence: {eos_prob:.4f}")

# Your exact model paths
model_paths = [
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335",
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp",
    
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp/checkpoint-77335",
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_main_exp/checkpoint-77335",
]

for path in model_paths:
    print(f"\n{'='*60}")
    print(f"Model: {path}")
    check_bos_eos_training(path)