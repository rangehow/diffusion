import torch
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "model_output/prefixlm_main_exp_m17/checkpoint-38668"

from transformers import AutoModel, AutoTokenizer

print(f"Testing model: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test with bfloat16
print("\n=== Testing bfloat16 ===")
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.cuda().eval()

text = "Hello world this is a test of the language model to see if it produces NaN"
tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

mask_token_id = tokenizer.mask_token_id
print(f"mask_token_id: {mask_token_id}")
print(f"input_ids shape: {input_ids.shape}")

# Test 1: no masking
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"No masking - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")

# Test 2: mask last 3 tokens
input_ids_masked = input_ids.clone()
input_ids_masked[0, -3:] = mask_token_id
with torch.no_grad():
    out = model(input_ids=input_ids_masked, attention_mask=attention_mask)
    print(f"Masked last 3 - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")

# Test 3: longer sequence (simulate eval)
long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
tokens_long = tokenizer(long_text, return_tensors="pt", max_length=2048, truncation=True)
input_ids_long = tokens_long["input_ids"].cuda()
attention_mask_long = tokens_long["attention_mask"].cuda()
input_ids_long_masked = input_ids_long.clone()
input_ids_long_masked[0, -10:] = mask_token_id
print(f"\nLong sequence length: {input_ids_long.shape[1]}")
with torch.no_grad():
    out = model(input_ids=input_ids_long_masked, attention_mask=attention_mask_long)
    print(f"Long masked - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")

# Test 4: batch of multiple sequences (like MC eval)
batch_ids = input_ids_masked.repeat(16, 1)
batch_mask = attention_mask.repeat(16, 1)
with torch.no_grad():
    out = model(input_ids=batch_ids, attention_mask=batch_mask)
    print(f"Batch 16 - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")

# Test 5: padded batch (different lengths)
text1 = "Short text"
text2 = "This is a longer text that has more tokens in it for padding test"
batch = tokenizer([text1, text2], return_tensors="pt", padding=True)
batch_ids = batch["input_ids"].cuda()
batch_mask = batch["attention_mask"].cuda()
# Mask some tokens
batch_ids[0, -2:] = mask_token_id
batch_ids[1, -5:] = mask_token_id
with torch.no_grad():
    out = model(input_ids=batch_ids, attention_mask=batch_mask)
    print(f"Padded batch - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")
    if out.logits.isnan().any():
        for i in range(batch_ids.shape[0]):
            nan_in_row = out.logits[i].isnan().any(dim=-1)
            print(f"  Row {i}: nan positions = {nan_in_row.nonzero().flatten().tolist()[:10]}")
            print(f"  Row {i} input: {batch_ids[i].tolist()[:20]}...")
            print(f"  Row {i} attn_mask: {batch_mask[i].tolist()[:20]}...")

# Test 6: try float32
print("\n=== Testing float32 ===")
del model
torch.cuda.empty_cache()
model_fp32 = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
model_fp32 = model_fp32.cuda().eval()
with torch.no_grad():
    out = model_fp32(input_ids=batch_ids, attention_mask=batch_mask)
    print(f"Padded batch fp32 - logits nan: {out.logits.isnan().any()}, inf: {out.logits.isinf().any()}")

print("\nDone!")
