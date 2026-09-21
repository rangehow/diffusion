"""
Debug script to trace MDLM inference step by step.
Run this on a GPU node to understand why MDLM fails on Countdown.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_tasks.char_tokenizer import ReasoningCharTokenizer
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig

def main():
    tokenizer = ReasoningCharTokenizer()
    mask_id = tokenizer.mask_token_id  # 3
    eos_id = tokenizer.eos_token_id    # 2
    pad_id = tokenizer.pad_token_id    # 0
    bos_id = tokenizer.bos_token_id    # 1
    
    print(f"Special tokens: PAD={pad_id}, BOS={bos_id}, EOS={eos_id}, MASK={mask_id}")
    
    model_path = 'model_output/reasoning_cd3_reasoning_mdlm_110m/checkpoint-15000'
    config = LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path, config=config, dtype=torch.bfloat16)
    model = model.cuda()
    model.eval()
    
    # Test: 17,21,22,26 → 21-17=4,22+4=26
    prompt = '17,21,22,26'
    target = '21-17=4,22+4=26'
    prompt_ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    
    print(f"\nPrompt: {prompt}")
    print(f"Target: {target}")
    print(f"Prompt IDs: {prompt_ids} (len={len(prompt_ids)})")
    print(f"Target IDs: {target_ids} (len={len(target_ids)})")
    
    # ===== TEST 1: Single forward pass with ALL masks =====
    print("\n" + "="*70)
    print("TEST 1: Single forward pass — all response positions masked")
    print("="*70)
    
    max_new = 24
    gen_len = max_new + 1  # +1 for EOS
    plen = len(prompt_ids)
    total_len = plen + gen_len
    
    input_ids = torch.full((1, total_len), pad_id, dtype=torch.long, device='cuda')
    input_ids[0, :plen] = torch.tensor(prompt_ids)
    input_ids[0, plen:plen+gen_len] = mask_id
    attn_mask = torch.zeros(1, total_len, dtype=torch.long, device='cuda')
    attn_mask[0, :plen+gen_len] = 1
    
    print(f"Input: {tokenizer.decode(input_ids[0].tolist())}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        logits[..., mask_id] = float('-inf')
        logits[..., pad_id] = float('-inf')
        
        preds = logits[0].argmax(dim=-1)
        
        print("\nPos-by-pos predictions (mask positions only):")
        for pos in range(plen, plen + gen_len):
            pred_tok = preds[pos].item()
            pred_char = tokenizer.decode([pred_tok])
            
            # Show what target token should be at this position
            target_pos = pos - plen
            if target_pos < len(target_ids):
                expected = tokenizer.decode([target_ids[target_pos]])
                match = "✓" if pred_tok == target_ids[target_pos] else "✗"
            elif target_pos == len(target_ids):
                expected = "[EOS]"
                match = "✓" if pred_tok == eos_id else "✗"
            else:
                expected = "[beyond]"
                match = ""
            
            # Top-3
            probs = torch.softmax(logits[0, pos], dim=-1)
            top3_vals, top3_idx = probs.topk(3)
            top3 = [(tokenizer.decode([idx.item()]), f'{val.item():.3f}') for val, idx in zip(top3_vals, top3_idx)]
            
            print(f"  pos={pos:2d}: pred={pred_char!r:5s} expected={expected!r:5s} {match} top3={top3}")
    
    pred_text_test1 = tokenizer.decode(preds[plen:plen+gen_len].tolist())
    print(f"\nFull prediction (single pass): {pred_text_test1!r}")
    print(f"Target:                        {target!r}")
    
    # ===== TEST 2: Give the model the CORRECT input (prompt + target) and see what it predicts =====
    print("\n" + "="*70)
    print("TEST 2: Forward pass with CORRECT response (no masking) — sanity check")
    print("="*70)
    
    correct_ids = prompt_ids + target_ids + [eos_id]
    correct_len = len(correct_ids)
    # Pad to total_len
    input_ids2 = torch.full((1, total_len), pad_id, dtype=torch.long, device='cuda')
    input_ids2[0, :correct_len] = torch.tensor(correct_ids)
    attn_mask2 = torch.zeros(1, total_len, dtype=torch.long, device='cuda')
    attn_mask2[0, :correct_len] = 1
    
    with torch.no_grad():
        outputs2 = model(input_ids=input_ids2, attention_mask=attn_mask2)
        logits2 = outputs2.logits
        preds2 = logits2[0].argmax(dim=-1)
        
        print("Does model reproduce input when given clean input?")
        for pos in range(correct_len):
            inp = input_ids2[0, pos].item()
            pred = preds2[pos].item()
            inp_char = tokenizer.decode([inp])
            pred_char = tokenizer.decode([pred])
            match = "✓" if inp == pred else "✗"
            
            probs = torch.softmax(logits2[0, pos], dim=-1)
            conf = probs[inp].item()
            
            marker = " (prompt)" if pos < plen else ""
            print(f"  pos={pos:2d}: input={inp_char!r:5s} pred={pred_char!r:5s} {match} conf={conf:.4f}{marker}")
    
    # ===== TEST 3: Mask only LAST token — does model predict it? =====
    print("\n" + "="*70)
    print("TEST 3: Only last response token masked — can model predict it?")
    print("="*70)
    
    input_ids3 = input_ids2.clone()
    # Mask only the EOS position
    eos_pos = len(prompt_ids) + len(target_ids)
    input_ids3[0, eos_pos] = mask_id
    
    with torch.no_grad():
        outputs3 = model(input_ids=input_ids3, attention_mask=attn_mask2)
        logits3 = outputs3.logits
        logits3[..., mask_id] = float('-inf')
        pred_at_eos = logits3[0, eos_pos].argmax().item()
        pred_char = tokenizer.decode([pred_at_eos])
        print(f"  EOS position ({eos_pos}): pred={pred_char!r} (expected [EOS]={eos_id})")
        print(f"  Correct: {pred_at_eos == eos_id}")
    
    # ===== TEST 4: Step-by-step unmasking (LLaDA style) =====
    print("\n" + "="*70)
    print("TEST 4: LLaDA-style iterative unmasking (20 steps)")
    print("="*70)
    
    num_steps = 20
    x = input_ids.clone()  # Start with all masks
    maskable = torch.zeros(1, total_len, dtype=torch.bool, device='cuda')
    maskable[0, plen:plen+gen_len] = True
    
    for step in range(num_steps):
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=attn_mask)
            lgt = out.logits
            lgt[..., mask_id] = float('-inf')
            lgt[..., pad_id] = float('-inf')
            
            log_scores = torch.log_softmax(lgt, dim=-1)
            best_scores, best_tokens = log_scores.max(dim=-1)
            
            is_masked = (x == mask_id) & maskable
            n_masked = is_masked.sum().item()
            
            # Fill all masks with predictions
            x0 = x.clone()
            x0[is_masked] = best_tokens[is_masked]
            
            if step < num_steps - 1 and n_masked > 0:
                remask_rate = (num_steps - 1 - step) / num_steps
                mask_pos = is_masked[0].nonzero(as_tuple=True)[0]
                n_remask = int(len(mask_pos) * remask_rate)
                
                if n_remask > 0:
                    pos_scores = best_scores[0, mask_pos]
                    _, sorted_idx = pos_scores.sort()
                    remask_idx = mask_pos[sorted_idx[:n_remask]]
                    x0[0, remask_idx] = mask_id
            
            x = x0
            
            # Decode current state
            current = []
            for pos in range(plen, plen + gen_len):
                tok = x[0, pos].item()
                if tok == mask_id:
                    current.append('_')
                elif tok == eos_id:
                    current.append('[E]')
                else:
                    current.append(tokenizer.decode([tok]))
            
            new_masked = (x[0, plen:plen+gen_len] == mask_id).sum().item()
            print(f"  Step {step+1:2d}: {''.join(current)} ({new_masked} masks left)")
    
    # Extract final prediction
    gen_tokens = x[0, plen:plen+gen_len].tolist()
    valid = []
    for tok in gen_tokens:
        if tok == mask_id:
            continue
        if tok == eos_id:
            break
        if tok == pad_id:
            break
        valid.append(tok)
    final_pred = tokenizer.decode(valid)
    
    print(f"\n  Final prediction: {final_pred!r}")
    print(f"  Target:           {target!r}")
    print(f"  Match: {final_pred.strip() == target.strip()}")


if __name__ == "__main__":
    main()
