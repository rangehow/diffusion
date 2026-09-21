# correct_compare.py
import torch
import sys

# Use YOUR model class to load - this ensures same parameter order
from diffusion.modeling.modeling_niu import ModernBertForDiffusionLM
from diffusion.modeling.configuration_niu import NiuConfig

checkpoint_path = sys.argv[1]

# Load EMA state
ema_state = torch.load(f"{checkpoint_path}/ema_state.pt", map_location='cpu')
shadow_params = ema_state['shadow_params']

# Load model the SAME way as eval_loss.py does
config = NiuConfig.from_pretrained(checkpoint_path)
model = ModernBertForDiffusionLM.from_pretrained(checkpoint_path, config=config)

# Get params in SAME order as EMA was saved
model_params = [p for p in model.parameters() if p.requires_grad]

print(f"EMA params: {len(shadow_params)}")
print(f"Model params: {len(model_params)}")

print(f"\n=== Correct Weight Comparison ===")
all_match = True
for i, (ema_p, model_p) in enumerate(zip(shadow_params, model_params)):
    if ema_p.shape != model_p.shape:
        print(f"❌ Param {i}: SHAPE MISMATCH! EMA={ema_p.shape}, Model={model_p.shape}")
        all_match = False
    elif i < 10:  # Print first 10
        diff = (ema_p - model_p).abs().mean().item()
        print(f"✓ Param {i}: shape={ema_p.shape}, diff={diff:.8f}")

if all_match:
    print(f"\n✅ All {len(shadow_params)} parameters match in shape!")

    # Calculate average difference
    total_diff = sum((ep - mp).abs().mean().item() 
                     for ep, mp in zip(shadow_params, model_params))
    avg_diff = total_diff / len(shadow_params)
    print(f"📊 Average weight difference: {avg_diff:.8f}")
    print(f"   (Should be small but non-zero, ~0.001-0.01 is typical)")
else:
    print(f"\n❌ Shape mismatches found - there's a real problem!")