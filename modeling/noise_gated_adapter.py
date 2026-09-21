"""
Noise-Gated Residual Adapter for Causal Diffusion LMs.

Zero-initialized at insertion, so the model is bit-for-bit identical to the
original ARM. During training with noisy contexts, the adapter activates
proportionally to the local noise fraction, absorbing denoising-specific
gradients and keeping the backbone sharp for NTP.

Usage:
    model = AutoModelForCausalLM.from_pretrained(...)
    model = insert_adapters(model, rank=64)
    # model is still exact ARM. Train with SuffixMaskCollator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NoiseGatedAdapter(nn.Module):
    """
    Low-rank adapter that activates proportionally to context noise.
    
    At initialization: up.weight = 0, so output = input (exact ARM).
    During training:   output = input + noise_fraction * up(silu(down(input)))
    At NTP inference:  noise_fraction = 0, output = input (zero overhead).
    """

    def __init__(self, hidden_dim: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        # Critical: zero-initialize up so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        noise_fraction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [total_tokens, hidden_dim] or [batch, seq_len, hidden_dim]
            noise_fraction: [total_tokens] or [batch, seq_len], values in [0, 1].
                            If None or all zeros, adapter is a no-op.
        """
        if noise_fraction is None:
            return hidden_states

        # Fast path: skip computation if no noise at all
        if noise_fraction.max().item() == 0:
            return hidden_states

        adapter_out = self.up(F.silu(self.down(hidden_states)))

        # Expand noise_fraction to match hidden_states shape
        if noise_fraction.dim() < hidden_states.dim():
            noise_fraction = noise_fraction.unsqueeze(-1)

        return hidden_states + noise_fraction * adapter_out


def insert_adapters(
    model: nn.Module,
    rank: int = 64,
    verbose: bool = True,
) -> nn.Module:
    """
    Insert zero-initialized noise-gated adapters into a HuggingFace causal LM.
    
    Works with Qwen3, LLaMA, Mistral, and any model whose layers are accessible
    via model.model.layers (standard HF layout).
    
    The adapters are attached AFTER each transformer layer's forward pass.
    We wrap each layer's forward method to:
      1. Run the original forward
      2. Apply the adapter to the output hidden_states
      3. Return the modified output
    
    Args:
        model: HuggingFace CausalLM model
        rank: adapter bottleneck dimension
        verbose: whether to print parameter counts
    
    Returns:
        model with adapters inserted (same object, modified in-place)
    """
    # Find the layers list
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(
            "Cannot find transformer layers. Expected model.model.layers or model.transformer.h"
        )

    hidden_dim = model.config.hidden_size

    for i, layer in enumerate(layers):
        adapter = NoiseGatedAdapter(hidden_dim, rank)
        # Register as a submodule so parameters are tracked
        layer.add_module("adapter", adapter)

        # Store original forward
        original_forward = layer.forward

        # Create a closure that captures the original forward and adapter
        def make_wrapped_forward(orig_fn, adpt):
            def wrapped_forward(*args, **kwargs):
                # Extract noise_fraction before passing to original forward
                noise_fraction = kwargs.pop("noise_fraction", None)

                # Call original layer forward
                outputs = orig_fn(*args, **kwargs)

                # Apply adapter to hidden_states
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                    hidden_states = adpt(hidden_states, noise_fraction)
                    outputs = (hidden_states,) + outputs[1:]
                else:
                    outputs = adpt(outputs, noise_fraction)

                return outputs

            return wrapped_forward

        layer.forward = make_wrapped_forward(original_forward, adapter)

    # Store noise_fraction propagation hook on the model
    # This ensures noise_fraction flows through all layers
    _patch_model_forward_for_noise_fraction(model)

    if verbose:
        adapter_params = sum(
            p.numel() for n, p in model.named_parameters() if "adapter" in n
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Inserted adapters: {adapter_params/1e6:.1f}M params "
            f"({adapter_params/total_params*100:.2f}% of {total_params/1e6:.0f}M), "
            f"rank={rank}, hidden_dim={hidden_dim}, num_layers={len(layers)}"
        )
        print(
            f"[NoiseGatedAdapter] Inserted: {adapter_params/1e6:.1f}M / "
            f"{total_params/1e6:.0f}M ({adapter_params/total_params*100:.2f}%), "
            f"rank={rank}"
        )

    return model


def _patch_model_forward_for_noise_fraction(model: nn.Module):
    """
    Patch the model's forward to:
    1. Accept noise_fraction in kwargs
    2. Propagate it through each layer via kwargs
    
    For Qwen3/LLaMA, the model.model (the inner model without lm_head)
    iterates over layers in a simple loop. We need noise_fraction to reach
    each layer's wrapped forward.
    
    Strategy: Patch model.model.forward to inject noise_fraction into each
    layer call. This is cleaner than patching each layer individually.
    """
    inner_model = model.model if hasattr(model, "model") else model.transformer

    original_inner_forward = inner_model.forward

    def patched_inner_forward(*args, **kwargs):
        noise_fraction = kwargs.pop("noise_fraction", None)
        
        # Store on the inner model temporarily so layers can access it
        inner_model._current_noise_fraction = noise_fraction
        
        try:
            outputs = original_inner_forward(*args, **kwargs)
        finally:
            inner_model._current_noise_fraction = None
        
        return outputs

    inner_model.forward = patched_inner_forward

    # Also patch each layer to read from inner_model._current_noise_fraction
    for layer in (inner_model.layers if hasattr(inner_model, "layers") else inner_model.h):
        wrapped_fwd = layer.forward  # This is already our make_wrapped_forward closure

        def make_nf_aware_forward(existing_wrapped, parent_model):
            def nf_aware_forward(*args, **kwargs):
                # Inject noise_fraction from parent if not already present
                if "noise_fraction" not in kwargs:
                    nf = getattr(parent_model, "_current_noise_fraction", None)
                    if nf is not None:
                        kwargs["noise_fraction"] = nf
                return existing_wrapped(*args, **kwargs)
            return nf_aware_forward

        layer.forward = make_nf_aware_forward(wrapped_fwd, inner_model)

    # Finally, patch the outermost model.forward to accept and pass noise_fraction
    original_model_forward = model.forward

    def patched_model_forward(*args, **kwargs):
        noise_fraction = kwargs.pop("noise_fraction", None)
        
        # The inner model's patched forward will pick this up
        if noise_fraction is not None:
            # We need to pass it through to model.model.forward
            # The simplest way: temporarily store it
            inner_model._current_noise_fraction = noise_fraction
        
        try:
            outputs = original_model_forward(*args, **kwargs)
        finally:
            inner_model._current_noise_fraction = None
        
        return outputs

    model.forward = patched_model_forward
