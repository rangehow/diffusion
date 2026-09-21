# modeling_bd3lm.py

import math
import typing
import einops
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import MaskedLMOutput
from transformers.loss.loss_utils import fixed_cross_entropy
from typing import Optional, Union, Tuple, List, Dict, Any
from tqdm import tqdm
import numpy as np

try:

    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
    print("flex_attention enabled")
except ImportError as e:
    print(f"flex_attention failed {e}")
    FLEX_ATTN_AVAILABLE = False

from .configuration_bd3lm import BD3LMConfig

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

# -----------------------------------------------------------------------------
#                               Helper Functions
# -----------------------------------------------------------------------------

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask.
    """
    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                          (q_idx - n) // block_size,
                          q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                           (kv_idx - n) // block_size,
                           kv_idx // block_size)

    # 1. Block Diagonal Mask (M_BD)
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # 2. Offset Block-Causal Mask (M_OBC)
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # 3. Block-Causal Mask (M_BC)
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    return block_diagonal | offset_block_causal | block_causal

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)

def bias_dropout_add_scale(x, bias, scale, residual, prob, training):
    # scale is used for gating, it should be [B, S, D] here.
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out

@torch.compile(dynamic=True)
def bias_dropout_add_scale_fused_train(x, bias: typing.Optional[torch.Tensor], scale, residual: typing.Optional[torch.Tensor], prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)

@torch.compile(dynamic=True)
def bias_dropout_add_scale_fused_inference(x, bias: typing.Optional[torch.Tensor], scale, residual: typing.Optional[torch.Tensor], prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

@torch.compile(dynamic=True)
def modulate_fused(x, shift, scale):
    # FIX: Removed hardcoded .unsqueeze(1).
    # This allows `shift` and `scale` to be spatial [B, S, D] matching `x`.
    return x * (1 + scale) + shift

# -----------------------------------------------------------------------------
#                               Loss Function
# -----------------------------------------------------------------------------

def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    per_token_weights: Optional[torch.Tensor] = None,
    **kwargs,
):
    logits = logits.float()
    
    # Flatten
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    
    # Case 1: Standard Cross Entropy (No weights)
    if per_token_weights is None:
        labels = labels.to(logits.device)
        loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
        return loss

    # Case 2: Weighted Cross Entropy (For Diffusion: weight ~ 1/t)
    per_token_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
    
    weights = per_token_weights.reshape(-1).to(per_token_loss.device)
    # Ensure types match for mixed precision
    weights = weights.type_as(per_token_loss)

    if per_token_loss.shape != weights.shape:
        if weights.shape[0] != per_token_loss.shape[0]:
             raise ValueError(f"Shape mismatch: loss {per_token_loss.shape} vs weights {weights.shape}")
        
    weighted_loss = per_token_loss * weights
    
    if num_items_in_batch is None:
        num_valid_tokens = (labels != ignore_index).sum()
    else:
        num_valid_tokens = num_items_in_batch
        
    if torch.is_tensor(num_valid_tokens):
        num_valid_tokens = num_valid_tokens.to(weighted_loss.device).float()
    else:
        num_valid_tokens = torch.tensor(num_valid_tokens, device=weighted_loss.device, dtype=torch.float32)

    total_weighted_loss = weighted_loss.sum()
    
    if num_valid_tokens > 0:
        final_loss = total_weighted_loss / num_valid_tokens
    else:
        final_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)


    return final_loss

# -----------------------------------------------------------------------------
#                               Layers & Embeddings
# -----------------------------------------------------------------------------

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq_len, qkv, head, dim
            # Expand for broadcasting
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # v identity
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    # qkv: [B, S, 3, H, D]
    if cos.shape[1] > qkv.shape[1]:
        cos = cos[:, :qkv.shape[1], ...]
        sin = sin[:, :qkv.shape[1], ...]
    return (qkv * cos) + (rotate_half(qkv) * sin)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.amp.autocast('cuda',enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        
        # FIX: Correct broadcasting for arbitrary t shape (e.g., [B, L])
        # t.unsqueeze(-1) -> [..., 1]
        # freqs -> [half]
        # Result -> [..., half]
        args = t.float().unsqueeze(-1) * freqs
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DDiTBlock(nn.Module):
    def __init__(self, n, block_size, dim, n_heads, cond_dim, causal=False,
                 mlp_ratio=4, dropout=0.1, adaln=True, attn_backend='sdpa'):
        super().__init__()
        self.n = n 
        self.block_size = block_size
        self.n_heads = n_heads
        self.attn_backend = attn_backend
        self.kv_cache = None
        self.cache_idx = 0
        self.causal = causal

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.adaln = adaln
        if self.adaln:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
    
    def get_qkv(self, x, rotary_cos_sin, store_kv=False):
        if self.kv_cache is not None:
            new_qkv = self.attn_qkv(x)
            self.kv_cache[:, self.cache_idx:self.cache_idx+self.block_size] = new_qkv
            qkv = self.kv_cache[:, :self.cache_idx+self.block_size].clone()
        else:
            qkv = self.attn_qkv(x)
        
        if store_kv:
            self.cache_idx += self.block_size
            if self.cache_idx >= self.n:
                 self.cache_idx = self.n - self.block_size
                 self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

        qkv = einops.rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        with torch.amp.autocast('cuda',enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb_torchscript(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        return qkv

    def cross_attn(self, x, qkv, mask=None):
        scale = qkv.shape[-1]
        qkv = qkv.transpose(1, 3) # B, 3, H, S, D
        mask = mask.bool() if mask is not None else None
        
        x_out = F.scaled_dot_product_attention(
            query=qkv[:, :, 0],
            key=qkv[:, :, 1],
            value=qkv[:, :, 2],
            attn_mask=mask,
            is_causal=self.causal,
            scale=1 / math.sqrt(scale))
            
        x_out = x_out.transpose(1, 2)
        x_out = einops.rearrange(x_out, 'b s h d -> b s (h d)')
        return x_out
    
    def cross_attn_flex(self, qkv, mask=None):
        qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
        x = fused_flex_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
        x = einops.rearrange(x, 'b h s d -> b s (h d)')
        return x
  
    def forward(self, x, rotary_cos_sin, c=None, mask=None, sample_mode=False, store_kv=False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Handle AdaLN modulation if c is provided
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
        
        use_adaln = self.adaln and (c is not None)

        if use_adaln:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)

        x_skip = x
        if use_adaln:
            x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        else:
            x = self.norm1(x)

        if mask is not None and not sample_mode:
            n = x.shape[1] // 2 
            qkv_x = self.get_qkv(x[:, :n], rotary_cos_sin)
            qkv_x0 = self.get_qkv(x[:, n:], rotary_cos_sin)
            qkv = torch.cat((qkv_x, qkv_x0), dim=1)
        else:
            qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)

        if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            from torch.nn.attention.flex_attention import BlockMask
            if isinstance(mask, BlockMask):
                for attr in ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices"]:
                    t = getattr(mask, attr)
                    if torch.is_tensor(t):
                        torch._dynamo.mark_static(t)


            x = self.cross_attn_flex(qkv, mask=mask)
        else:
            x = self.cross_attn(x, qkv, mask=mask)
        
        if self.kv_cache is not None:
            x = x[:, -self.block_size:]

        # MLP with Gating logic matching Reference
        if use_adaln:
            # Gate derived from timestep
            x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            # Identity Gate (Scale = 1) equivalent to standard Transformer
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)

        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, dim)
        torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding(x)

class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaln=True):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaln = adaln
        if self.adaln:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        if self.adaln and c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = modulate_fused(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x

# -----------------------------------------------------------------------------
#                               Backbone
# -----------------------------------------------------------------------------

class DITBackbone(nn.Module):
    def __init__(self, config: BD3LMConfig):
        super().__init__()
        self.config = config
        self.cross_attn = config.cross_attn
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n = config.model_length 

        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
        self.adaln = config.adaln
        if self.adaln:
            self.sigma_map = TimestepEmbedder(config.cond_dim)
        
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            blocks.append(DDiTBlock(
                self.n, self.block_size, config.hidden_dim, config.n_heads,
                config.cond_dim, causal=config.causal, dropout=config.dropout,
                adaln=config.adaln, attn_backend=config.attn_backend
            ))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.hidden_dim, config.vocab_size, config.cond_dim, adaln=config.adaln
        )
        
        self.mask = None
        if self.cross_attn:
            self.gen_mask(config.model_length, self.block_size, attn_backend=config.attn_backend)
            
        self.precision = torch.float32

    def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
        if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            self.mask = create_block_mask(
                partial(block_diff_mask, block_size=block_size, n=seqlen),
                B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
        else:
            self.mask = block_diff_mask(
                b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
                kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)

    def _get_dynamic_mask(self, seqlen, block_size, device):
        if self.config.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
             mask = create_block_mask(
                partial(block_diff_mask, block_size=block_size, n=seqlen),
                B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
        else:
            mask = block_diff_mask(
                b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
                kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
        return mask.to(device)

    def reset_kv_cache(self, batch_size: int = 1):
        """Reset KV cache for all blocks."""
        for block in self.blocks:
            block.kv_cache = torch.zeros(
                batch_size, 
                self.n, 
                3 * self.config.hidden_dim,
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype
            )
            block.cache_idx = 0

    def clear_kv_cache(self):
        """Clear KV cache for all blocks."""
        for block in self.blocks:
            block.kv_cache = None
            block.cache_idx = 0

    def forward(self, indices, sigma, sample_mode=False,
             store_kv=False, output_hidden_states=False):
        
        all_hidden_states = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden_states.append(x)
            
        c = None
        if self.adaln:
            # MODIFIED: If sigma is None, keep c as None unless configuration forces time conditioning.
            if sigma is not None:
                c = F.silu(self.sigma_map(sigma))
            elif self.config.time_conditioning:
                # Only if we enforce conditioning but inputs are missing, default to 0
                sigma_zero = torch.zeros(indices.shape, device=indices.device, dtype=self.precision)
                c = F.silu(self.sigma_map(sigma_zero))
            else:
                c = None
            
        mask = None
        rotary_cos_sin = None

        if self.cross_attn:
            curr_seq_len = x.shape[1] // 2
            
            if self.mask is not None and (self.mask.shape[-1] // 2 == curr_seq_len):
                mask = self.mask.to(x.device)
            else:
                mask = self._get_dynamic_mask(curr_seq_len, self.block_size, x.device)
            
            n = curr_seq_len
            
            if not sample_mode:
                rotary_cos_sin = self.rotary_emb(x[:, :n])
            else:
                if self.blocks[0].kv_cache is not None:
                    mask = None
                    accum_length = self.blocks[0].cache_idx + self.block_size
                    x_full = torch.zeros((x.shape[0], accum_length, x.shape[2]), device=x.device)
                    rotary_cos_sin = self.rotary_emb(x_full)
                else:
                    rotary_cos_sin = self.rotary_emb(x[:, :n])
                    mask = mask[n:n+x.shape[1], n:n+x.shape[1]]
        else:
            mask = None
            rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, 
                                   rotary_cos_sin,
                                   c,
                                   mask=mask,
                                   sample_mode=sample_mode,
                                   store_kv=store_kv)
                if output_hidden_states:
                    all_hidden_states.append(x)
            logits = self.output_layer(x, c)
            
        if self.cross_attn and not sample_mode:
            logits = logits[:, :n]
            all_hidden_states = [h[:, :n] for h in all_hidden_states]
            
        return logits, all_hidden_states

# -----------------------------------------------------------------------------
#                               HF Model Wrapper
# -----------------------------------------------------------------------------

class BD3LM(transformers.PreTrainedModel):
    config_class = BD3LMConfig
    base_model_prefix = "bd3lm"
    _tied_weights_keys = {"backbone.output_layer.linear.weight": "backbone.vocab_embed.embedding.weight"}

    def __init__(self, config: BD3LMConfig):
        super().__init__(config)
        self.config = config
        self.backbone = DITBackbone(config)
        
        self.mask_token_id = config.mask_token_id 
        if self.mask_token_id is None:
            self.mask_token_id = config.vocab_size - 1 

        # Generation-related attributes
        self.neg_infinity = -1000000.0

        self.post_init()

    def _apply_subs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Subs 参数化在 Logits 空间的简化实现。
        仅需屏蔽 Mask Token 本身的预测，不需要进行 LogSumExp 或 Softmax。
        """
        # 防止模型预测出 Mask Token 本身
        # 使用足够小的负数即可，-1e4 在 fp16/bf16 下通常安全
        neg_inf = -10000.0 
        
        # 原地修改 logits
        logits[..., self.mask_token_id] = neg_inf
        
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cond_ids: Optional[torch.LongTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.FloatTensor] = None,
        num_items_in_batch: Optional[int] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 1. 准备模型输入 (Cross Attention 处理)
        if self.config.cross_attn:
            if cond_ids is None:
                model_input = input_ids
            else:
                model_input = torch.cat([input_ids, cond_ids], dim=1)
        else:
            model_input = input_ids

        # 2. Backbone Forward
        # 传入 sigma=None，因为我们在这里不使用 Time Conditioning Embedding
        # 或者如果你想支持 Time Conditioning，可以在这里把 timesteps 传给 sigma
        logits, _ = self.backbone(
            indices=model_input,
            sigma=None,   
            sample_mode=False 
        )

        loss = None
        if labels is not None:
            # 截取 logits (如果是 cross-attn 模式，backbone 输出可能包含 cond 部分)
            n_tokens = input_ids.shape[1]
            logits = logits[:, :n_tokens]

            # =========================================================
            # [核心逻辑] 准备 Labels 实现 Subs 参数化
            # =========================================================
            final_labels = labels.clone()
            
            # A. 处理 Padding: Pad 位置不计算 Loss
            if attention_mask is not None:
                final_labels[attention_mask == 0] = -100
            elif self.config.pad_token_id is not None:
                final_labels[input_ids == self.config.pad_token_id] = -100

            # B. 处理 Subs 逻辑: 
            # 只有被 Mask 的位置才需要计算 Loss。
            # 对于 Clean Token (input_ids != mask_id)，我们已知其值，Loss 为 0 -> 设为 -100
            is_clean_token = (input_ids != self.mask_token_id)
            final_labels[is_clean_token] = -100
            # =========================================================

            # 3. 处理 Logits
            # 禁止预测 Mask Token 本身
            logits = self._apply_subs(logits)

            # 4. 准备 Loss 权重 (Weight ~ 1/t)
            per_token_weights = None
            if timesteps is not None:
                # timesteps 这里代表 noise level (mask ratio)
                # 加 1e-4 防止除以 0
                per_token_weights = 1.0 / (timesteps + 1e-4)

            # 5. 计算 Loss (Cross Entropy)
            # 此时 logits 是 raw logits，final_labels 已经正确设置了 ignore_index (-100)
            loss = ForMaskedLMLoss(
                logits=logits,
                labels=final_labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
                per_token_weights=per_token_weights,
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    # -------------------------------------------------------------------------
    #                         Generation Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution using Gumbel-max trick."""
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def _nucleus_sample(self, p_x0: torch.Tensor, nucleus_p: float = 1.0) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to probabilities."""
        if nucleus_p >= 1.0:
            return p_x0
        
        block_size = self.config.block_size
        p_x0_ = p_x0[:, -block_size:].clone()
        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= nucleus_p
        nucleus_mask[..., 0] = True  # Always keep at least one token
        sorted_probs = sorted_probs * nucleus_mask
        p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        p_x0_ = p_x0_ / p_x0_.sum(-1, keepdim=True)
        p_x0[:, -block_size:] = p_x0_
        return p_x0

    def _get_score(
        self, 
        x: torch.Tensor, 
        sigma: torch.Tensor,
        sample_mode: bool = True,
        store_kv: bool = False,
        nucleus_p: float = 1.0,
    ) -> torch.Tensor:
        """Get model score (probability distribution) for input x at noise level sigma."""
        logits, _ = self.backbone(
            indices=x,
            sigma=sigma,
            sample_mode=sample_mode,
            store_kv=store_kv,
        )
        
        # Apply subs parameterization: mask token gets -inf
        logits[:, :, self.mask_token_id] = self.neg_infinity
        
        # Convert to log probabilities
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        
        # Apply subs: unmasked positions get identity
        unmasked_indices = (x[:, :logits.shape[1]] != self.mask_token_id)
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, x[:, :logits.shape[1]][unmasked_indices]] = 0
        
        # Convert to probabilities and apply nucleus sampling
        probs = log_probs.exp()
        if nucleus_p < 1.0:
            probs = self._nucleus_sample(probs, nucleus_p)
        
        return probs

    def _ddpm_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        p_x0: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        nucleus_p: float = 1.0,
        first_hitting: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Perform a single DDPM update step.
        
        Args:
            x: Current sequence [batch_size, seq_len]
            t: Current timestep (noise level) [batch_size, 1]
            dt: Timestep delta
            p_x0: Cached probability distribution (optional)
            use_kv_cache: Whether to use KV caching
            nucleus_p: Nucleus sampling probability
            first_hitting: Whether to use first-hitting time sampler
            
        Returns:
            Tuple of (cached p_x0 or None, updated sequence)
        """
        block_size = self.config.block_size
        
        # Compute move chances (probability of being masked)
        # Using simple linear schedule: move_chance = t
        move_chance_t = t
        move_chance_s = t - dt
        
        # Compute mask probability for this step
        mask_prob = move_chance_s / (move_chance_t + 1e-8)
        
        # Get sigma for model input
        sigma_t = -torch.log(1 - move_chance_t.clamp(max=0.9999))
        
        # Get model predictions if not cached
        if p_x0 is None:
            if use_kv_cache:
                model_input = x[:, -block_size:]
            else:
                model_input = x
                
            p_x0 = self._get_score(
                model_input, 
                sigma_t, 
                sample_mode=True,
                store_kv=False,
                nucleus_p=nucleus_p,
            ).to(torch.float64)
            
            p_x0 = p_x0[:, -block_size:]

        # Sample new tokens
        if first_hitting:
            # First hitting time sampler: unmask one token at a time
            x_block = self._sample_categorical(p_x0)
            num_masked = (x[:, -block_size:] == self.mask_token_id).sum(-1)
            # Randomly select one masked position to unmask
            for b in range(x.shape[0]):
                masked_positions = (x[b, -block_size:] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(masked_positions) > 0:
                    idx = masked_positions[torch.randint(len(masked_positions), (1,))]
                    mask = torch.zeros(block_size, device=x.device, dtype=x.dtype)
                    mask[idx] = 1
                    x_block[b] = x_block[b] * mask + x[b, -block_size:] * (1 - mask)
        else:
            # Standard DDPM update
            q_xs = p_x0 * (1 - mask_prob)
            q_xs[:, :, self.mask_token_id] = mask_prob.squeeze(-1)
            x_block = self._sample_categorical(q_xs)
        
        # Copy flag: don't modify already unmasked tokens
        copy_flag = (x[:, -block_size:] != self.mask_token_id).to(x.dtype)
        x_block = copy_flag * x[:, -block_size:] + (1 - copy_flag) * x_block
        
        # Update sequence
        x_new = torch.cat([x[:, :-block_size], x_block], dim=-1)
        
        # Store KV cache if block is fully unmasked
        if use_kv_cache and self.mask_token_id not in x_block:
            _ = self.backbone(
                x_block.unsqueeze(1) if x_block.dim() == 1 else x_block,
                sigma_t,
                sample_mode=True,
                store_kv=True,
            )
        
        # Return None for p_x0 cache if sequence changed (need to recompute)
        if not torch.allclose(x_new, x):
            return None, x_new
        else:
            return p_x0, x_new

    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy of token distribution."""
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        return torch.special.entr(counts.float() / counts.sum()).sum()

    def _check_stop_conditions(
        self, 
        x: torch.Tensor, 
        eos_token_id: Optional[int] = None,
        variable_length: bool = False,
    ) -> Tuple[bool, torch.Tensor]:
        """
        Check if generation should stop.
        
        Args:
            x: Current sequence
            eos_token_id: EOS token ID for variable length generation
            variable_length: Whether to use variable length stopping
            
        Returns:
            Tuple of (should_stop, potentially_truncated_sequence)
        """
        stop = False
        truncate_idx = None
        
        # Check entropy (low entropy = repetitive/degenerate)
        entropy = self._compute_entropy(x[:, -256:] if x.shape[1] > 256 else x)
        if entropy < 4:
            stop = True
        
        if variable_length and eos_token_id is not None:
            # Check for EOS token
            eos_positions = (x == eos_token_id).nonzero(as_tuple=True)
            if len(eos_positions[0]) > 1:
                stop = True
                # Find second EOS and truncate there
                eos_indices = eos_positions[1]
                if len(eos_indices) > 1:
                    truncate_idx = min(eos_indices[1].item() + 1, x.shape[1])
            
            if entropy < 4:
                truncate_idx = max(x.shape[1] - 256, 1)
        
        if truncate_idx is not None:
            x = x[:, :truncate_idx]
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        return stop, x

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_steps: int = 256,
        nucleus_p: float = 1.0,
        use_kv_cache: bool = True,
        first_hitting: bool = False,
        variable_length: bool = False,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate sequences using semi-autoregressive block diffusion sampling.
        
        Args:
            input_ids: Optional input prompt tensor [batch_size, prompt_len].
                      If provided, generation continues from this prompt.
                      If None, generates from scratch starting with BOS token.
            attention_mask: Optional attention mask for input_ids [batch_size, prompt_len].
                           1 for real tokens, 0 for padding.
            max_length: Maximum total length of the sequence (prompt + generated).
                       Defaults to config.model_length.
            max_new_tokens: Maximum number of new tokens to generate.
                           If set, overrides max_length calculation.
            num_steps: Number of diffusion steps per block.
            nucleus_p: Nucleus sampling probability (1.0 = no nucleus sampling).
            use_kv_cache: Whether to use KV caching for efficiency.
            first_hitting: Use first-hitting time sampler (unmask one token at a time).
            variable_length: Allow variable length generation (stop at EOS).
            bos_token_id: Beginning of sequence token ID.
            eos_token_id: End of sequence token ID (for variable length).
            pad_token_id: Padding token ID.
            show_progress: Show progress bar.
            **kwargs: Additional arguments (for compatibility).
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        # Determine device
        device = next(self.parameters()).device
        
        # Handle input_ids
        if input_ids is not None:
            input_ids = input_ids.to(device)
            batch_size = input_ids.shape[0]
            prompt_len = input_ids.shape[1]
        else:
            batch_size = kwargs.get('batch_size', 1)
            prompt_len = 0
            
        # Determine generation length
        block_size = self.config.block_size
        
        if max_new_tokens is not None:
            # Generate exactly max_new_tokens (rounded up to block boundary)
            gen_len = ((max_new_tokens + block_size - 1) // block_size) * block_size
            seq_len = prompt_len + gen_len
        elif max_length is not None:
            seq_len = max_length
        else:
            seq_len = self.config.model_length
            
        # Ensure seq_len is a multiple of block_size
        seq_len = ((seq_len + block_size - 1) // block_size) * block_size
        
        # Calculate number of blocks to generate
        if prompt_len > 0:
            # Round prompt to block boundary for processing
            prompt_blocks = (prompt_len + block_size - 1) // block_size
            prompt_len_padded = prompt_blocks * block_size
            total_blocks = seq_len // block_size
            gen_blocks = total_blocks - prompt_blocks + 1  # +1 to complete partial prompt block
        else:
            total_blocks = seq_len // block_size
            gen_blocks = total_blocks
            prompt_len_padded = 0
        
        # Initialize KV cache if requested
        if use_kv_cache:
            self.backbone.reset_kv_cache(batch_size=batch_size)
        
        # Initialize sequence
        if input_ids is not None:
            # Start with prompt, pad to block boundary if needed
            if prompt_len < prompt_len_padded:
                # Pad prompt to block boundary with mask tokens
                padding = torch.full(
                    (batch_size, prompt_len_padded - prompt_len),
                    self.mask_token_id,
                    dtype=torch.long,
                    device=device
                )
                x = torch.cat([input_ids, padding], dim=1)
            else:
                x = input_ids.clone()
            
            # Track which positions are from prompt (should not be modified)
            prompt_mask = torch.zeros(prompt_len_padded, dtype=torch.bool, device=device)
            prompt_mask[:prompt_len] = True
            
            start_block = prompt_blocks - 1  # Start from the block containing end of prompt
        else:
            # Initialize with mask tokens for first block
            x = torch.full(
                (batch_size, block_size), 
                self.mask_token_id, 
                dtype=torch.long, 
                device=device
            )
            
            # Set BOS token if provided
            if bos_token_id is not None:
                x[:, 0] = bos_token_id
            
            prompt_mask = torch.zeros(block_size, dtype=torch.bool, device=device)
            if bos_token_id is not None:
                prompt_mask[0] = True  # Don't modify BOS
                
            start_block = 0
        
        ones = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        
        # Iterate through blocks
        iterator = range(start_block, total_blocks)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating blocks")
        
        for block_idx in iterator:
            # Add new masked block if needed
            current_len = x.shape[1]
            target_len = (block_idx + 1) * block_size
            
            if current_len < target_len:
                new_block = torch.full(
                    (batch_size, target_len - current_len),
                    self.mask_token_id,
                    dtype=torch.long,
                    device=device
                )
                x = torch.cat([x, new_block], dim=1)
                
                # Extend prompt mask
                prompt_mask_ext = torch.zeros(target_len - current_len, dtype=torch.bool, device=device)
                prompt_mask = torch.cat([prompt_mask, prompt_mask_ext])
            
            # Determine which indices to process (current block)
            block_start = block_idx * block_size
            block_end = (block_idx + 1) * block_size
            
            # Context window for model (limited by model_length)
            context_start = max(block_end - self.config.model_length, 0)
            fwd_idx = torch.arange(context_start, block_end, device=device)
            
            # Get prompt mask for current block
            block_prompt_mask = prompt_mask[block_start:block_end]
            
            # Skip if entire block is from prompt
            if block_prompt_mask.all():
                # Store KV cache for prompt blocks
                if use_kv_cache:
                    block_tokens = x[:, block_start:block_end]
                    sigma_zero = torch.zeros((batch_size, 1), device=device)
                    _ = self.backbone(
                        block_tokens,
                        sigma_zero,
                        sample_mode=True,
                        store_kv=True,
                    )
                continue
            
            # Diffusion timesteps
            dt = 1.0 / num_steps
            p_x0_cache = None
            
            if first_hitting:
                # First hitting time sampler
                t = 1.0
                while True:
                    # Check only non-prompt positions for mask tokens
                    block_tokens = x[:, block_start:block_end]
                    non_prompt_mask = ~block_prompt_mask
                    masked_in_block = (block_tokens == self.mask_token_id) & non_prompt_mask.unsqueeze(0)
                    
                    if not masked_in_block.any():
                        break
                    
                    num_masked = masked_in_block.sum(-1).float()
                    if num_masked.max() == 0:
                        break
                    u = np.random.rand()
                    t *= u ** (1.0 / num_masked.max().item())
                    
                    p_x0_cache, x_new = self._ddpm_update(
                        x=x[:, fwd_idx],
                        t=t * ones,
                        dt=dt,
                        p_x0=p_x0_cache,
                        use_kv_cache=use_kv_cache,
                        nucleus_p=nucleus_p,
                        first_hitting=True,
                    )
                    
                    # Only update non-prompt positions
                    x_block_new = x_new[:, -block_size:]
                    x_block_old = x[:, block_start:block_end]
                    update_mask = ~block_prompt_mask.unsqueeze(0).expand(batch_size, -1)
                    x[:, block_start:block_end] = torch.where(
                        update_mask, x_block_new, x_block_old
                    )
            else:
                # Standard DDPM sampling
                timesteps = torch.linspace(1, 0, num_steps, device=device)
                for step_idx in range(num_steps):
                    # Check only non-prompt positions
                    block_tokens = x[:, block_start:block_end]
                    non_prompt_mask = ~block_prompt_mask
                    masked_in_block = (block_tokens == self.mask_token_id) & non_prompt_mask.unsqueeze(0)
                    
                    if not masked_in_block.any():
                        break
                    
                    t = timesteps[step_idx]
                    
                    p_x0_cache, x_new = self._ddpm_update(
                        x=x[:, fwd_idx],
                        t=t * ones,
                        dt=dt,
                        p_x0=p_x0_cache,
                        use_kv_cache=use_kv_cache,
                        nucleus_p=nucleus_p,
                        first_hitting=False,
                    )
                    
                    # Only update non-prompt positions
                    x_block_new = x_new[:, -block_size:]
                    x_block_old = x[:, block_start:block_end]
                    update_mask = ~block_prompt_mask.unsqueeze(0).expand(batch_size, -1)
                    x[:, block_start:block_end] = torch.where(
                        update_mask, x_block_new, x_block_old
                    )
            
            # Store KV cache after block is complete
            if use_kv_cache and self.mask_token_id not in x[:, block_start:block_end]:
                block_tokens = x[:, block_start:block_end]
                sigma_zero = torch.zeros((batch_size, 1), device=device)
                _ = self.backbone(
                    block_tokens,
                    sigma_zero,
                    sample_mode=True,
                    store_kv=True,
                )
            
            # Check stopping conditions
            if x.shape[1] > 256:
                stop, x_truncated = self._check_stop_conditions(
                    x, 
                    eos_token_id=eos_token_id,
                    variable_length=variable_length
                )
                if stop:
                    if variable_length:
                        x = x_truncated
                        break
                    # For non-variable length, continue but log warning
        
        # Clear KV cache
        if use_kv_cache:
            self.backbone.clear_kv_cache()
        
        return x

    def get_input_embeddings(self):
        return self.backbone.vocab_embed.embedding

    def set_input_embeddings(self, value):
        self.backbone.vocab_embed.embedding = value

    def get_output_embeddings(self):
        return self.backbone.output_layer.linear

    def set_output_embeddings(self, new_embeddings):
        self.backbone.output_layer.linear = new_embeddings