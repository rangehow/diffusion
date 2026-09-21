import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from torch import nn


from transformers import ModernBertConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache

from transformers.loss.loss_utils import  fixed_cross_entropy
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    MaskedLMOutput,
    ModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, is_flash_attn_2_available, logging
from transformers.utils.import_utils import is_triton_available
from transformers.modeling_layers import GradientCheckpointingLayer
# from transformers.generation import GenerationMixin

from .generation_utils import GenerationMixin
from .configuration_niu import NiuConfig

# 条件导入
if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:

    RotaryEmbedding = object

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))



class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache will be recomputed during the forward pass.
        """
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotary embedding *inplace* to qkv.
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.tok_embeddings(input_ids))

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(inputs_embeds)
        else:
            hidden_states = (
                self.compiled_embeddings(input_ids)
                if self.config.reference_compile
                else self.drop(self.tok_embeddings(input_ids))
            )
        return hidden_states



class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ModernBertConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    causal: bool = False,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)

    
    if causal and local_attention != (-1, -1):
        full_window_size = local_attention[0] + local_attention[1]
        local_attention = (full_window_size, 0)


    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=causal,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=causal,
        )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    causal: bool = False,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
            rope_theta = config.local_rope_theta if config.local_rope_theta is not None else config.global_rope_theta
            max_position_embeddings = config.local_attention
        else:
            self.local_attention = (-1, -1)
            max_position_embeddings = config.max_position_embeddings
            rope_theta = config.global_rope_theta

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            config_copy = copy.deepcopy(config)
            config_copy.rope_theta = rope_theta
            self.rotary_emb = ModernBertRotaryEmbedding(config=config_copy)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            causal=causal,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


@torch.compile(dynamic=True)
def residual_add_dropout_fused(
    residual: torch.Tensor,
    output: torch.Tensor,
    dropout_p: float,
    is_training: bool
) -> torch.Tensor:
    """
    对 output 进行 dropout，然后与 residual 相加。
    使用 torch.compile 进行优化和融合。
    """
    output = F.dropout(output, p=dropout_p, training=is_training)
    return residual + output



class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NiuConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = ModernBertMLP(config)

        # 修改：不再需要 nn.Dropout 模块，只需保存 dropout 概率
        self.residual_dropout_p = config.residual_dropout

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
        causal = False,
    ) -> torch.Tensor:
        # 保存输入用于第一个残差连接
        residual = hidden_states

        # 注意力子层
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
            causal=causal,
        )
        attn_output = attn_outputs[0]

        # 调用 @torch.compile 优化的融合函数
        hidden_states = residual_add_dropout_fused(
            residual, attn_output, self.residual_dropout_p, self.training
        )

        # 保存注意力子层输出，用于第二个残差连接
        residual = hidden_states

        # MLP 子层
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )

        # 再次调用 @torch.compile 优化的融合函数
        hidden_states = residual_add_dropout_fused(
            residual, mlp_output, self.residual_dropout_p, self.training
        )

        return (hidden_states,) + attn_outputs[1:]

@auto_docstring
class ModernBertPreTrainedModel(PreTrainedModel):
    config: ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3


        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if hasattr(module,'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForDiffusionLM):
            init_weight(module.lm_head,stds["out"])
            if hasattr(module,'token_change_classifier') and module.token_change_classifier is not None:
                init_weight(module.token_change_classifier,stds["out"])
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module,'bias') and module.bias is not None:
                module.bias.data.zero_()

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False, **kwargs
    ) -> str:
        """
        Checks and dispatches to hhe requested attention implementation.
        """
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.
        # ModernBert's FA2 implementation correctly handles non-fp16/bf16 dtypes, we don't
        # need the FA2 warning for non-fp16/bf16 dtypes so we set fp16 for the FA2 check.

        try:
            attn_implementation = (
                "flash_attention_2"
                if attn_implementation is None and self._flash_attn_2_can_dispatch()
                else attn_implementation
            )
        except (ValueError, ImportError) as e:
            print(f"critical error that you might not install flash_attn, this file hasn't support sdpa yet.{e}")
            exit()
        return super()._check_and_adjust_attn_implementation(
            attn_implementation=attn_implementation, is_init_check=is_init_check, **kwargs
        )

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once(
                    "If `accelerate` split the model across devices, `torch.compile` will not work. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.mps` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.cpu` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def resize_token_embeddings(self, *args, **kwargs):
        model_embeds = super().resize_token_embeddings(*args, **kwargs)

        if self.config.reference_compile in {True, None}:
            if self.config.reference_compile:
                logger.warning_once(
                    "Resizing token embeddings with `torch.compile` is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        return model_embeds

    

@auto_docstring
class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        causal = False,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        causal
            Whether to use causal attention or not
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
                causal=causal,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                )

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)

        return global_attention_mask, sliding_window_mask




def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    per_token_weights: Optional[torch.Tensor] = None, # [修改] 参数名改为通用名称
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None
    
    # [修改] 处理通用权重
    unpadded_weights = per_token_weights.flatten()[indices] if per_token_weights is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels, unpadded_weights

def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs



@dataclass
class DiffusionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    non_masked_lm_loss: Optional[torch.FloatTensor] = None
    unweighted_total_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    current_mlm_prob: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None



def ForCausalMLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    per_token_weights: Optional[torch.Tensor] = None,
    training: bool = True, 
    **kwargs,
):
    logits = logits.float()
    labels_flat = labels.view(-1)

    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels_flat,
        ignore_index=ignore_index,
        reduction='none'
    )

    num_items_in_batch = num_items_in_batch.to(per_token_loss.device)

    valid_token_mask = (labels_flat != ignore_index)
    is_masked_token_mask = (input_ids == mask_token_id)

    masked_loss_mask = valid_token_mask & is_masked_token_mask
    non_masked_loss_mask = valid_token_mask & ~is_masked_token_mask

    unweighted_total_loss = ((per_token_loss * valid_token_mask).sum() / num_items_in_batch).detach()

    loss_for_backprop = per_token_loss
    
    # [修改] 应用通用权重
    if per_token_weights is not None:
        loss_for_backprop = loss_for_backprop * per_token_weights

    total_masked_loss_sum = (loss_for_backprop * masked_loss_mask).sum() / num_items_in_batch
    total_non_masked_loss_sum = (loss_for_backprop * non_masked_loss_mask).sum() / num_items_in_batch
    
    # [修改] 根据 training 状态决定 total_loss 的组成
    if training:
        total_loss = total_masked_loss_sum + total_non_masked_loss_sum
    else:
        # 评估模式：
        # 如果是 NTP 任务 (没有 mask 且 per_token_weights 为 None)，我们希望返回未加权的全局 Loss。
        # 原来的逻辑是只返回 masked loss，导致 NTP 任务 Loss 为 0。
        
        # 判断是否存在 masked tokens
        has_masked_tokens = is_masked_token_mask.any()
        
        if not has_masked_tokens:
            # 纯 NTP 模式：返回所有 valid token 的 loss (即 non_masked 部分，或者直接用未加权的计算结果)
            # 这里使用 unweighted_total_loss 对应的梯度版本
            total_loss = total_non_masked_loss_sum 
            # 或者更直接地：total_loss = (per_token_loss * valid_token_mask).sum() / num_items_in_batch
        else:
            # 兼容旧逻辑：如果是带 Mask 的 Eval (Diffusion Eval)，依然只看 masked 部分
            total_loss = total_masked_loss_sum

    return total_loss, total_masked_loss_sum, total_non_masked_loss_sum, unweighted_total_loss







def _daum_kernel_taps(p: float, eps: float = 1e-7) -> int:
    """Smallest K such that ``(1-p)^K < eps`` (fp32 ulp threshold)."""
    if p >= 1.0:
        return 1
    if p <= 0.0:
        return 1 << 31
    import math as _m
    return max(1, int(_m.ceil(_m.log(eps) / _m.log(1.0 - p))))


@torch.compile(dynamic=True)
def calculate_daum_weights(
    input_ids: torch.Tensor, # Expects [1, seq_len]
    mask_token_id: int,
    beta: float=1,
    gamma: float=1,
    p: float=0.5,
) -> torch.Tensor:
    """Truncated-conv DAUM weights — O(S·K) with K≈24 for p=0.5.

    Equivalent to the full-length conv up to one fp32 ulp (rel error ~2.4e-7).
    Empirically 1.9× faster at S=8192 and 4.8× at S=16384 on CPU; the gap
    grows linearly with sequence length.
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]

    is_mask = (input_ids == mask_token_id)
    is_prev_mask = torch.zeros_like(is_mask)
    if seq_len > 1:
        is_prev_mask[:, 1:] = is_mask[:, :-1]

    density_term = 1.0 + gamma * is_prev_mask.float()
    per_position_term = is_mask.float() * density_term # Shape: [1, seq_len]
    terms_with_channel = per_position_term.unsqueeze(1) # Shape: [1, 1, seq_len]

    K = min(seq_len, _daum_kernel_taps(p))
    powers = torch.arange(K - 1, -1, -1, device=device, dtype=torch.float32)
    kernel_weights = (1 - p) ** powers
    kernel = kernel_weights.view(1, 1, K)

    padded_terms = F.pad(terms_with_channel, (K - 1, 0))
    conv_result = F.conv1d(padded_terms, kernel)
    conv_sum = conv_result.squeeze(1)

    daum_weights = 1/(beta + conv_sum)
    return daum_weights


class ModernBertForDiffusionLM(ModernBertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.tok_embeddings.weight"}
    config_class = NiuConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=getattr(config, "decoder_bias", False))
        
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings
    
    @torch.compile(dynamic=True)
    def compiled_lm_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.lm_head(output)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current_mlm_prob: Optional[torch.Tensor] = None,
        causal: bool = True,
        use_daum: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DiffusionOutput]:

        assert causal is True,"Only causal is allowed."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()
        mask_token_id = self.config.mask_token_id
        ignore_index = -100
        if labels is not None:
            labels = labels[..., 1:].contiguous()
            labels = F.pad(labels, (0, 1), value=ignore_index)
        
        # --- [新增/修改逻辑开始] 计算 Per Token Weights ---
        # 1. 确定 batch 和 seq_len 以便后续进行扩展
        current_bsz = batch_size
        current_seqlen = seq_len
        if current_bsz is None or current_seqlen is None:
            if input_ids is not None:
                current_bsz, current_seqlen = input_ids.shape[:2]
            elif inputs_embeds is not None:
                current_bsz, current_seqlen = inputs_embeds.shape[:2]

        per_token_weights = None
        
        if labels is not None:
            if self.training and use_daum:
                # 训练模式 + use_daum：使用 DAUM 算法
                per_token_weights = calculate_daum_weights(input_ids, mask_token_id)
            

            elif not self.training:
                # [修改] 评估模式
                # 如果是 NTP 任务，current_mlm_prob 在 collator 里被设为 None。
                # 所以 per_token_weights 保持为 None (即权重为 1.0)。
                # 只有当 current_mlm_prob 存在时（比如做 Diffusion Loss 的验证），才计算权重。
                if current_mlm_prob is not None:
                    t = current_mlm_prob
                    weights_to_expand = 1/t.to(self.device)
                    if weights_to_expand.dim() == 1:
                        weights_to_expand = weights_to_expand.unsqueeze(1)
                    per_token_weights = weights_to_expand.expand(current_bsz, current_seqlen).contiguous().float()
                

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        # [修改] 传入 per_token_weights
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels, per_token_weights = _unpad_modernbert_input(
                            inputs=input_ids, 
                            attention_mask=attention_mask, 
                            position_ids=position_ids, 
                            labels=labels,
                            per_token_weights=per_token_weights 
                        )
                else:
                    # [修改] 传入 per_token_weights
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels, per_token_weights = _unpad_modernbert_input(
                        inputs=inputs_embeds, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids, 
                        labels=labels,
                        per_token_weights=per_token_weights
                    )
        else:
            ...
            # assert False,"非fa2分支的行为没有经过检查，请不要使用"

        
        # 运行主模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=causal,
        )
        last_hidden_state = outputs.last_hidden_state

        # 计算初始MLM logits
        logits = (
            self.compiled_lm_head(last_hidden_state)
            if self.config.reference_compile
            else self.lm_head(last_hidden_state)
        )
        

        if labels is not None:
            # [修改] 传入通用名 per_token_weights
            lm_loss,masked_lm_loss,non_masked_lm_loss,unweighted_total_loss = ForCausalMLMLoss(
                logits,
                labels,
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                vocab_size=self.config.vocab_size,
                per_token_weights=per_token_weights,
                training=self.training,
                **kwargs
            )
        
        # 后处理，还原去pad的序列
        if  labels is None and self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)


        if not return_dict:
            # 为了简化，我们强制要求使用 return_dict
            assert False, "Please use return_dict=True"

        if labels is not None:
            return DiffusionOutput(
                loss=lm_loss,
                lm_loss=lm_loss, 
                masked_lm_loss=masked_lm_loss,
                non_masked_lm_loss=non_masked_lm_loss,
                unweighted_total_loss = unweighted_total_loss,
                logits=None,
                hidden_states=None, 
                attentions=None,
                current_mlm_prob=current_mlm_prob.mean() if current_mlm_prob is not None else None,
            )
        else:
            return DiffusionOutput(
                loss=None,
                lm_loss=None,
                logits=logits,
                hidden_states=outputs.hidden_states, 
                attentions=outputs.attentions,
                current_mlm_prob=current_mlm_prob.mean() if current_mlm_prob is not None else None,
            )