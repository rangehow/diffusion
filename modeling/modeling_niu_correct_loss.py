
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


from transformers import ModernBertConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.loss.loss_utils import  fixed_cross_entropy
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm =nn.Identity()
        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = (
                self.compiled_embeddings(input_ids)
                if self.config.reference_compile
                else self.drop(self.norm(self.tok_embeddings(input_ids)))
            )
        return hidden_states



class ModernBertMLP(nn.Module):
    """Applies the SwishGLU at the end of each ModernBERT layer.

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
    def __init__(self, config, dim: int, base: float, device: Optional[torch.device] = None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
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
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
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
    local_attention: Tuple[int, int],
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
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            self.rotary_emb = ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)

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


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        # if layer_id == 0:
        #     self.attn_norm = nn.Identity()
        # else:
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = ModernBertMLP(config)

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
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


@auto_docstring
class ModernBertPreTrainedModel(PreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.
        # ModernBert's FA2 implementation correctly handles non-fp16/bf16 dtypes, we don't
        # need the FA2 warning for non-fp16/bf16 dtypes so we set fp16 for the FA2 check.

        if config._attn_implementation_internal is None:

            config._attn_implementation_internal = "flash_attention_2"
            try:
                return cls._check_and_enable_flash_attn_2(
                    config,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    hard_check_only=False,
                    check_device_map=check_device_map,
                )
            except (ValueError, ImportError):
                config._attn_implementation_internal = None
        return super()._autoset_attn_implementation(
            config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            check_device_map=check_device_map,
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

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    sliding_window_mask,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                    output_attentions,
                    causal,
                )
            else:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
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

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels



def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
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
    """
    我们自定义的模型输出类
    """
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    non_masked_lm_loss: Optional[torch.FloatTensor] = None
    corrector_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    corrector_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    current_mlm_prob: Optional[torch.Tensor] = None


def ForCausalLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    per_token_weights: Optional[torch.Tensor] = None,
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

    # 3. Apply per-token weights if they are provided.
    # If no weights are given, this step is skipped, and we effectively use a weight of 1
    # for every valid token, because we will sum the `per_token_loss` directly.
    if per_token_weights is not None:
        weights = per_token_weights.reshape(-1).to(per_token_loss.device)
        
        if per_token_loss.shape != weights.shape:
            raise ValueError(
                f"Shape mismatch between per_token_loss ({per_token_loss.shape}) "
                f"and weights ({weights.shape})."
            )
            
        # Apply the weights to each token's loss
        final_per_token_loss = per_token_loss * weights
    else:
        # If no weights, the final loss per token is just the original loss
        final_per_token_loss = per_token_loss


    if num_items_in_batch is not None:
        # Use the pre-calculated number if provided (more efficient)
        normalizer = num_items_in_batch.to(logits.device)
    else:
        # Otherwise, calculate it by counting non-ignored labels
        normalizer = (labels_flat != ignore_index).sum()

    # 5. Calculate the final loss with a robust normalization.
    # Sum all token losses (weighted or not) and divide by the number of valid tokens.
    total_loss = final_per_token_loss.sum()
    
    # CRITICAL: This check prevents the 0 / 0 = NaN issue.
    # If there are no valid tokens, the loss is 0.
    if normalizer > 0:
        final_loss = total_loss / normalizer
    else:
        # This handles the case where the entire batch has labels set to `ignore_index`.
        final_loss = torch.tensor(0.0, device=logits.device)

    return final_loss



class ModernBertForDiffusionLM(ModernBertPreTrainedModel,GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.tok_embeddings.weight"}
    config_class = NiuConfig
    def __init__(self, config,tokenizer):
        super().__init__(config)
        self.config = config
        self.config._attn_implementation = "flash_attention_2"
        self.tokenizer = tokenizer #debug usage
        self.model = ModernBertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=getattr(config, "decoder_bias", False))
        
        # 纠错头，预测哪些 token 需要被细化
        self.corrector_head = nn.Linear(config.hidden_size, 1)
        # 用于计算 corrector loss 的损失函数
        if getattr(config, "corrector_loss_weight", "auto") == "auto":
            # 默认使用自动计算的权重
            initial_lm_loss = math.log(config.vocab_size)
            initial_corrector_loss = math.log(2)
            self.corrector_loss_weight = initial_lm_loss / initial_corrector_loss
            print(f"Automatically determined corrector_loss_weight: {self.corrector_loss_weight:.2f}")
        else:
            # 允许手动覆盖
            self.corrector_loss_weight = config.corrector_loss_weight
            print(f"Manually set corrector_loss_weight: {self.corrector_loss_weight}")

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings
    
    @torch.compile(dynamic=True)
    def compiled_lm_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.lm_head(output)

    @torch.compile(dynamic=True)
    def compiled_corrector_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.corrector_head(output)

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
        trainint_stage = 1,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DiffusionOutput]:
 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()
        
        ignore_index = -100  # HF默认的忽略标签
        if labels is not None:
            labels = labels[..., 1:].contiguous()
            labels = F.pad(labels, (0, 1), value=ignore_index)
        
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

                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )
        else:
            assert False,"非fa2分支的行为没有经过检查，请不要使用"


        mask_token_id = self.config.mask_token_id
        
       
        
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
        
        if trainint_stage == 2 or labels is None:
            corrector_logits = (
                self.compiled_corrector_head(last_hidden_state)
                if self.config.reference_compile
                else self.corrector_head(last_hidden_state)
            )
        

        # 初始化总损失
        total_non_masked_lm_loss = torch.tensor(0.0, device=logits.device)
        total_corrector_loss = torch.tensor(0.0, device=logits.device)
        total_masked_lm_loss = torch.tensor(0.0, device=logits.device)



        # 其实如果不加权，整句话可以合在一起算，只是不好看loss了，感觉快不了多少，没必要
        if labels is not None:
            # --- 核心修改：物理分离数据并计算 Loss ---

            # 1. 定义哪些 token 最初是 MASK。这是我们分离数据的依据。
            # input_ids 此时是 unpadded 的，形状为 (total_tokens,)
            initial_mask_zone = (input_ids == mask_token_id)
            non_masked_zone = ~initial_mask_zone
            
            # 2. 计算 per_token_weights，它现在只为 MASK 区域的 token 创建
            weights_for_masked_tokens = None
            num_masked_tokens = initial_mask_zone.sum()
            if num_masked_tokens > 0:
                # 使用布尔索引直接筛选出 MASK 区域的数据
                masked_logits = logits[initial_mask_zone]
                masked_labels = labels[initial_mask_zone]
                


                total_masked_lm_loss = ForCausalLMLoss(
                    masked_logits, 
                    masked_labels, 
                    vocab_size=self.config.vocab_size,
                    per_token_weights=weights_for_masked_tokens
                )


                    
            num_non_masked_tokens = non_masked_zone.sum()
            # 这个 loss 只计算一次，因为非 MASK 区域的 token 在 refinement 中不会改变
            if num_non_masked_tokens > 0:
                # 使用布尔索引直接筛选出非 MASK 区域的数据
                non_masked_logits = logits[non_masked_zone]
                non_masked_labels = labels[non_masked_zone]

                # 此处不传入任何权重
                total_non_masked_lm_loss = ForCausalLMLoss(
                    non_masked_logits,
                    non_masked_labels,
                    vocab_size=self.config.vocab_size,
                    # 这里不传num_items_in_batch主要是因为这是全部的计数（包括mask和非mask），用mean reduction吧
                )

        if trainint_stage == 2:
            
            with torch.no_grad():
                # 这里是防止梯度影响lm_head。
                # 1. 获取 LM Head 的预测 (unpadded)
                predicted_ids = torch.argmax(logits, dim=-1) # Shape: (total_tokens,)
                corrector_target = (predicted_ids == labels).float()

            corrector_logits_for_loss = corrector_logits.squeeze(-1) # Shape: (total_tokens,)

            loss_mask = (labels != ignore_index)
            num_positives = (corrector_target * loss_mask).sum()
            num_valid_tokens = loss_mask.sum()

            if num_valid_tokens > 0:
                num_negatives = num_valid_tokens - num_positives
                if num_positives > 0:
                    pos_weight = num_negatives / num_positives
                else:
                    pos_weight = torch.tensor(1.0, device=logits.device)
            else:
                pos_weight = torch.tensor(1.0, device=logits.device)

            corrector_loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
            per_token_corrector_loss = corrector_loss_fct(corrector_logits_for_loss, corrector_target)
            masked_corrector_loss = per_token_corrector_loss * loss_mask

            if num_valid_tokens > 0:
                total_corrector_loss = masked_corrector_loss.sum() / num_valid_tokens * self.corrector_loss_weight 
            else:
                total_corrector_loss = torch.tensor(0.0, device=logits.device)
        
        
        # 后处理，还原去pad的序列
        if  labels is None and self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)
                corrector_logits = _pad_modernbert_output(inputs=corrector_logits, indices=indices, batch=batch_size, seqlen=seq_len)

        # --- 最终损失计算 ---
        total_loss = None
        if labels is not None:

            total_lm_loss = total_masked_lm_loss + total_non_masked_lm_loss
            if trainint_stage == 1:
                total_loss = total_lm_loss
            else:
                total_loss = total_lm_loss +  total_corrector_loss


        if not return_dict:
            # 为了简化，我们强制要求使用 return_dict
            assert False, "Please use return_dict=True"

        if labels is not None:
            return DiffusionOutput(
                loss=total_loss,
                lm_loss=total_lm_loss,
                masked_lm_loss=total_masked_lm_loss,
                non_masked_lm_loss=total_non_masked_lm_loss,
                corrector_loss=total_corrector_loss,
                logits=None,
                corrector_logits=None,
                hidden_states=None, # 返回最后一轮的 hidden_states
                attentions=None,
                current_mlm_prob=current_mlm_prob.mean() if current_mlm_prob is not None else None,
            )
        else:
            return DiffusionOutput(
                loss=total_loss,
                mlm_loss=total_lm_loss,
                corrector_loss=total_corrector_loss,
                logits=logits,
                corrector_logits=corrector_logits,
                hidden_states=outputs.hidden_states, # 返回最后一轮的 hidden_states
                attentions=outputs.attentions,
                current_mlm_prob=current_mlm_prob.mean() if current_mlm_prob is not None else None,
            )


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_diffusion_steps: int = 10,
        temperature_mlm: float = 1.0,
        use_token_change_classifier = True,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        debug: bool = False,
        tokenizer = None,
        decode_top_k_positions = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        自定义的扩散生成方法

        Args:
            input_ids: 输入的token ids，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            max_new_tokens: 要生成的新token数量 (L)
            num_diffusion_steps: 扩散迭代次数 (T)
            temperature_mlm: MLM采样的温度参数
            do_sample: 是否使用采样，False则使用贪心解码
            top_k: top-k采样参数
            top_p: top-p采样参数
            mask_token_id: mask token的id，如果为None则尝试自动获取
            debug: 是否启用调试模式，输出每步迭代的详细信息
            tokenizer: 用于将token id转换为文本的tokenizer（可选）

        Returns:
            生成的完整序列，形状为 (batch_size, original_seq_len + max_new_tokens)
        """
        batch_size, original_seq_len = input_ids.shape
        device = input_ids.device

        # 1. 在输入后填充L个mask token
        mask_tokens = torch.full((batch_size, max_new_tokens), mask_token_id,
                                dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([input_ids, mask_tokens], dim=1)

        # 扩展attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        extended_attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, max_new_tokens), dtype=torch.bool, device=device)
        ], dim=1)

        # 2. 迭代T次
        current_sequence = extended_input_ids.clone()

        if debug:
            logger.info("=" * 80)
            logger.info("🚀 开始扩散生成过程")
            logger.info(f"📊 参数设置: max_new_tokens={max_new_tokens}, num_diffusion_steps={num_diffusion_steps}")
            logger.info(f"🎯 采样设置: do_sample={do_sample}, temperature_mlm={temperature_mlm}")
            if top_k is not None:
                logger.info(f"🔝 Top-k采样: k={top_k}")
            if top_p is not None:
                logger.info(f"🎲 Top-p采样: p={top_p}")

            # 显示初始序列
            for batch_idx in range(batch_size):
                initial_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 初始序列: {initial_text}")
            logger.info("=" * 80)

        for step in range(num_diffusion_steps):
            if debug:
                logger.info(f"\n🔄 === 步骤 {step + 1}/{num_diffusion_steps} ===")

            # 保存当前序列用于比较
            prev_sequence = current_sequence.clone()

            outputs = self.forward(
                input_ids=current_sequence,
                attention_mask=extended_attention_mask,
                return_dict=True,
                causal=True if step == 0 else False,
            )

            mlm_logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # 只处理需要生成的部分（后L个位置）
            generation_start_idx = original_seq_len
            mlm_logits_gen = mlm_logits[:, generation_start_idx:, :]  # (batch_size, max_new_tokens, vocab_size)




            # 3. 从MLM logits中生成候选token
            if do_sample:
                # 采样生成
                mlm_logits_gen = mlm_logits_gen / temperature_mlm

                if top_k is not None:
                    # Top-k采样
                    top_k_logits, top_k_indices = torch.topk(mlm_logits_gen, k=min(top_k, mlm_logits_gen.size(-1)))
                    mlm_logits_gen = torch.full_like(mlm_logits_gen, float('-inf'))
                    mlm_logits_gen.scatter_(-1, top_k_indices, top_k_logits)

                if top_p is not None:
                    # Top-p采样
                    sorted_logits, sorted_indices = torch.sort(mlm_logits_gen, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    mlm_logits_gen = mlm_logits_gen.masked_fill(indices_to_remove, float('-inf'))

                # 多项式采样
                probs = torch.softmax(mlm_logits_gen, dim=-1)

                candidate_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, max_new_tokens)

            else:
                # 贪心解码
                candidate_tokens = torch.argmax(mlm_logits_gen, dim=-1)  # (batch_size, max_new_tokens)



            if decode_top_k_positions is not None:
                if debug:
                    logger.info(f"  [策略] Top-K Positions (Non-Replacing): 在剩余MASK中解码置信度最高的 {decode_top_k_positions} 个位置。")

                # 1. 识别哪些位置当前仍然是 MASK
                current_gen_part = current_sequence[:, generation_start_idx:]
                is_mask_mask = (current_gen_part == mask_token_id)

                # 如果已经没有 MASK token，可以提前跳过此步骤
                if not is_mask_mask.any():
                    if debug: logger.info("  [信息] 没有剩余的MASK token，跳过更新。")
                    break

                # 2. 计算所有生成位置的置信度
                probs = torch.softmax(mlm_logits_gen, dim=-1)
                confidence_scores, _ = torch.max(probs, dim=-1)  # Shape: (batch_size, max_new_tokens)

                # 3. 【关键步骤】将已经被填充的位置的置信度设为-1，使其在topk中不被选中
                # torch.where(condition, value_if_true, value_if_false)
                masked_confidence_scores = torch.where(
                    is_mask_mask,          # 条件：是MASK吗？
                    confidence_scores,     # 如果是，保留原始置信度
                    -1.0                   # 如果不是，设为-1，使其失去竞争力
                )

                # 4. 在被屏蔽后的置信度中找到 top k。
                # k 的值不能超过总生成长度
                k = min(decode_top_k_positions, max_new_tokens)
                # topk 会自动从非-1的分数中选择最高的k个
                _, top_k_indices = torch.topk(masked_confidence_scores, k=k, dim=1)

                # 5. 创建更新掩码，只在选中的 top-k 位置为 True
                final_update_mask = torch.zeros_like(confidence_scores, dtype=torch.bool, device=device)
                final_update_mask.scatter_(1, top_k_indices, True)

                # 6. 【安全保障】最终的掩码必须确保只更新原先是MASK的位置。
                # 这一步可以防止在 MASK 数量少于 k 时，topk 选到 -1 的位置。
                final_update_mask = final_update_mask & is_mask_mask

                # 7. 根据最终掩码更新序列
                new_tokens = torch.where(
                    final_update_mask,
                    candidate_tokens,
                    current_gen_part
                )
                current_sequence[:, generation_start_idx:] = new_tokens

            elif use_token_change_classifier:
                # token change classifier only decice change clean token， so we can update all mask。
                if step == 0:
                    if debug:
                        logger.info("  [策略] Token-Change Classifier (Step 0): 强制全部更新以初始化序列。")
                    # 在第一步，我们不使用分类器，而是直接更新所有位置。
                    # 这有助于从纯MASK状态快速生成一个合理的初始草稿。
                    final_update_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
                else:

                    if debug:
                        logger.info("  [策略] Token-Change Classifier: 使用分类器决定更新位置。")
                    token_change_logits = outputs.token_change_logits
                    change_logits_gen = token_change_logits[:, generation_start_idx:, :]

                    change_decisions = change_logits_gen.sigmoid().squeeze(-1) > 0.5
                    sentences_with_no_changes = ~torch.any(change_decisions, dim=1)
                    final_update_mask = change_decisions | sentences_with_no_changes.unsqueeze(1)

                new_tokens = torch.where(
                    final_update_mask,
                    candidate_tokens,
                    current_sequence[:, generation_start_idx:]
                )
                current_sequence[:, generation_start_idx:] = new_tokens

            # 模式三：无分类器，全部更新
            else:
                if debug:
                    logger.info("  [策略] Update All: 更新所有生成位置。")
                current_sequence[:, generation_start_idx:] = candidate_tokens
                # 为调试创建一个全True的掩码
                final_update_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)


            if debug:
                mask_positions = current_sequence[:, generation_start_idx:] == mask_token_id
                self._debug_step_changes(
                    step + 1,
                    prev_sequence,
                    current_sequence,
                    candidate_tokens,
                    final_update_mask,

                    mask_positions,
                    generation_start_idx,
                    batch_size,
                    tokenizer,
                    mask_token_id
                )

            import pdb; pdb.set_trace()
        if debug:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 扩散生成完成!")
            for batch_idx in range(batch_size):
                final_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 最终序列: {final_text}")
            logger.info("=" * 80)

        
        return current_sequence

    def _debug_step_changes(
        self,
        step: int,
        prev_sequence: torch.Tensor,
        current_sequence: torch.Tensor,
        candidate_tokens: torch.Tensor,
        change_decisions: torch.Tensor,
        mask_positions: torch.Tensor,
        generation_start_idx: int,
        batch_size: int,
        tokenizer,
        mask_token_id: int
    ):
        """
        输出每步迭代的详细变化信息
        """
        # 直接通过前后序列对比找出变化
        prev_gen_tokens = prev_sequence[:, generation_start_idx:]
        curr_gen_tokens = current_sequence[:, generation_start_idx:]
        actual_changes = prev_gen_tokens != curr_gen_tokens  # (batch_size, max_new_tokens)

        total_changes = actual_changes.sum().item()
        total_masks = mask_positions.sum().item()

        logger.info(f"📈 统计信息:")
        logger.info(f"   • 剩余MASK位置: {total_masks}")
        logger.info(f"   • 实际发生的变化: {total_changes}")

        # 对每个batch进行详细分析
        for batch_idx in range(batch_size):
            if batch_size > 1:
                logger.info(f"\n🔍 === Batch {batch_idx} 详细分析 ===")

            prev_tokens = prev_gen_tokens[batch_idx]
            curr_tokens = curr_gen_tokens[batch_idx]
            candidates = candidate_tokens[batch_idx]
            changes = actual_changes[batch_idx]
            masks = mask_positions[batch_idx]

            # 找出所有发生变化的位置
            changed_positions = torch.where(changes)[0].tolist()
            mask_positions_list = torch.where(masks)[0].tolist()

            if changed_positions:
                logger.info(f"✅ 发生变化的位置 ({len(changed_positions)}个):")
                for pos in changed_positions:
                    prev_token = prev_tokens[pos].item()
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()

                    prev_text = self._token_to_text(prev_token, tokenizer, mask_token_id)
                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)

                    logger.info(f"   位置 {pos:2d}: {prev_text} → {curr_text} (候选: {candidate_text})")
            else:
                logger.info("❌ 本步骤没有发生任何变化")

            # 显示候选token与实际选择不同的位置
            candidate_different = candidates != curr_tokens
            different_but_unchanged = candidate_different & ~changes
            different_positions = torch.where(different_but_unchanged)[0].tolist()

            if different_positions:

                logger.info(f"🤔 候选与实际不同但未变化的位置 ({len(different_positions)}个):")
                for pos in different_positions:
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()

                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)

                    logger.info(f"   位置 {pos:2d}: 保持 {curr_text} (候选: {candidate_text})")
                logger.info(change_decisions)
            # 显示剩余的MASK位置
            remaining_masks = torch.where(curr_tokens == mask_token_id)[0].tolist()
            if remaining_masks:
                logger.info(f"🎭 剩余MASK位置 ({len(remaining_masks)}个): {remaining_masks}")
            else:
                logger.info("🎊 所有MASK已被替换!")

            # 显示当前生成部分的完整文本
            current_gen_text = self._tokens_to_text(curr_tokens, tokenizer, mask_token_id)
            logger.info(f"📄 当前生成部分: {current_gen_text}")

    def _tokens_to_text(self, tokens: torch.Tensor, tokenizer, mask_token_id: int) -> str:
        """
        将token序列转换为可读文本
        """
        if tokenizer is None:
            # 如果没有tokenizer，直接显示token id
            token_strs = []
            for token_id in tokens.tolist():
                if token_id == mask_token_id:
                    token_strs.append("[MASK]")
                else:
                    token_strs.append(f"<{token_id}>")
            return " ".join(token_strs)
        else:
            # 使用tokenizer解码
            try:
                # 将MASK token替换为特殊标记以便正确显示
                display_tokens = tokens.clone()
                display_tokens[tokens == mask_token_id] = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else mask_token_id
                text = tokenizer.decode(display_tokens, skip_special_tokens=False)
                return text
            except Exception as e:
                logger.warning(f"Tokenizer解码失败: {e}")
                return self._tokens_to_text(tokens, None, mask_token_id)

    def _token_to_text(self, token_id: int, tokenizer, mask_token_id: int) -> str:
        """
        将单个token id转换为可读文本
        """
        if token_id == mask_token_id:
            return "[MASK]"

        if tokenizer is None:
            return f"<{token_id}>"
        else:
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                return f"'{text}'"
            except Exception as e:
                return f"<{token_id}>"
