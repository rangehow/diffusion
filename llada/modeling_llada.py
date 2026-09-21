# modeling_llada.py
from __future__ import annotations
import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from dataclasses import dataclass, fields
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache
from transformers.utils import is_flash_attn_2_available
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

# === Flash Attention and Unpadded RoPE Imports Start ===
if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    FlashRotaryEmbedding = object
    flash_attn_varlen_qkvpacked_func = None
    apply_rotary = None
# === Flash Attention and Unpadded RoPE Imports End ===

# NOTE: This version is simplified for a fixed block_type='llama' and block_group_size=1.
# Unused classes like LLaDASequentialBlock and LLaDABlockGroup have been removed.
# The original class names (LLaDABlock, LLaDALlamaBlock) are preserved for state_dict compatibility.

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDALlamaBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]

log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.
    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


# === Unpadded RoPE for Flash Attention Start ===
class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(qk, cos, sin, seqlen_offsets=0, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=False, inplace=True)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
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
        return do, None, None, None, None


def apply_rotary_unpadded(qkv, cos, sin, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class LLaDAUnpaddedRotaryEmbedding(FlashRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen
        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: Optional[int] = None):
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        return apply_rotary_unpadded(qkv, self._cos_cached, self._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


# === Unpadded RoPE for Flash Attention End ===


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k
        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))


class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0
        self._activation_checkpoint_fn = None
        self.dropout = Dropout(config.residual_dropout)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True
        self.padded_rotary_emb = None
        self.unpadded_rotary_emb = None
        if self.config.rope:
            # Always create the standard rotary embedding for the padded path
            self.padded_rotary_emb = RotaryEmbedding(config, self.__cache)
            # If Flash Attention 2 is available, also create the unpadded version
            if is_flash_attn_2_available():
                self.unpadded_rotary_emb = LLaDAUnpaddedRotaryEmbedding(
                    dim=config.d_model // config.n_heads,
                    base=config.rope_theta,
                    max_seqlen=config.max_sequence_length,
                )
        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T_q, C = q.size()
        T_k = k.size(1)
        
        q = q.view(B, T_q, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # Note: T_k might be different from T_q in the padded case
        k = k.view(B, T_k, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T_k, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = (k, v) if use_cache else None
        
        query_len, key_len = q.shape[-2], k.shape[-2]
        if self.config.rope:
            q, k = self.padded_rotary_emb(q, k)
        
        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], k.dtype
            )
            
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
        att = att.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        # Simplified: since block_type is always "llama", we directly return LLaDALlamaBlock.
        return LLaDALlamaBlock(layer_id, config, cache)


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()  # This correctly initializes attn_out and ff_out as 'out_module'
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        
        # Correctly initialize all input projections with type_of_module
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # --- Unpadded Flash Attention path ---
        if self.config.flash_attention and cu_seqlens is not None:
            assert layer_past is None and not use_cache and attention_bias is None, "Unpadded path does not support caching or attention bias."
            x_normed = self.attn_norm(x)
            
            head_dim = self.config.d_model // self.config.n_heads
            
            # 1. 计算 Q, K, V
            q_unpad = self.q_proj(x_normed)
            k_unpad = self.k_proj(x_normed)
            v_unpad = self.v_proj(x_normed)
            
            # 2. Reshape Q, K, V 以便处理 heads
            q_reshaped = q_unpad.view(-1, self.config.n_heads, head_dim)
            k_reshaped = k_unpad.view(-1, self.config.effective_n_kv_heads, head_dim)
            v_reshaped = v_unpad.view(-1, self.config.effective_n_kv_heads, head_dim)
            
            # 3. 如果是 GQA，现在沿着正确的 head 维度进行扩展
            if self.config.n_heads != self.config.effective_n_kv_heads:
                num_reps = self.config.n_heads // self.config.effective_n_kv_heads
                k_reshaped = k_reshaped.repeat_interleave(num_reps, dim=1)
                v_reshaped = v_reshaped.repeat_interleave(num_reps, dim=1)

            # 4. 将 Q, K, V 堆叠成 flash-attn 期望的格式
            qkv = torch.stack([q_reshaped, k_reshaped, v_reshaped], dim=1)

            # 5. 应用 Rotary Embedding
            qkv_rotated = self.unpadded_rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            qkv_origin_dtype = qkv_rotated.dtype
            qkv_rotated = qkv_rotated.to(torch.bfloat16)
            attn_output = flash_attn_varlen_qkvpacked_func(
                qkv_rotated,
                cu_seqlens,
                max_seqlen,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                causal=False,
            )
            attn_output = attn_output.to(qkv_origin_dtype)
            attn_output = attn_output.view(-1, self.config.d_model)
            att = self.attn_out(attn_output)
            cache = None
        
        # --- Padded path (original logic) ---
        else:
            x_normed = self.attn_norm(x)
            q = self.q_proj(x_normed)
            k = self.k_proj(x_normed)
            v = self.v_proj(x_normed)

            if self._activation_checkpoint_fn is not None:
                att, cache = self._activation_checkpoint_fn(
                    self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # --- Common MLP logic for both paths ---
        x = x + self.dropout(att)
        og_x = x
        
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        # x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        # if self._activation_checkpoint_fn is not None:
        #     x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        # else:
        #     x = self.act(x)
        # x = x * x_up # new add

        gate = self.ff_proj(x)
        up = self.up_proj(x)

        # The self.act object is likely a SwiGLU module, which is incorrect here.
        # The standard SwiGLU activation uses F.silu on the gate.
        # Even if self.act is configured to be SiLU, this pattern is more explicit and correct.
        activated_gate = F.silu(gate)

        x = activated_gate * up

        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    scores: torch.FloatTensor


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")
        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn("Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning)
        
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        # The config option block_group_size is ignored as it is fixed to 1.
        # The check `n_layers % block_group_size == 0` is no longer needed.
        
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                    padding_idx=config.pad_token_id,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )
        
        # Simplified block creation, no grouping logic needed.
        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None


    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        # Simplified: always iterate over blocks, no block groups.
        for block in self.transformer.blocks:
            block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        init_weights(self.config, self.transformer.wte, std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0, type_of_module=ModuleType.emb)
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)
        self.transformer.ln_f.reset_parameters()
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)
        
        # Simplified: always iterate over blocks, no block groups.
        for block in self.transformer.blocks:
            block.reset_parameters()


    def get_bidirectional_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (bidirectional_bias := self.__cache.get("bidirectional_attention_bias")) is not None and bidirectional_bias.shape[-1] >= seq_len:
            if bidirectional_bias.device != device:
                bidirectional_bias = bidirectional_bias.to(device)
                self.__cache["bidirectional_attention_bias"] = bidirectional_bias
            return bidirectional_bias
        with torch.autocast(device.type, enabled=False):
            bidirectional_bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float)
        self.__cache["bidirectional_attention_bias"] = bidirectional_bias
        return bidirectional_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> LLaDAOutput:
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        if cu_seqlens is not None:
            assert (past_key_values is None and not use_cache), "The kvcache is not supported for the unpadded Flash Attention path."
        elif past_key_values is not None:
            assert len(past_key_values) == self.config.n_layers
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        if input_embeddings is not None:
            seq_len = input_embeddings.size(1) if input_embeddings.dim() == 3 else None
        else:
            seq_len = input_ids.size(1) if input_ids.dim() == 2 else None
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)
        if not (self.config.alibi or self.config.rope) and seq_len is not None:
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x
        x = self.transformer.emb_drop(x)
        if cu_seqlens is None:
            if attention_mask is not None and 0.0 in attention_mask:
                batch_size = attention_mask.shape[0]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            else:
                attention_mask = None
            if (attention_bias is not None or attention_mask is not None or self.config.alibi or past_key_values is not None):
                
                if attention_bias is None:
                    attention_bias = self.get_bidirectional_attention_bias(past_length + seq_len, x.device)
                elif attention_bias.dtype in (torch.int8, torch.bool):
                    attention_bias = attention_bias.to(dtype=torch.float)
                    attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)
                mask_len = seq_len if seq_len is not None else (past_key_values[0][0].shape[-2] + 1 if past_key_values else 0)
                if past_key_values is not None:
                    # During generation, the effective sequence length includes past keys
                    mask_len = past_length + seq_len
                attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
                if attention_mask is not None:
                    attention_bias = attention_bias + attention_mask
                    ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        else:
            attention_bias = None
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []
        common_block_kwargs = {
            "attention_bias": attention_bias,
            "use_cache": use_cache,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        
        # Simplified forward loop, no block groups.
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            should_checkpoint = (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
            )
            if should_checkpoint:
                x, cache = self._activation_checkpoint_fn(block, x, layer_past=layer_past, **common_block_kwargs)
            else:
                x, cache = block(x, layer_past=layer_past, **common_block_kwargs)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        return LLaDAOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)
    model_config = ModelConfig(**kwargs)
    return model_config


from transformers.modeling_outputs import ModelOutput




def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    per_token_weights: Optional[torch.Tensor],
    num_items_in_batch: torch.Tensor, # 必须传入，不再是 Optional
    ignore_index: int = -100,
    **kwargs,
):
    # 1. 统一预处理：展平维度，对齐设备
    logits = logits.float().view(-1, vocab_size)
    labels = labels.view(-1).to(logits.device)


    per_token_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    weights = per_token_weights.reshape(-1).to(per_token_loss.device)
    total_loss = (per_token_loss * weights).sum()

    # 3. 归一化：直接除以 num_items_in_batch
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(total_loss.device)
        
    return total_loss / num_items_in_batch




def _unpad_llada_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
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
    unpadded_labels = labels.flatten()[indices] if labels is not None else None
    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_labels


def _pad_llada_output(
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
class CausalLMOutputWithPastAndMLMProb(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    current_mlm_prob: Optional[torch.FloatTensor] = None

class LLaDAModelLM(PreTrainedModel):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDALlamaBlock"]
    _supports_flash_attn_2 = True
    # NOTE (transformers >= 5.3 compat): When weight_tying=True, there is
    # NO separate ff_out module — forward() uses F.linear(x, wte.weight)
    # directly. So there is no second parameter to "tie". We set this to
    # None and let get_expanded_tied_weights_keys() return {} because
    # tie_word_embeddings=False in config (we handle tying ourselves).
    _tied_weights_keys = None
    
    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)
        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model
        # NOTE (transformers >= 5.3 compat): post_init() sets instance attributes
        # that _finalize_model_loading / mark_tied_weights_as_initialized expect:
        #   - self.all_tied_weights_keys  (dict)
        #   - self._keep_in_fp32_modules (set)
        #   - self._no_split_modules     (set)
        # It also calls init_weights() + _backward_compatibility_gradient_checkpointing().
        # All standard HF models call this at the end of __init__.
        self.post_init()

    def _apply_subs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        参考 _subs_parameterization 实现：
        xt (即 input_ids) 是加噪后的输入。
        
        逻辑分解：
        1. 全局规则：预测结果 logits 中，Mask Token 的概率必须为 0 (Logit = -inf)。
        2. Unmasked 规则：对于 input_ids 中本身就不是 Mask 的位置，强制 logits 输出等于 input_ids 的值。
           这样做的效果是：模型对已知 Token 的预测概率为 1，产生的 Loss 为 0。
        """
        mask_token_id = self.config.mask_token_id
        neg_inf = float('-inf')
        
        # === 逻辑 1: 禁止预测 Mask Token ===
        # 对应参考代码: logits[:, :, self.mask_index] += self.neg_infinity
        # 无论在什么位置，模型都不应该预测出 [MASK] 这个词
        logits[..., mask_token_id] = neg_inf

        # === 逻辑 2: 强制 Unmasked 位置的输出 ===
        # 对应参考代码: 
        # unmasked_indices = (xt != self.mask_index)
        # logits[unmasked_indices] = self.neg_infinity
        # logits[unmasked_indices, xt[unmasked_indices]] = 0
        
        if input_ids is None:
            return logits

        # 找出 xt (input_ids) 中所有 "不是 Mask" 的位置
        xt_is_unmasked = (input_ids != mask_token_id)

        # (可选) 如果不是 repad 模式，需要排除 Padding，避免把 Padding 这种无意义 token 强制设为 1
        if attention_mask is not None and attention_mask.dim() == input_ids.dim():
             xt_is_unmasked = xt_is_unmasked & attention_mask.bool()

        if xt_is_unmasked.any():
            # 2.1 先把这些位置的所有 vocab logits 设为 -inf (清空概率分布)
            logits[xt_is_unmasked] = neg_inf

            # 2.2 获取这些位置对应的真实 Token ID (即 xt 的值)
            target_xt_tokens = input_ids[xt_is_unmasked]
            
            # 2.3 构造索引，将对应的 Token Logit 设为 0 (概率设为 1)
            # torch.where 返回 (indices_dim0, indices_dim1, ...)
            indices = torch.where(xt_is_unmasked)
            
            # 组合索引: (位置索引..., 目标Token索引)
            full_indices = indices + (target_xt_tokens,)
            
            logits[full_indices] = 0.0
        
        return logits


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current_mlm_prob=None,
        **kwargs
    ):
        

        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        repad = False

        original_batch_size, original_seq_len, unpad_indices = None, None, None
        cu_seqlens, max_seqlen = None, None
        if self.model.config.flash_attention and is_flash_attn_2_available() and past_key_values is None:
            repad = True
            if inputs_embeds is not None:
                original_batch_size, original_seq_len = inputs_embeds.shape[:2]
                device = inputs_embeds.device
            else:
                original_batch_size, original_seq_len = input_ids.shape[:2]
                device = input_ids.device
            if attention_mask is None:
                attention_mask = torch.ones((original_batch_size, original_seq_len), device=device, dtype=torch.bool)
            if inputs_embeds is None:

                input_ids, unpad_indices, cu_seqlens, max_seqlen, unpadded_labels = _unpad_llada_input(
                    inputs=input_ids, attention_mask=attention_mask, labels=labels
                )
            else:

                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen, unpadded_labels = _unpad_llada_input(
                    inputs=inputs_embeds, attention_mask=attention_mask, labels=labels
                )
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:

            loss_labels = unpadded_labels if repad else labels

            curr_input_ids = input_ids
            curr_att_mask = attention_mask

            # 特殊情况处理：
            # 如果使用了 inputs_embeds 且开启了 repad，原始 input_ids 还是 (B, L)，
            # 我们需要手动展平它以匹配 logits 的形状
            if repad and inputs_embeds is not None and curr_input_ids is not None and curr_input_ids.dim() > 1:
                curr_input_ids = curr_input_ids.flatten()[unpad_indices]
                curr_att_mask = None # repad 后不再需要 attention_mask 来过滤 padding
            
            # 如果是 repad 模式，input_ids 已经是展平且去除了 padding 的，不需要传入 attention_mask
            if repad:
                curr_att_mask = None

            
            logits = self._apply_subs(logits, curr_input_ids, curr_att_mask)
            
            # Check if MDM mode (discrete timestep with focal loss)
            discrete_t = kwargs.pop('discrete_t', None)
            loss_mask = kwargs.pop('loss_mask', None)
            mdm_config = kwargs.pop('mdm_config', None)
            
            if discrete_t is not None and loss_mask is not None and mdm_config is not None:
                # === MDM Loss (Ye et al. 2024) ===
                # 1. Compute per-token CE loss
                flat_logits = logits.float().view(-1, self.config.vocab_size)
                flat_labels = loss_labels.view(-1).to(flat_logits.device)
                per_token_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')
                
                # 2. Apply loss_mask: only [MASK] positions have loss
                if repad:
                    flat_loss_mask = loss_mask.reshape(-1)[unpad_indices].to(per_token_loss.device)
                else:
                    flat_loss_mask = loss_mask.reshape(-1).to(per_token_loss.device)
                per_token_loss = per_token_loss * flat_loss_mask.float()
                
                # 3. Token reweighting (focal loss): alpha * (1 - exp(-loss))^gamma * loss
                if mdm_config.get('token_reweighting', False):
                    alpha = mdm_config.get('focal_alpha', 0.25)
                    gamma = mdm_config.get('focal_gamma', 2.0)
                    per_token_loss = alpha * (1 - torch.exp(-per_token_loss)) ** gamma * per_token_loss
                
                # 4. Time reweighting
                T = mdm_config.get('diffusion_steps', 20)
                t_vals = discrete_t.to(per_token_loss.device)  # [B]
                time_rw = mdm_config.get('time_reweighting', 'linear')
                if time_rw == 'linear':
                    weight = (T - t_vals).float()  # [B]
                elif time_rw == 'original':
                    weight = 1.0 / (t_vals + 1).float()
                else:
                    weight = torch.ones_like(t_vals, dtype=torch.float32)
                
                seq_len = original_seq_len if repad else logits.shape[1]
                weight_expanded = weight.unsqueeze(1).expand(-1, seq_len)  # [B, L]
                if repad:
                    flat_weights = weight_expanded.reshape(-1)[unpad_indices].to(per_token_loss.device)
                else:
                    flat_weights = weight_expanded.reshape(-1).to(per_token_loss.device)
                
                per_token_loss = per_token_loss * flat_weights
                
                # 5. Normalize by number of masked tokens
                n_masked = flat_loss_mask.sum().clamp(min=1)
                loss = per_token_loss.sum() / n_masked
            else:
                # === Standard MDLM Loss (1/t DAUM weighting) ===
                per_token_mlm_weights = None
                if current_mlm_prob is not None:
                    t = current_mlm_prob.to(logits.device)
                    weights_per_sentence = 1.0 / (t + 1e-8)
                    
                    seq_len = original_seq_len if repad else logits.shape[1]
                    
                    padded_weights = weights_per_sentence.unsqueeze(1).expand(-1, seq_len)
                    
                    if repad:
                        per_token_mlm_weights = padded_weights.flatten()[unpad_indices]
                    else:
                        per_token_mlm_weights = padded_weights
                
                loss = ForMaskedLMLoss(
                    logits, 
                    loss_labels, 
                    vocab_size=self.config.vocab_size,
                    per_token_weights=per_token_mlm_weights,
                    **kwargs
                )

            
        if repad:
            logits = _pad_llada_output(inputs=logits, indices=unpad_indices, batch=original_batch_size, seqlen=original_seq_len)
            if hidden_states is not None:
                padded_hidden_states = []
                for hs in hidden_states:
                    padded_hidden_states.append(_pad_llada_output(inputs=hs, indices=unpad_indices, batch=original_batch_size, seqlen=original_seq_len))
                hidden_states = tuple(padded_hidden_states)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndMLMProb(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
            current_mlm_prob=current_mlm_prob.mean() if current_mlm_prob is not None else None,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self, **kwargs):
        # Weight tying is handled manually in forward() via:
        #   F.linear(x, self.transformer.wte.weight, None)
        # There is no separate ff_out module when weight_tying=True,
        # so we do NOT need to create any alias here.
        # For weight_tying=False, ff_out is a standalone Linear — no tying needed.
        pass

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)