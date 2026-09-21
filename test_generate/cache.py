# transformers cache
import torch
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Union
from transformers.configuration_utils import PreTrainedConfig

# --- 基础抽象层 ---

class CacheLayerMixin(ABC):
    """单层缓存的基础抽象类"""
    is_compileable = False

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None: ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def reset(self) -> None:
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """用于 Beam Search 的缓存重排"""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

# --- 动态层实现 ---

class DynamicLayer(CacheLayerMixin):
    """标准的动态增长缓存层"""
    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor):
        query_length = cache_position.shape[0]
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.keys.shape[-2] if self.is_initialized and self.keys.numel() > 0 else 0

    def get_max_cache_shape(self) -> int:
        return -1

class DynamicSlidingWindowLayer(DynamicLayer):
    """滑动窗口动态缓存层 (如 Mistral/Qwen 等模型使用)"""
    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        
        # 只保留滑动窗口大小的缓存
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        return full_key_states, full_value_states

    def get_seq_length(self) -> int:
        return self.cumulative_length

# --- Cache 容器 ---

class Cache:
    """管理所有层 CacheLayer 的容器"""
    def __init__(self, layers: List[CacheLayerMixin] = None, layer_class_to_replicate=None):
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # 如果层还没创建，则根据模板自动增加层
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
        
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def __len__(self):
        return len(self.layers)

class DynamicCache(Cache):
    """用户直接使用的动态缓存类"""
    def __init__(self, config: Optional[PreTrainedConfig] = None):
        layers = []
        if config is not None:
            # 根据模型配置自动识别是否需要滑动窗口
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None)
            num_layers = decoder_config.num_hidden_layers
            
            for _ in range(num_layers):
                if sliding_window is not None:
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())
        
        if len(layers) == 0:
            super().__init__(layer_class_to_replicate=DynamicLayer)
        else:
            super().__init__(layers=layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values