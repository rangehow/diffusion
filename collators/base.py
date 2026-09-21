# collators/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

class BaseCollator(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = "text",
        add_special_tokens: bool = True,
        pad_to_max_length: bool = False,  # <--- 新增开关
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.add_special_tokens = add_special_tokens
        self.pad_to_max_length = pad_to_max_length # <--- 保存状态

        # 必需 token 校验
        for name in ["pad_token_id", "bos_token_id", "eos_token_id"]:
            value = getattr(tokenizer, name)
            if value is None:
                # 有些模型 pad_token 默认为 None，这里做一个容错，如果为 None 则指向 eos
                if name == "pad_token_id" and tokenizer.eos_token_id is not None:
                    value = tokenizer.eos_token_id
                else:
                    raise ValueError(f"Tokenizer 缺少必需的 {name}")
            setattr(self, name, value)

        # 可选 token
        for name in ["mask_token_id", "cls_token_id", "sep_token_id"]:
            setattr(self, name, getattr(tokenizer, name))

    def _pad_batch(self, sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        """
        核心填充逻辑：根据 pad_to_max_length 决定是动态填充还是固定填充
        """
        # 1. 先进行动态填充 (Pad 到当前 Batch 最长)
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        # 2. 如果需要固定长度，再 Pad 到 max_length
        if self.pad_to_max_length:
            curr_len = padded.size(1)
            if curr_len < self.max_length:
                # F.pad 参数格式: (left, right, top, bottom, ...)
                # 我们只需要在 right 方向填充
                pad_len = self.max_length - curr_len
                padded = F.pad(padded, (0, pad_len), value=padding_value)
            # 注意：tokenize_single 已经做了 truncation，所以 curr_len 不会大于 max_length
            
        return padded

    def tokenize_single(self, text: str) -> List[int]:

        if self.add_special_tokens:

            text_part = text
            if self.tokenizer.bos_token:
                text_part = self.tokenizer.bos_token + text_part
            if self.tokenizer.eos_token:
                text_part = text_part + self.tokenizer.eos_token
        
        return self.tokenizer.encode(
            text_part,
            add_special_tokens=False, # 上面手动加了，这里关掉
            truncation=True,
            max_length=self.max_length,
        )

    def prepare_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        raw_sequences: List[torch.Tensor] = []
        for ex in examples:
            text = ex[self.text_key]
            ids = self.tokenize_single(text)
            raw_sequences.append(torch.tensor(ids, dtype=torch.long))

        # 使用封装好的填充方法
        input_ids = self._pad_batch(raw_sequences, padding_value=self.pad_token_id)
        
        # Mask 也可以基于 input_ids 自动生成，不需要再手动 pad 一次
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return input_ids, attention_mask, raw_sequences

    @abstractmethod
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ...