# collators/__init__.py
from typing import List, Dict, Any, Tuple
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from .base import BaseCollator
from .mixins import MaskedLanguageModelingMixin
from .reasoning import ReasoningCARDCollator, ReasoningARCollator, ReasoningMDLMCollator, ReasoningMDMCollator


class NTPCollator(BaseCollator):
    def __init__(self, tokenizer, max_length: int = 512, text_key: str = "text", pad_to_max_length: bool = False):
        super().__init__(tokenizer, max_length, text_key, add_special_tokens=True, pad_to_max_length=pad_to_max_length)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, raw_seqs = self.prepare_batch(examples)
        
        # 这里的 labels 必须和 input_ids 长度严格一致
        # 使用 self._pad_batch 替代 mixin 里的简单 pad_sequence
        labels = self._pad_batch(raw_seqs, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class CausalMLMCollator(BaseCollator, MaskedLanguageModelingMixin):
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        text_key: str = "text",
        start_prob: float = 0.0,
        end_prob: float = 1.0,
        tail_bias_factor: float = 1.5,
        use_daum: bool = False,
        pad_to_max_length: bool = False,
        is_eval: bool = False,  # <--- 新增：控制是否为评估模式
        **kwargs,
    ):
        super().__init__(tokenizer, max_length, text_key, add_special_tokens=True,pad_to_max_length=pad_to_max_length)
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.tail_bias_factor = max(1.0, tail_bias_factor)
        self.use_daum = use_daum
        self.is_eval = is_eval  # 保存状态

    def _tail_bias_indices(self, seq_len: int, prob: float) -> List[int]:
        # ... 保持不变 ...
        maskable = list(range(1, seq_len - 1))
        if not maskable:
            return []
        num = max(1, int(len(maskable) * prob))
        pool = min(int(num * self.tail_bias_factor), len(maskable))
        pool = max(pool, num)
        candidates = maskable[-pool:]
        return random.sample(candidates, num)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare_batch 已经返回了统一 padding 的 input_ids，
        # 但 CausalMLM 逻辑比较特殊，需要操作 raw_seqs 然后重新 pad。
        
        # 为了避免 prepare_batch 做的 padding 浪费，我们可以只取 raw_seqs
        # 但 attention_mask 稍后需要重新计算
        _, _, raw_seqs = self.prepare_batch(examples) 
        B = len(raw_seqs)

        # ==========================================
        # 分支 1: Eval 模式 (NTP, Clean Context)
        # ==========================================
        if self.is_eval:
            # 直接 Pad 原始序列，不做任何 Mask
            input_ids = self._pad_batch(raw_seqs, padding_value=self.pad_token_id)
            labels = self._pad_batch(raw_seqs, padding_value=-100)
            
            attention_mask = (input_ids != self.pad_token_id).long()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "current_mlm_prob": None,
                "causal": True,
                "use_daum": False,
                "is_eval": True # 可选：传给模型通过
            }

        # ==========================================
        # 分支 2: Train 模式 (离散扩散, Masking)
        # ==========================================
        
        # ... 这里的逻辑保持你原代码不变 ...
        p_min = max(1e-3, min(self.start_prob, self.end_prob))
        p_max = min(1.0 - 1e-3, max(self.start_prob, self.end_prob))
        t = self.sample_antithetic_probs(B, low=p_min, high=p_max)
        batch_probs_tensor = (1.0 - 1e-5) * t

        batch_inputs = []
        batch_labels = []

        for idx, seq in enumerate(raw_seqs):
            ids = seq.tolist()
            prob = batch_probs_tensor[idx].item()
            indices = self._tail_bias_indices(len(ids), prob)
            
            inputs = ids[:]
            for i in indices:
                inputs[i] = self.mask_token_id

            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_labels.append(torch.tensor(ids, dtype=torch.long))

        # === 关键修改 ===
        # 使用 self._pad_batch 替代 pad_sequence
        input_ids = self._pad_batch(batch_inputs, padding_value=self.pad_token_id)
        labels = self._pad_batch(batch_labels, padding_value=-100)
        
        # 重新生成 mask (因为长度可能变了)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": batch_probs_tensor,
            "causal": True,
            "use_daum": self.use_daum,
        }

class MDLMCollator(BaseCollator, MaskedLanguageModelingMixin):
    def __init__(self, tokenizer, max_length: int = 512, text_key: str = "text", eps: float = 1e-5, pad_to_max_length: bool = False):
        super().__init__(tokenizer, max_length, text_key, add_special_tokens=True, pad_to_max_length=pad_to_max_length)
        self.eps = eps

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare_batch 内部已经调用了 self._pad_batch，所以 input_ids 已经是符合 padding 要求的形状
        input_ids, attention_mask, _ = self.prepare_batch(examples)
        B = input_ids.size(0)

        t = self.sample_antithetic_probs(B, low=0.001, high=0.999)
        probs = (1.0 - 1e-5) * t

        # mask_tokens 是基于 input_ids 操作的，所以形状会自动保持
        inputs, labels = self.mask_tokens(input_ids, probs)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": probs,
            "zero_mask_prob": True,
        }


class PrefixLMCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    真正的 Prefix Language Model Collator（用于双向 attention 的 LLaDA backbone）。
    随机选取一个分割点，prefix 保持原始 token（双向可见），suffix 全部替换为 [MASK]。
    Loss 仅在 suffix 上计算。
    """
    def __init__(self, tokenizer, max_length: int = 512, text_key: str = "text",
                 eps: float = 1e-5, pad_to_max_length: bool = False):
        super().__init__(tokenizer, max_length, text_key, add_special_tokens=True,
                         pad_to_max_length=pad_to_max_length)
        self.eps = eps

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, raw_seqs = self.prepare_batch(examples)
        B, L = input_ids.shape

        # 使用和 MDLM 相同的 antithetic 采样策略来决定 suffix 比例
        # t 表示被 mask 的比例（即 suffix 占序列的比例）
        t = self.sample_antithetic_probs(B, low=0.001, high=0.999)
        suffix_ratios = (1.0 - self.eps) * t

        special_tokens_mask = self.build_special_tokens_mask(input_ids)
        labels = input_ids.clone()
        xt = input_ids.clone()

        for i in range(B):
            # 找到可 mask 的位置（排除 BOS、EOS、PAD 等特殊 token）
            maskable = (~special_tokens_mask[i]).nonzero(as_tuple=True)[0]
            if len(maskable) == 0:
                continue
            # 根据 suffix_ratio 确定分割点
            n_suffix = max(1, int(len(maskable) * suffix_ratios[i].item()))
            # suffix 是序列尾部的 n_suffix 个可 mask 位置
            suffix_positions = maskable[-n_suffix:]
            # Prefix 保持原始 token，suffix 替换为 [MASK]
            xt[i, suffix_positions] = self.mask_token_id
            # NOTE: Do NOT set prefix labels to -100 here!
            # Prefix positions keep their original token IDs in labels,
            # just like MDLM does. The model's _apply_subs will force
            # unmasked positions' logits to one-hot (loss=0 automatically).
            # This ensures num_items_in_batch (computed by HF Trainer as
            # count of labels != -100) includes ALL content tokens, keeping
            # the loss scale consistent with MDLM.

        # PAD 位置的 label 设为 -100 (与 MDLM mask_tokens 一致:
        # 只将 pad 设为 -100, BOS/EOS 保留有效 label —— _apply_subs 会
        # 让这些 unmasked 位置的 loss 自动为 0)
        labels[input_ids == self.pad_token_id] = -100

        return {
            "input_ids": xt,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_mlm_prob": suffix_ratios,
        }


class BD3LMCollator(BaseCollator, MaskedLanguageModelingMixin):
    """
    BD3LM (Block Denoising Discrete Diffusion Language Models) 专用 Collator。
    已添加：保证每条数据至少 mask 一个 token。
    """
    def __init__(
        self, 
        tokenizer, 
        max_length: int = 512, 
        text_key: str = "text",
        block_size: int = 16,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
        resample: bool = True,
        pad_to_max_length: bool = False,
        **kwargs
    ):
        super().__init__(tokenizer, max_length, text_key, add_special_tokens=True, pad_to_max_length=pad_to_max_length)
        self.block_size = block_size
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.resample = resample
        # 预先计算特殊 token 集合
        self.special_token_ids = {
             k for k in [
                 self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                 self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                 self.tokenizer.sep_token_id, self.tokenizer.mask_token_id
             ] if k is not None
        }

    # ... _sample_block_t 保持不变 ...
    def _sample_block_t(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = input_ids.shape
        num_blocks = (L + self.block_size - 1) // self.block_size
        total_units = B * num_blocks
        offset = torch.arange(total_units, device=input_ids.device) / total_units
        offset = offset.view(B, num_blocks)
        rand_noise = torch.rand((B, num_blocks), device=input_ids.device)
        t_blocks = (rand_noise / total_units + offset) % 1.0
        t_expanded = t_blocks.repeat_interleave(self.block_size, dim=-1)
        t_expanded = t_expanded[:, :L]
        t_final = t_expanded * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
        return t_final, num_blocks

    # ... _resample_mask 保持不变 ...
    def _resample_mask(self, input_ids, xt, t_map, special_mask):
        if not self.resample: return xt
        B, L = input_ids.shape
        pad_len = (self.block_size - (L % self.block_size)) % self.block_size
        def to_blocks(x, pad_val=0):
            if pad_len > 0: x = torch.nn.functional.pad(x, (0, pad_len), value=pad_val)
            return x.view(B, -1, self.block_size)
        def from_blocks(x_blocked): return x_blocked.view(B, -1)[:, :L]
        
        t_blocks = to_blocks(t_map)[:, :, 0]
        
        for _ in range(5): # 最多尝试修正5次
            xt_blocked = to_blocks(xt, pad_val=self.tokenizer.pad_token_id)
            is_masked = (xt_blocked == self.mask_token_id).float()
            perc_masked = is_masked.sum(dim=-1) / self.block_size
            
            too_low = perc_masked < self.sampling_eps_min
            too_high = perc_masked > self.sampling_eps_max
            if self.sampling_eps_min <= 1e-3: too_low = torch.zeros_like(too_low, dtype=torch.bool)
            if self.sampling_eps_max >= 1.0: too_high = torch.zeros_like(too_high, dtype=torch.bool)
            
            needs_regen = too_low | too_high
            if not needs_regen.any(): break
            
            current_p = t_blocks
            current_p_expanded = current_p.unsqueeze(-1).expand_as(xt_blocked)
            new_decisions = torch.rand_like(current_p_expanded) < current_p_expanded
            
            mask_needs_update = needs_regen.unsqueeze(-1).expand_as(xt_blocked)
            input_blocked = to_blocks(input_ids, pad_val=self.tokenizer.pad_token_id)
            special_mask_blocked = to_blocks(special_mask, pad_val=1)
            
            new_xt_blocked = torch.where(new_decisions, self.mask_token_id, input_blocked)
            update_locs = mask_needs_update & (~special_mask_blocked)
            xt_blocked = torch.where(update_locs, new_xt_blocked, xt_blocked)
            xt = from_blocks(xt_blocked)
        return xt

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, _ = self.prepare_batch(examples)
        
        # 1. 准备 Special Mask
        special_tokens_mask = self.build_special_tokens_mask(input_ids)
        special_tokens_mask = special_tokens_mask | (input_ids == self.pad_token_id)

        # 2. 采样 t
        t_map, _ = self._sample_block_t(input_ids)
        
        # 3. 生成初始 Mask
        rand_matrix = torch.rand_like(t_map)
        mask_decision = rand_matrix < t_map
        mask_decision = mask_decision & (~special_tokens_mask)
        
        xt = input_ids.clone()
        xt[mask_decision] = self.mask_token_id
        
        # 4. Resampling
        if self.resample:
            xt = self._resample_mask(input_ids, xt, t_map, special_tokens_mask)


        # =========================================================

        # 6. 准备 Labels 和返回
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100

        return {
            "input_ids": xt,
            "attention_mask": attention_mask,
            "labels": labels,
            "cond_ids": input_ids,
            "timesteps": t_map,
            "causal": False,
        }