# collator.py

import random
import torch
from typing import Dict, List, Any,Tuple
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence



class NTPCollator:
    """
    用于NTP（Next Token Prediction）预训练任务的数据整理器
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
    ):
        """
        Args:
            tokenizer: 预训练的分词器
            max_length: 最大序列长度
            text_key: 如果指定，会从examples中提取该key对应的文本并转换为input_ids
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # 获取特殊token的id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            
            text = example[self.text_key]
            # input_ids = self.tokenizer.encode(
            #     text,
            #     add_special_tokens=False,  # 手动添加BOS和EOS
            #     truncation=True,
            #     max_length=self.max_length - 2  # 留出BOS和EOS位置
            # )

            
            
            # # 确保序列以BOS开头，EOS结尾
            # if input_ids[0] != self.bos_token_id:
            #     input_ids = [self.bos_token_id] + input_ids
            # if input_ids[-1] != self.eos_token_id:
            #     input_ids = input_ids + [self.eos_token_id]
            
            # # 确保序列长度不超过max_length
            # if len(input_ids) > self.max_length:
            #     input_ids = input_ids[:self.max_length]
            
            text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            input_ids = self.tokenizer.encode(text,add_special_tokens=False,max_length=self.max_length,truncation=True)

            # 创建NTP标签：直接使用input_ids，让模型前向处理移位
            labels = input_ids.copy()  # 直接复制，不做移位操作
            
            # 创建attention mask
            attention_mask = [1] * len(input_ids)
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'return_dict': True,
        }
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """
        对序列进行padding
        """
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    



class CausalLMCollator:
    """
    一个用于因果语言模型（Causal LM）的 Data Collator。
    它会随机 mask 输入序列的一部分，用于去噪自编码（denoising autoencoding）式的预训练。
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
        start_prob: float = 0.0,
        end_prob : float = 1.0,
        tail_bias_factor: float = 1.5,
        use_daum = False,
        **kwargs
    ):
        """
        Args:
            tokenizer: 预训练的分词器。
            max_length: 最大序列长度。
            text_key: 如果指定，会从examples中提取该key对应的文本并转换为input_ids。
            start_prob: 随机 mask 比例的下限。
            end_prob: 随机 mask 比例的上限。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.tail_bias_factor = tail_bias_factor
        self.use_daum = use_daum
        assert self.tail_bias_factor >= 1.0, "tail_bias_factor must be >= 1.0"

        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id

        self.vocab_size = tokenizer.vocab_size
        if kwargs:
            print("未被使用的参数有: ", kwargs)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_mlm_probs = []

        for example in examples:
            
            text = example[self.text_key]
            text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            input_ids = self.tokenizer.encode(text,add_special_tokens=False,max_length=self.max_length,truncation=True)

            current_mlm_prob = self._get_mlm_probability()
            batch_mlm_probs.append(current_mlm_prob)


            corrupted_input_ids, labels = self._corrupt_and_create_labels(
                input_ids,
                current_mlm_prob
            )

            attention_mask = [1] * len(corrupted_input_ids)

            batch_input_ids.append(corrupted_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)

        batch = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'current_mlm_prob': torch.tensor(batch_mlm_probs, dtype=torch.float),
            'return_dict': True,
            'causal': True,
            'num_items_in_batch':None,
            'use_daum':self.use_daum
        }

        return batch

    def _get_mlm_probability(self) -> float:
        """获取当前的MLM概率"""
        # 确保概率在 (0, 1) 区间内，避免极端情况
        prob = random.uniform(min(self.start_prob, self.end_prob), max(self.start_prob, self.end_prob))
        prob = (1 - 1e-3) * prob + 1e-3
        return prob


    def _corrupt_and_create_labels(self, input_ids: List[int], current_mlm_prob: float) -> tuple:
        """
        对序列进行尾部偏置的随机加噪（仅使用MASK），并创建labels。
        """
        labels = input_ids.copy()
        
        # 确定可以被mask的token的索引范围 (排除BOS和EOS)
        all_maskable_indices = list(range(1, len(input_ids) - 1))

        if not all_maskable_indices:
            return input_ids, labels

        # 1. 计算需要加噪的token数量
        num_to_mask = max(1, int(len(all_maskable_indices) * current_mlm_prob))
        
        # 2. 根据 tail_bias_factor 计算尾部候选池的大小
        pool_size = int(num_to_mask * self.tail_bias_factor)
        # 确保池子大小不超过所有可mask的数量，且不小于需要mask的数量
        pool_size = min(pool_size, len(all_maskable_indices))
        pool_size = max(pool_size, num_to_mask)

        # 3. 从所有可mask的索引中，提取出尾部的候选池
        # 例如: all_maskable_indices = [1,2,3,4,5,6,7,8], pool_size = 4
        # candidate_indices 就会是 [5,6,7,8]
        candidate_indices = all_maskable_indices[-pool_size:]
        
        # 4. 从这个尾部候选池中，随机抽取最终要mask的位置
        # 例如: 从 [5,6,7,8] 中随机抽取 num_to_mask 个
        indices_to_mask = random.sample(candidate_indices, num_to_mask)
        
        for pos in indices_to_mask:
            input_ids[pos] = self.mask_token_id

        # 返回被破坏的输入和原始的标签
        return input_ids, labels


    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """将序列列表padding到该批次中的最大长度"""
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        padded_sequences = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return padded_sequences







class LLaDACollator:
    """
    用于离散扩散模型训练任务的数据整理器。
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # 获取并验证特殊token的ID
        special_tokens = {
            'mask_token_id': tokenizer.mask_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'cls_token_id': tokenizer.cls_token_id,
            'sep_token_id': tokenizer.sep_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token_id': tokenizer.bos_token_id,
        }
        for name, token_id in special_tokens.items():
            if token_id is None:
                raise ValueError(
                    f"Tokenizer is missing '{name}'. Please ensure it's a BERT-like or"
                    " a model with a complete set of special tokens."
                )
            setattr(self, name, token_id)

    def _get_mlm_probability(self, eps: float = 1e-5) -> float:
        """
        为单个样本生成一个随机的掩码概率t，模拟扩散过程中的不同时间步。
        """
        # 从一个接近0但不等于0的范围开始，到1结束
        t = random.uniform(eps, 1.0-eps)
        # 这是一个简单的线性噪声调度，可以根据需要换成更复杂的，如cosine schedule
        return t

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据。
        该方法分为两个阶段：
        1. 逐样本处理：将文本转换为token ID，并添加特殊token。
        2. 批量向量化处理：对整个batch进行padding、掩码和标签生成。
        """
        
        # --- 阶段1: 逐样本处理 ---
        sequences = []
        batch_mlm_probs = []
        
        for example in examples:
            text = example[self.text_key]
            text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            input_ids = self.tokenizer.encode(text,add_special_tokens=False,max_length=self.max_length,truncation=True)

            sequences.append(torch.tensor(input_ids, dtype=torch.long))
            batch_mlm_probs.append(self._get_mlm_probability())

        # --- 阶段2: 批量向量化处理 ---
        
        # 使用pad_sequence进行高效padding
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        
        # 准备labels和attention_mask
        labels = input_ids.clone()
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 创建一个布尔张量，标记所有特殊token的位置
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id) |
            (input_ids == self.bos_token_id) |
            (input_ids == self.eos_token_id)
        )

        # 进行概率掩码
        # 将每个样本的掩码概率转换为(batch_size, 1)的张量，以便广播
        probs_tensor = torch.tensor(batch_mlm_probs, dtype=torch.float).unsqueeze(1)
        # 生成一个与input_ids形状相同的随机数矩阵
        rand = torch.rand(input_ids.shape)
        # 当随机数小于该位置的掩码概率时，标记为待掩码
        masked_indices = (rand < probs_tensor) & (~special_tokens_mask)
        
        # 鲁棒性检查：确保每个序列至少有一个token被掩码
        for i in range(input_ids.size(0)):
            if not masked_indices[i].any():
                # 找到该样本中可以被mask的位置
                candidate_indices = (~special_tokens_mask[i]).nonzero(as_tuple=True)[0]
                if len(candidate_indices) > 0:
                    # 随机选择一个位置进行mask
                    chosen_idx = candidate_indices[torch.randint(0, len(candidate_indices), (1,))]
                    masked_indices[i, chosen_idx] = True

        # 将未被掩码的token的label设置为-100，以便在计算损失时被忽略
        labels[~masked_indices] = -100
        
        # 将被选中的位置替换为mask_token_id
        input_ids[masked_indices] = self.mask_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'current_mlm_prob': torch.tensor(batch_mlm_probs, dtype=torch.float),
        }




class MDLMCollator:
    """
    用于离散扩散模型训练任务的最终版数据整理器。
  
    该版本严格对齐了论文作者官方代码库中的 `LogLinearNoise` 实现。
    核心逻辑是掩码概率与时间步 t 呈几乎线性的关系: p_mask = (1 - eps) * t。
    其中 eps 是一个为了数值稳定性而引入的微小常数。
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
        eps: float = 1e-5, # 使用一个微小的eps，与作者代码保持一致
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.eps = eps
        
        # 获取并验证特殊token的ID
        special_tokens = {
            'mask_token_id': tokenizer.mask_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'cls_token_id': tokenizer.cls_token_id,
            'sep_token_id': tokenizer.sep_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token_id': tokenizer.bos_token_id,
        }
        for name, token_id in special_tokens.items():
            if token_id is None:
                raise ValueError(
                    f"Tokenizer is missing '{name}'. Please ensure it's a BERT-like or"
                    " a model with a complete set of special tokens."
                )
            setattr(self, name, token_id)

    def _get_low_discrepancy_timesteps(self, batch_size: int) -> torch.Tensor:
        """为batch生成低差异/分层采样的时间步 t in (0, 1]"""
        i = torch.arange(batch_size, dtype=torch.float32)
        timesteps = (i + torch.rand(batch_size)) / batch_size
        return timesteps

    def _noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        根据时间步 t 计算掩码概率，严格遵循作者的 LogLinearNoise 实现。
        推导结果为 p_mask = (1 - eps) * t。
        """
        mask_prob = (1.0 - self.eps) * t
        return torch.clamp(mask_prob, 0.0, 1.0)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        sequences = []
        batch_size = len(examples)
        
        for example in examples:
            text = example[self.text_key]
            text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            input_ids = self.tokenizer.encode(text,add_special_tokens=False,max_length=self.max_length,truncation=True)
            sequences.append(torch.tensor(input_ids, dtype=torch.long))

        timesteps = self._get_low_discrepancy_timesteps(batch_size)
        batch_mlm_probs = self._noise_schedule(timesteps)

        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        
        labels = input_ids.clone()
        attention_mask = (input_ids != self.pad_token_id).long()
        
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id) |
            (input_ids == self.bos_token_id) |
            (input_ids == self.eos_token_id)
        )

        # 使用每个样本对应的掩码概率进行掩码
        probs_tensor = batch_mlm_probs.unsqueeze(1).expand_as(input_ids)
        rand = torch.rand(input_ids.shape)
        masked_indices = (rand < probs_tensor) & (~special_tokens_mask)
        
        # 确保每个序列至少有一个token被掩码，这有助于稳定训练
        for i in range(input_ids.size(0)):
            if not masked_indices[i].any():
                candidate_indices = (~special_tokens_mask[i]).nonzero(as_tuple=True)[0]
                if len(candidate_indices) > 0:
                    chosen_idx = candidate_indices[torch.randint(0, len(candidate_indices), (1,))]
                    masked_indices[i, chosen_idx] = True

        labels[~masked_indices] = -100 # -100 是 PyTorch CrossEntropyLoss 的忽略索引
        input_ids[masked_indices] = self.mask_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'current_mlm_prob': batch_mlm_probs,
            'zero_mask_prob': True,
        }




# collator_bd3lm.py
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer
import numpy as np

# collator_bd3lm.py  （完全对齐版，专为 block_size=16 以及任意 block_size 设计）

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


class BD3LMDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        block_size: int = 16,                  # ← 你的情况就是 16
        cross_attn: bool = True,
        ignore_bos: bool = True,
        var_min: bool = False,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
        sigma_max: float = 80.0,               # 原项目 loglinear 默认值
        antithetic_sampling: bool = True,      # ← 必须开，和 config 保持一致
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.cross_attn = cross_attn
        self.ignore_bos = ignore_bos
        self.var_min = var_min
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.sigma_max = sigma_max
        self.antithetic_sampling = antithetic_sampling

        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must have mask_token_id")
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _sample_t_stratified_antithetic(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        100% 复现原 Diffusion._sample_t 的 stratified + antithetic 逻辑
        返回 shape: (B, num_blocks)  在 [0, 1) 上均匀、方差最小的 t
        """
        total = batch_size * num_blocks

        rand = torch.rand(batch_size, num_blocks, device=device)

        if self.antithetic_sampling:
            # 全局均匀偏移，和原代码一模一样
            offset = torch.arange(total, device=device, dtype=torch.float32) / total
            offset = offset.view(batch_size, num_blocks)
            t_block = (rand + offset) % 1.0
        else:
            t_block = rand

        # 映射到用户指定的采样区间 [eps_min, eps_max]
        t_block = t_block * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
        return t_block  # (B, num_blocks)

    def _resample_q_xt(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        move_mask: torch.Tensor,
        p_block: torch.Tensor,   # (B, num_blocks)
    ) -> torch.Tensor:
        """完全复现原 _resample_q_xt 的 while-loop 逻辑"""
        B, L = x0.shape
        bs = self.block_size
        n_blocks = L // bs

        xt = xt.view(B, n_blocks, bs)
        move_mask = move_mask.view(B, n_blocks, bs)
        p_block = p_block.unsqueeze(-1)          # (B, n_blocks, 1)

        while True:
            masked_ratio = (xt == self.mask_id).float().mean(-1)      # (B, n_blocks)

            regen = (masked_ratio < self.sampling_eps_min) | (masked_ratio > self.sampling_eps_max)
            if not regen.any():
                break

            # 只对不达标的 block 重新采样
            new_move = torch.rand_like(move_mask, dtype=torch.float32) < p_block
            move_mask = torch.where(regen.unsqueeze(-1), new_move, move_mask)
            xt = torch.where(move_mask, self.mask_id, x0.view(B, n_blocks, bs))

        xt = xt.view(B, -1)
        return xt

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Tokenize
        input_ids_list = [
            torch.tensor(self.tokenizer.encode(ex["text"], add_special_tokens=True), dtype=torch.long)
            for ex in examples
        ]
        if any(len(ids) > self.max_length for ids in input_ids_list):
            input_ids_list = [ids[:self.max_length] for ids in input_ids_list]

        x0 = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_id)   # (B, L)
        attention_mask = (x0 != self.pad_id).long()
        B, L = x0.shape
        assert L % self.block_size == 0, f"seq len {L} must be divisible by block_size {self.block_size}"
        num_blocks = L // self.block_size

        device = x0.device

        # 2. 采样 t（关键！完全对齐原版）
        t_block = self._sample_t_stratified_antithetic(B, num_blocks, device)   # (B, num_blocks)

        # 3. LogLinear noise schedule（per-block）
        p_block = 1.0 - torch.exp(-self.sigma_max * t_block)                    # (B, num_blocks)
        sigma_block = -torch.log(1.0 - p_block + 1e-12)                         # (B, num_blocks)
        loss_scale_block = self.sigma_max / torch.expm1(sigma_block + 1e-12)   # (B, num_blocks)

        # 扩展到 token 粒度
        p = p_block.repeat_interleave(self.block_size, dim=-1)                  # (B, L)
        sigma = sigma_block.repeat_interleave(self.block_size, dim=-1)
        loss_scale = loss_scale_block.repeat_interleave(self.block_size, dim=-1)

        # 4. q(xt|x0)
        move = torch.rand_like(x0, dtype=torch.float32) < p
        xt = torch.where(move, self.mask_id, x0)

        # 5. var_min 重采样（和原版完全一致）
        if self.var_min:
            xt = self._resample_q_xt(x0, xt, move, p_block)

        # 6. ignore_bos
        if self.ignore_bos:
            xt[:, 0] = x0[:, 0]

        # 7. 构造模型输入
        if self.cross_attn:
            model_input_ids = torch.cat([xt, x0], dim=1)               # (B, 2*L)
            model_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
        else:
            model_input_ids = xt
            model_attention_mask = attention_mask

        return {
            "input_ids": model_input_ids,              # 送给模型
            "attention_mask": model_attention_mask,
            "labels": x0,                               # 原始干净序列（前 L）
            "timesteps": sigma.unsqueeze(1),            # (B, 1) 模型 conditioning
            "loss_scale": loss_scale.unsqueeze(1),      # (B, 1) 损失权重
            "original_attention_mask": attention_mask,  # 用于外部 metric
        }