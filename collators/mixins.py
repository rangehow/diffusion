# collators/mixins.py
from typing import List, Tuple, Optional
import torch




class MaskedLanguageModelingMixin:
    """MLM / Diffusion 通用掩码工具，不要继承 BaseCollator！"""
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 仍然可以正常调用
        if not hasattr(self, 'mask_token_id') or self.mask_token_id is None:
            raise ValueError("使用 MLM 类 Collator 时 tokenizer 必须有 mask_token_id")

    def build_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in [self.pad_token_id, self.cls_token_id, self.sep_token_id,
                         self.bos_token_id, self.eos_token_id]:
            if token_id is not None:
                mask = mask | (input_ids == token_id)
        return mask

    def sample_antithetic_probs(self, batch_size: int, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """
        生成对立/分层采样的概率值。
        将区间 [low, high] 划分为 batch_size 个小区间，每个区间内采样一个点。
        """
        # 1. 生成 0 到 B-1 的序列
        steps = torch.arange(batch_size, dtype=torch.float32)
        
        # 2. 在每个阶梯上加一点随机噪声 [0, 1)
        noise = torch.rand(batch_size)
        
        # 3. 归一化到 [0, 1] -> t 均匀分布在 [0, 1] 但彼此间隔均匀
        # t = (i + epsilon) / B
        probs_01 = (steps + noise) / batch_size
        
        # 4. 缩放到目标区间 [low, high]
        probs = low + probs_01 * (high - low)
        
        # 5. 打乱顺序 (重要！否则 Batch 中前面的样本总是低噪声，后面的总是高噪声)
        shuffled_indices = torch.randperm(batch_size)
        return probs[shuffled_indices]


    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        mlm_probability: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if special_tokens_mask is None:
            special_tokens_mask = self.build_special_tokens_mask(input_ids)

        prob_matrix = mlm_probability.unsqueeze(1).expand_as(input_ids)
        masked_indices = torch.rand_like(input_ids, dtype=torch.float) < prob_matrix
        masked_indices = masked_indices & ~special_tokens_mask

        # 保证每个样本至少 mask 一个 token
        for i in range(input_ids.size(0)):
            if not masked_indices[i].any():
                candidates = (~special_tokens_mask[i]).nonzero(as_tuple=True)[0]
                if len(candidates) > 0:
                    chosen = candidates[torch.randint(len(candidates), (1,))]
                    masked_indices[i, chosen] = True

        labels = input_ids.clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        inputs = input_ids.clone()
        inputs[masked_indices] = self.mask_token_id

        return inputs, labels