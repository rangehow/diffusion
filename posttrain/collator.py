import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any

class LLaDASFTCollator:
    """
    用于 LLaDA 监督微调（SFT）任务的数据整理器。
    
    该整理器专门为 LLaDA 的 SFT 流程设计，其行为与标准的 SFT Collator 有两个关键区别：
    根据llada论文里的B.1，llada在sft时需要关注到pad token，而且此时pad token是使用eos token，从而让模型学会在一个给定的预算长度下通过eos来压缩长度
    因此多出的EOS也要被列入学习目标。
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): 用于获取 padding token ID 的分词器。
            pad_to_multiple_of (int): 为了提高硬件（如 Tensor Cores）利用率，
                                      将序列长度填充到这个值的倍数。
        """
        self.tokenizer = tokenizer
        
        # 对于 LLaDA SFT, label 的填充值就是 eos_token_id 本身
        self.label_pad_token_id = self.tokenizer.eos_token_id
        

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个批次的数据。
        
        该方法的核心任务是将预处理好的 `input_ids` 和 `labels` 列表进行填充，
        并组装成一个符合 LLaDA SFT 训练要求的批次。
        """
        
        # --- 阶段1: 从样本中提取 input_ids 和 labels ---
        # 假设输入的 'examples' 是 sft_map_fn 处理后的结果列表
        # 每个 example 是一个字典，包含 'input_ids' 和 'labels' 键
        
        messages_list = [self.template.apply(e) for e in examples['messages']]


        input_ids_list = [torch.tensor(example['input_ids'], dtype=torch.long) for example in examples]
        labels_list = [torch.tensor(example['labels'], dtype=torch.long) for example in examples]

        # --- 阶段2: 批量向量化处理 (主要是填充) ---

        # 使用 pad_sequence 进行高效填充
        # input_ids 用 pad_token_id 填充
        input_ids = pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        # labels 用我们指定的 label_pad_token_id 填充
        labels = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=self.label_pad_token_id
        )
        
        
        # --- 最终组装 ---
        batch = {
            'input_ids': input_ids,
            'labels': labels,
        }
        
        return batch


class NormalDLLMSFTCollator:
    ...


class ARMSFTCollator:
    ...