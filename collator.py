import random
import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer


class MLMCollator:
    """
    用于BERT预训练MLM任务的数据整理器
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.15,  # 可以是数值或可调用对象
        mask_probability: float = 0.8,
        random_probability: float = 0.1,
        max_length: int = 512,
        text_key: str = 'text',
        mode: str = 'niu',
        use_all_tokens_for_loss: bool = True,
    ):
        """
        Args:
            tokenizer: 预训练的分词器
            mlm_probability: 被mask的token比例，可以是：
                - float: 固定比例 (如 0.15)
                - callable: 动态计算函数 (如 lambda: random.uniform(0.1, 0.2))
            mask_probability: 在被选中的token中，替换为[MASK]的比例
            random_probability: 在被选中的token中，替换为随机token的比例
            max_length: 最大序列长度
            text_key: 如果指定，会从examples中提取该key对应的文本并转换为input_ids
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability  # 保持原样，可以是数值或函数
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.max_length = max_length
        self.text_key = text_key  # 保存text_key
        self.mode = mode
        self.use_all_tokens_for_loss = use_all_tokens_for_loss
        # 获取特殊token的id
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.eos_token_id = tokenizer.eos_token_id  # 添加EOS token
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
        
        # 可用于随机替换的token范围
        self.vocab_size = tokenizer.vocab_size
        
        # 获取需要排除的token id集合
        self.excluded_token_ids = set()
        self.excluded_token_ids.add(self.mask_token_id)
        self.excluded_token_ids.add(self.pad_token_id)
        self.excluded_token_ids.add(self.cls_token_id)
        self.excluded_token_ids.add(self.sep_token_id)
        self.excluded_token_ids.add(self.bos_token_id)  # BOS也不能被mask
        
        # 添加所有added tokens
        if hasattr(tokenizer, 'added_tokens_decoder'):
            self.excluded_token_ids.update(tokenizer.added_tokens_decoder.keys())
        
        # 创建可用于随机替换的token列表
        self.valid_token_ids = [
            i for i in range(self.vocab_size) 
            if i not in self.excluded_token_ids
        ]
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels, token_change_labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_token_change_labels = []
        
        for example in examples:
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False,  # 我们手动添加BOS和EOS
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和EOS位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            
            # 确保序列以BOS开头，EOS结尾
            if input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids
            if input_ids[-1] != self.eos_token_id:
                input_ids = input_ids + [self.eos_token_id]
            
            # 确保序列长度不超过max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length-1] + [self.eos_token_id]
            

            current_mlm_prob = self._get_mlm_probability()
            # 创建MLM mask和labels
            masked_input_ids, labels, token_change_labels = self._mask_tokens(input_ids,current_mlm_prob)
         
            # 创建attention mask
            attention_mask = [1] * len(masked_input_ids)
            
            batch_input_ids.append(masked_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_token_change_labels.append(token_change_labels)
        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)
        batch_token_change_labels = self._pad_sequences(batch_token_change_labels, 0)
  
        if self.mode == 'llada':
            return {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
                'labels': torch.tensor(batch_labels, dtype=torch.long),
                'current_mlm_prob': torch.tensor(current_mlm_prob, dtype=torch.float),
                'return_dict':True,
            }
        else:
            return {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
                'labels': torch.tensor(batch_labels, dtype=torch.long),
                'token_change_labels': torch.tensor(batch_token_change_labels, dtype=torch.long),
                'current_mlm_prob': torch.tensor(current_mlm_prob, dtype=torch.float),
                'return_dict': True,
            }
    
    def _get_mlm_probability(self) -> float:
        """获取当前的MLM概率"""
        if callable(self.mlm_probability):
            prob = self.mlm_probability()
            # 确保返回值在合理范围内
            return max(0.0, min(1.0, prob))
        else:
            return self.mlm_probability

    def _mask_tokens(self, input_ids: List[int], current_mlm_prob ) -> tuple:
        """
        对输入序列进行MLM masking
        
        Args:
            input_ids: 输入token序列
            
        Returns:
            (masked_input_ids, labels, token_change_labels): mask后的序列、MLM标签和token变化标签
        """
        
        # 根据模式设置labels的初始值
        if self.use_all_tokens_for_loss:
            labels = [-100] * len(input_ids)  # llada模式：只计算被mask位置的loss
        else:
            labels = input_ids.copy()  # 其他模式：使用所有token
        
        token_change_labels = [0] * len(input_ids)  # 0表示未改变，1表示已改变
        
        # 获取可以被mask的位置（排除特殊token，但包括EOS）
        maskable_positions = []
        eos_position = None  # 记录EOS token的位置
        
        for i, token_id in enumerate(input_ids):
            if token_id not in [self.cls_token_id, self.sep_token_id, 
                               self.pad_token_id, self.bos_token_id]:  # BOS也不能被mask
                maskable_positions.append(i)
                if token_id == self.eos_token_id:
                    eos_position = i
        
        # 强制mask EOS token
        if eos_position is not None:
            labels[eos_position] = input_ids[eos_position]  # 保存原始EOS token作为标签
            
            rand = random.random()
            if rand < self.mask_probability:       
                input_ids[eos_position] = self.mask_token_id
                token_change_labels[eos_position] = 1 
            elif rand < self.mask_probability + self.random_probability:
                input_ids[eos_position] = random.choice(self.valid_token_ids)
                token_change_labels[eos_position] = 1
            
            # 从可mask位置中移除EOS位置，避免重复处理
            if eos_position in maskable_positions:
                maskable_positions.remove(eos_position)
        
        # 随机选择其他需要mask的位置
        num_to_mask = max(1, int(len(maskable_positions) * current_mlm_prob))
        masked_positions = random.sample(maskable_positions, 
                                       min(num_to_mask, len(maskable_positions)))
        
        for pos in masked_positions:
            labels[pos] = input_ids[pos]  # 保存原始token作为标签
            
            rand = random.random()
            if rand < self.mask_probability:       
                input_ids[pos] = self.mask_token_id
                token_change_labels[pos] = 1 
            elif rand < self.mask_probability + self.random_probability:
                input_ids[pos] = random.choice(self.valid_token_ids)
                token_change_labels[pos] = 1

        return input_ids, labels, token_change_labels
    
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
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,  # 手动添加BOS和EOS
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和EOS位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            
            # 确保序列以BOS开头，EOS结尾
            if input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids
            if input_ids[-1] != self.eos_token_id:
                input_ids = input_ids + [self.eos_token_id]
            
            # 确保序列长度不超过max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length-1] + [self.eos_token_id]
            
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