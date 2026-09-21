# eval/data/collator.py
"""
Data collators for batching evaluation samples.
"""

from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizer


class LogProbCollator:
    """
    Collator for log-probability evaluation.
    
    Takes a list of processed samples and creates a batched tensor dictionary
    suitable for model inference.
    
    Each sample is expected to have:
        - input_ids: Token IDs for the prompt
        - continuation_ids: Token IDs for the continuation
        - group_id: Identifier linking choices to the same question
        - is_correct: Whether this option is the correct answer
        - continuation_len: Length of continuation in tokens
        - continuation_char_len: Length of continuation in characters
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Dictionary with batched tensors and metadata
        """
        input_ids_list = []
        continuation_masks = []
        attention_masks = []
        
        group_ids = []
        is_correct_list = []
        continuation_len_list = []
        continuation_char_len_list = []
        
        max_length = 0
        
        # First pass: collect sequences and find max length
        for item in batch:
            full_sequence = item['input_ids'] + item['continuation_ids']
            input_ids_list.append(full_sequence)
            
            # Continuation mask: 0 for prompt, 1 for continuation
            continuation_mask = (
                [0] * len(item['input_ids']) + 
                [1] * len(item['continuation_ids'])
            )
            continuation_masks.append(continuation_mask)
            
            attention_masks.append([1] * len(full_sequence))
            
            group_ids.append(item['group_id'])
            is_correct_list.append(item['is_correct'])
            continuation_len_list.append(item['continuation_len'])
            continuation_char_len_list.append(
                item.get('continuation_char_len', 1)
            )
            
            max_length = max(max_length, len(full_sequence))
        
        # Second pass: pad to max length
        padded_input_ids = []
        padded_continuation_masks = []
        padded_attention_masks = []
        
        for seq, cont_mask, att_mask in zip(
            input_ids_list, continuation_masks, attention_masks
        ):
            pad_length = max_length - len(seq)
            padded_input_ids.append(seq + [self.pad_token_id] * pad_length)
            padded_continuation_masks.append(cont_mask + [0] * pad_length)
            padded_attention_masks.append(att_mask + [0] * pad_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'continuation_mask': torch.tensor(padded_continuation_masks, dtype=torch.bool),
            'group_ids': group_ids,
            'is_correct': is_correct_list,
            'continuation_len': continuation_len_list,
            'continuation_char_len': continuation_char_len_list,
        }