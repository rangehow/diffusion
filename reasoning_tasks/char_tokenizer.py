"""
Character-level tokenizer for reasoning tasks (Countdown + Sudoku).
Matches Ye et al. (2024) "Beyond Autoregression" tokenizer design:
each character maps to exactly one token.

Vocab (22 tokens):
  0: [PAD]     5: 0    10: 5    15: ,    20: =
  1: [BOS]     6: 1    11: 6    16: +    21: (space)
  2: [EOS]     7: 2    12: 7    17: -
  3: [MASK]    8: 3    13: 8    18: *
  4: [SEP]     9: 4    14: 9    19: /

Compatible with HuggingFace PreTrainedTokenizer for seamless integration
with the existing training/eval pipeline.
"""

import os
import json
from typing import List, Optional, Dict
from transformers import PreTrainedTokenizer


# Vocabulary definition
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"]
CHAR_TOKENS = list("0123456789,+-*/= ")

VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + CHAR_TOKENS)}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

VOCAB_SIZE = len(VOCAB)  # 22

# Verify
assert VOCAB_SIZE == 22, f"Expected 22 tokens, got {VOCAB_SIZE}"
assert VOCAB["[PAD]"] == 0
assert VOCAB["[BOS]"] == 1
assert VOCAB["[EOS]"] == 2
assert VOCAB["[MASK]"] == 3
assert VOCAB["[SEP]"] == 4


class ReasoningCharTokenizer(PreTrainedTokenizer):
    """
    Character-level tokenizer for Countdown and Sudoku tasks.
    Each character in the vocabulary maps to exactly one token ID.
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(self, **kwargs):
        # Set special token attributes BEFORE calling super().__init__
        # so that they are properly registered
        self._vocab = dict(VOCAB)
        self._id_to_token = dict(ID_TO_TOKEN)
        
        # Pop any special token args that might conflict
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        kwargs.pop("mask_token", None)
        kwargs.pop("sep_token", None)
        
        super().__init__(
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            sep_token="[SEP]",
            **kwargs,
        )
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize by splitting into individual characters."""
        tokens = []
        for ch in text:
            if ch in self._vocab:
                tokens.append(ch)
            else:
                # Skip unknown characters (shouldn't happen with our data)
                pass
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get("[PAD]", 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "[PAD]")
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Join tokens back into a string."""
        return "".join(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1=None) -> List[int]:
        """Add BOS at start, EOS at end."""
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if t in (self.bos_token_id, self.eos_token_id, self.pad_token_id, 
                               self.mask_token_id, self.sep_token_id) else 0 
                    for t in token_ids_0]
        return [1] + [0] * len(token_ids_0) + [1]
    
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * (len(token_ids_0) + 2)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Save vocabulary as a JSON file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        prefix = filename_prefix + "-" if filename_prefix else ""
        vocab_file = os.path.join(save_directory, prefix + "vocab.json")
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load from a directory. If vocab.json exists, load it; otherwise use defaults."""
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
            if os.path.exists(vocab_file):
                # Vocab exists, but we use the hardcoded one anyway
                # (it should be identical)
                pass
            # Check for tokenizer_config.json
            config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
            if os.path.exists(config_file):
                with open(config_file) as f:
                    config = json.load(f)
                # Remove tokenizer_class to avoid auto-resolution issues
                config.pop("tokenizer_class", None)
                config.pop("auto_map", None)
                kwargs.update(config)
        
        return cls(**kwargs)


def save_tokenizer(output_dir: str):
    """Save the reasoning character tokenizer to a directory."""
    tokenizer = ReasoningCharTokenizer()
    tokenizer.save_pretrained(output_dir)
    
    # Also save a simpler tokenizer_config.json
    config = {
        "tokenizer_class": "ReasoningCharTokenizer",
        "vocab_size": VOCAB_SIZE,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "sep_token": "[SEP]",
        "model_max_length": 512,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved tokenizer to {output_dir}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Special tokens: PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, "
          f"EOS={tokenizer.eos_token_id}, MASK={tokenizer.mask_token_id}, SEP={tokenizer.sep_token_id}")
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="reasoning_tasks/char_tokenizer")
    args = parser.parse_args()
    
    tokenizer = save_tokenizer(args.output_dir)
    
    # Test
    print("\n--- Tokenization Tests ---")
    tests = [
        "4,17,19,49",           # CD3 prompt
        "4*17=68,68-19=49",     # CD3 response
        "050140309",            # Sudoku puzzle (no spaces)
        "852146379",            # Sudoku solution (no spaces)
    ]
    for text in tests:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        print(f"  '{text}' -> {ids} (len={len(ids)}) -> '{decoded}'")
    
    # Test with special tokens
    text = "4,17,19,49"
    ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"\n  With BOS/EOS: '{text}' -> {ids} (len={len(ids)})")
    
    # Sudoku full sequence length
    puzzle = "0" * 81  # 81 digits
    solution = "1" * 81  # 81 digits
    p_ids = tokenizer.encode(puzzle, add_special_tokens=False)
    s_ids = tokenizer.encode(solution, add_special_tokens=False)
    print(f"\n  Sudoku puzzle: {len(p_ids)} tokens")
    print(f"  Sudoku solution: {len(s_ids)} tokens")
    print(f"  Total (BOS + puzzle + SEP + solution + EOS): {1 + len(p_ids) + 1 + len(s_ids) + 1} = {3 + 81 + 81}")
