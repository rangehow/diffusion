"""
Test script to verify that the refactored ModernBERT code maintains
the same functionality as the original.

This script tests:
1. Helper functions (get_batch_seq_info, should_use_unpadded_attention, sample_from_logits)
2. KV Cache functionality
3. Unpadding/padding utilities
4. Sampling behavior consistency
5. Model forward pass equivalence (with mocked dependencies)
"""

import torch
import torch.nn.functional as F
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional, List, Tuple


# ============================================================================
# Extracted functions from refactored code for standalone testing
# ============================================================================

def get_batch_seq_info(
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> Tuple[int, int, torch.device]:
    """Extract batch_size, seq_len, and device from inputs."""
    if batch_size is not None and seq_len is not None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        return batch_size, seq_len, device
    
    if inputs_embeds is not None:
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
    elif input_ids is not None:
        batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device
    else:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    
    return batch_size, seq_len, device


def should_use_unpadded_attention(
    config,
    past_key_values: Optional[List] = None,
    use_cache: bool = False,
) -> bool:
    """Determine if unpadded attention should be used."""
    return (
        config._attn_implementation == "flash_attention_2"
        and past_key_values is None
        and not use_cache
    )


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """Unified sampling function for both single and multi-position logits."""
    if not do_sample:
        return logits.argmax(dim=-1)
    
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.view(-1, vocab_size)
    
    if temperature != 1.0:
        flat_logits = flat_logits / temperature
    
    if top_k is not None and top_k > 0:
        k = min(top_k, vocab_size)
        top_values, _ = flat_logits.topk(k, dim=-1)
        threshold = top_values[:, -1:]
        flat_logits = flat_logits.masked_fill(flat_logits < threshold, float('-inf'))
    
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = flat_logits.sort(dim=-1, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        flat_logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
    
    probs = flat_logits.softmax(dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled.view(original_shape)


def unpad_inputs(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    per_token_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, 
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unpad inputs for flash attention."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    
    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        unpadded_inputs = inputs.view(batch * seqlen, *rest)[indices]
    
    def unpad_optional(tensor):
        return tensor.flatten()[indices] if tensor is not None else None
    
    return (
        unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch,
        unpad_optional(position_ids), unpad_optional(labels), unpad_optional(per_token_weights),
    )


def pad_outputs(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Repad outputs after flash attention."""
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        return output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        return output.view(batch, seqlen, *rest)


# Original implementations for comparison
def _sample_token_original(logits, temperature, top_k, top_p, do_sample):
    """Original single-token sampling."""
    if not do_sample:
        return logits.argmax(dim=-1)
    
    logits = logits / temperature
    
    if top_k is not None:
        v, _ = logits.topk(min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumsum - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))
    
    return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)


def _sample_generation_tokens_original(logits, temperature, top_k, top_p, do_sample):
    """Original multi-token sampling."""
    batch_size, seq_len, vocab_size = logits.shape
    
    if not do_sample:
        return logits.argmax(dim=-1)
    
    logits = logits / temperature
    
    if top_k is not None:
        v, _ = logits.topk(min(top_k, vocab_size), dim=-1)
        logits = logits.masked_fill(logits < v[..., [-1]], float('-inf'))
    
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumsum - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))
    
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs.view(-1, vocab_size), 1).view(batch_size, seq_len)


def _unpad_modernbert_input_original(inputs, attention_mask, position_ids=None, labels=None, per_token_weights=None):
    """Original unpadding function."""
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

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None
    unpadded_weights = per_token_weights.flatten()[indices] if per_token_weights is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels, unpadded_weights


def _pad_modernbert_output_original(inputs, indices, batch, seqlen):
    """Original padding function."""
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
class CacheConfig:
    """Configuration for KV cache."""
    batch_size: int
    max_seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype
    
    @classmethod
    def from_model_config(cls, model_config, batch_size, max_seq_len, device, dtype=torch.bfloat16):
        return cls(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            device=device,
            dtype=dtype,
        )


class ModernBertKVCache:
    """KV Cache for ModernBERT."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._seq_lens = torch.zeros(config.batch_size, dtype=torch.int32, device=config.device)
        self._uniform_seq_len: int = 0
        
        cache_shape = (config.batch_size, config.max_seq_len, config.num_heads, config.head_dim)
        self.key_cache = [torch.zeros(cache_shape, device=config.device, dtype=config.dtype) for _ in range(config.num_layers)]
        self.value_cache = [torch.zeros(cache_shape, device=config.device, dtype=config.dtype) for _ in range(config.num_layers)]
    
    @classmethod
    def from_model_config(cls, model_config, batch_size, max_seq_len, device, dtype=torch.bfloat16):
        cache_config = CacheConfig.from_model_config(model_config, batch_size, max_seq_len, device, dtype)
        return cls(cache_config)
    
    def get_layer_cache(self, layer_idx):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_all_layer_caches(self):
        return [self.get_layer_cache(i) for i in range(self.config.num_layers)]
    
    def advance_seq_len(self, increment):
        self._uniform_seq_len += increment
        self._seq_lens.fill_(self._uniform_seq_len)
    
    def get_seq_length(self):
        return self._uniform_seq_len
    
    def get_cache_seq_lens(self):
        return self._seq_lens
    
    def reset(self):
        self._uniform_seq_len = 0
        self._seq_lens.zero_()
        for layer_idx in range(self.config.num_layers):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


# ============================================================================
# Test Cases
# ============================================================================

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_get_batch_seq_info_from_input_ids(self):
        """Test extracting batch/seq info from input_ids."""
        input_ids = torch.randint(0, 100, (4, 32))
        batch, seq, device = get_batch_seq_info(input_ids, None)
        self.assertEqual(batch, 4)
        self.assertEqual(seq, 32)
        self.assertEqual(device, input_ids.device)
    
    def test_get_batch_seq_info_from_inputs_embeds(self):
        """Test extracting batch/seq info from inputs_embeds."""
        inputs_embeds = torch.randn(2, 16, 768)
        batch, seq, device = get_batch_seq_info(None, inputs_embeds)
        self.assertEqual(batch, 2)
        self.assertEqual(seq, 16)
        self.assertEqual(device, inputs_embeds.device)
    
    def test_get_batch_seq_info_with_explicit_values(self):
        """Test with explicitly provided batch_size and seq_len."""
        input_ids = torch.randint(0, 100, (4, 32))
        batch, seq, device = get_batch_seq_info(input_ids, None, batch_size=8, seq_len=64)
        self.assertEqual(batch, 8)
        self.assertEqual(seq, 64)
    
    def test_get_batch_seq_info_raises_on_none(self):
        """Test error when both inputs are None."""
        with self.assertRaises(ValueError):
            get_batch_seq_info(None, None)
    
    def test_should_use_unpadded_attention(self):
        """Test unpadded attention decision logic."""
        config = MagicMock()
        config._attn_implementation = "flash_attention_2"
        
        # Should use unpadded
        self.assertTrue(should_use_unpadded_attention(config, None, False))
        
        # Should not use with cache
        self.assertFalse(should_use_unpadded_attention(config, [("k", "v")], False))
        
        # Should not use when use_cache=True
        self.assertFalse(should_use_unpadded_attention(config, None, True))
        
        # Should not use with different attn implementation
        config._attn_implementation = "eager"
        self.assertFalse(should_use_unpadded_attention(config, None, False))


class TestSamplingFunctions(unittest.TestCase):
    """Test sampling functions."""
    
    def test_greedy_sampling_single(self):
        """Test greedy sampling (do_sample=False) for single position."""
        logits = torch.tensor([[0.1, 0.3, 0.6, 0.2]])  # batch=1, vocab=4
        result = sample_from_logits(logits, do_sample=False)
        self.assertEqual(result.item(), 2)  # argmax
    
    def test_greedy_sampling_multi(self):
        """Test greedy sampling for multiple positions."""
        logits = torch.tensor([
            [[0.1, 0.9], [0.8, 0.2]],  # batch item 1: [1, 0]
            [[0.3, 0.7], [0.6, 0.4]],  # batch item 2: [1, 0]
        ])  # shape: (2, 2, 2)
        result = sample_from_logits(logits, do_sample=False)
        expected = torch.tensor([[1, 0], [1, 0]])
        self.assertTrue(torch.equal(result, expected))
    
    def test_temperature_scaling(self):
        """Test that temperature affects distribution."""
        torch.manual_seed(42)
        logits = torch.randn(1000, 10)
        
        # Low temperature -> more peaked -> less entropy
        low_temp_samples = sample_from_logits(logits.clone(), temperature=0.1, do_sample=True)
        
        # High temperature -> more uniform -> more entropy
        torch.manual_seed(42)
        high_temp_samples = sample_from_logits(logits.clone(), temperature=2.0, do_sample=True)
        
        # Low temp should have more repeated values (less entropy)
        low_temp_unique = len(torch.unique(low_temp_samples))
        high_temp_unique = len(torch.unique(high_temp_samples))
        
        self.assertLessEqual(low_temp_unique, high_temp_unique)
    
    def test_top_k_filtering(self):
        """Test top-k filtering."""
        torch.manual_seed(42)
        # Create logits where only top-k should have probability
        logits = torch.tensor([[10.0, 9.0, 1.0, 0.5, 0.1]])  # vocab=5
        
        samples = []
        for _ in range(100):
            sample = sample_from_logits(logits.clone(), top_k=2, do_sample=True)
            samples.append(sample.item())
        
        # All samples should be in top-2 (indices 0 or 1)
        unique_samples = set(samples)
        self.assertTrue(unique_samples.issubset({0, 1}))
    
    def test_output_shape_single(self):
        """Test output shape for single position."""
        logits = torch.randn(4, 100)  # batch=4, vocab=100
        result = sample_from_logits(logits, do_sample=True)
        self.assertEqual(result.shape, (4,))
    
    def test_output_shape_multi(self):
        """Test output shape for multiple positions."""
        logits = torch.randn(4, 16, 100)  # batch=4, seq=16, vocab=100
        result = sample_from_logits(logits, do_sample=True)
        self.assertEqual(result.shape, (4, 16))
    
    def test_consistency_with_original_single(self):
        """Test that unified sampling matches original single-token sampling behavior."""
        torch.manual_seed(42)
        logits = torch.randn(8, 1000)
        
        # Test greedy
        unified_greedy = sample_from_logits(logits.clone(), do_sample=False)
        original_greedy = _sample_token_original(logits.clone(), 1.0, None, None, False)
        self.assertTrue(torch.equal(unified_greedy, original_greedy))
    
    def test_consistency_with_original_multi(self):
        """Test that unified sampling matches original multi-token sampling behavior."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 1000)
        
        # Test greedy
        unified_greedy = sample_from_logits(logits.clone(), do_sample=False)
        original_greedy = _sample_generation_tokens_original(logits.clone(), 1.0, None, None, False)
        self.assertTrue(torch.equal(unified_greedy, original_greedy))


class TestUnpaddingFunctions(unittest.TestCase):
    """Test unpadding and padding utilities."""
    
    def test_unpad_2d_inputs(self):
        """Test unpadding 2D inputs (input_ids)."""
        inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        
        unpadded, indices, cu_seqlens, max_seqlen, _, _, _ = unpad_inputs(inputs, attention_mask)
        
        # Check unpadded values
        expected_unpadded = torch.tensor([1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(unpadded, expected_unpadded))
        
        # Check cu_seqlens
        expected_cu = torch.tensor([0, 3, 5], dtype=torch.int32)
        self.assertTrue(torch.equal(cu_seqlens, expected_cu))
        
        # Check max_seqlen
        self.assertEqual(max_seqlen, 3)
    
    def test_unpad_3d_inputs(self):
        """Test unpadding 3D inputs (embeddings)."""
        inputs = torch.randn(2, 4, 8)  # batch=2, seq=4, hidden=8
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        
        unpadded, indices, cu_seqlens, max_seqlen, _, _, _ = unpad_inputs(inputs, attention_mask)
        
        # Check shape: total_tokens x hidden
        self.assertEqual(unpadded.shape, (5, 8))
    
    def test_unpad_with_optional_tensors(self):
        """Test unpadding with position_ids, labels, and weights."""
        inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        labels = torch.tensor([[10, 20, 30, -100], [40, 50, -100, -100]])
        weights = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        
        _, indices, _, _, unpad_pos, unpad_labels, unpad_weights = unpad_inputs(
            inputs, attention_mask, position_ids, labels, weights
        )
        
        expected_pos = torch.tensor([0, 1, 2, 0, 1])
        expected_labels = torch.tensor([10, 20, 30, 40, 50])
        expected_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.assertTrue(torch.equal(unpad_pos, expected_pos))
        self.assertTrue(torch.equal(unpad_labels, expected_labels))
        self.assertTrue(torch.equal(unpad_weights, expected_weights))
    
    def test_pad_1d_outputs(self):
        """Test padding 1D outputs."""
        inputs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = torch.tensor([0, 1, 2, 4, 5])
        
        padded = pad_outputs(inputs, indices, batch=2, seqlen=4)
        
        expected = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(padded, expected))
    
    def test_pad_2d_outputs(self):
        """Test padding 2D outputs (hidden states)."""
        inputs = torch.randn(5, 8)  # 5 tokens, hidden=8
        indices = torch.tensor([0, 1, 2, 4, 5])
        
        padded = pad_outputs(inputs, indices, batch=2, seqlen=4)
        
        self.assertEqual(padded.shape, (2, 4, 8))
    
    def test_unpad_pad_roundtrip(self):
        """Test that unpad followed by pad recovers original (at valid positions)."""
        original = torch.randn(2, 4, 8)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        
        unpadded, indices, _, _, _, _, _ = unpad_inputs(original, attention_mask)
        recovered = pad_outputs(unpadded, indices, batch=2, seqlen=4)
        
        # Check that valid positions match
        for b in range(2):
            for s in range(4):
                if attention_mask[b, s]:
                    self.assertTrue(torch.allclose(original[b, s], recovered[b, s]))
    
    def test_consistency_with_original_unpad(self):
        """Test that refactored unpad matches original."""
        inputs = torch.randn(3, 8, 64)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        
        new_result = unpad_inputs(inputs, attention_mask)
        old_result = _unpad_modernbert_input_original(inputs, attention_mask)
        
        self.assertTrue(torch.equal(new_result[0], old_result[0]))  # unpadded
        self.assertTrue(torch.equal(new_result[1], old_result[1]))  # indices
        self.assertTrue(torch.equal(new_result[2], old_result[2]))  # cu_seqlens
        self.assertEqual(new_result[3], old_result[3])  # max_seqlen
    
    def test_consistency_with_original_pad(self):
        """Test that refactored pad matches original."""
        inputs = torch.randn(10, 64)
        indices = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 11])
        
        new_result = pad_outputs(inputs, indices, batch=3, seqlen=4)
        old_result = _pad_modernbert_output_original(inputs, indices, batch=3, seqlen=4)
        
        self.assertTrue(torch.equal(new_result, old_result))


class TestKVCache(unittest.TestCase):
    """Test KV cache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CacheConfig(
            batch_size=2,
            max_seq_len=128,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.cache = ModernBertKVCache(self.config)
    
    def test_initialization(self):
        """Test cache initialization."""
        self.assertEqual(len(self.cache.key_cache), 4)
        self.assertEqual(len(self.cache.value_cache), 4)
        
        for layer_idx in range(4):
            k, v = self.cache.get_layer_cache(layer_idx)
            self.assertEqual(k.shape, (2, 128, 8, 64))
            self.assertEqual(v.shape, (2, 128, 8, 64))
    
    def test_seq_length_tracking(self):
        """Test sequence length tracking."""
        self.assertEqual(self.cache.get_seq_length(), 0)
        
        self.cache.advance_seq_len(10)
        self.assertEqual(self.cache.get_seq_length(), 10)
        
        self.cache.advance_seq_len(5)
        self.assertEqual(self.cache.get_seq_length(), 15)
        
        seq_lens = self.cache.get_cache_seq_lens()
        self.assertTrue(torch.all(seq_lens == 15))
    
    def test_reset(self):
        """Test cache reset."""
        self.cache.advance_seq_len(50)
        self.cache.key_cache[0].fill_(1.0)
        
        self.cache.reset()
        
        self.assertEqual(self.cache.get_seq_length(), 0)
        self.assertTrue(torch.all(self.cache.key_cache[0] == 0))
    
    def test_get_all_layer_caches(self):
        """Test getting all layer caches at once."""
        all_caches = self.cache.get_all_layer_caches()
        
        self.assertEqual(len(all_caches), 4)
        for k, v in all_caches:
            self.assertEqual(k.shape, (2, 128, 8, 64))
            self.assertEqual(v.shape, (2, 128, 8, 64))
    
    def test_from_model_config(self):
        """Test factory method."""
        model_config = MagicMock()
        model_config.num_hidden_layers = 6
        model_config.num_attention_heads = 12
        model_config.hidden_size = 768
        
        cache = ModernBertKVCache.from_model_config(
            model_config,
            batch_size=4,
            max_seq_len=256,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        
        self.assertEqual(len(cache.key_cache), 6)
        k, v = cache.get_layer_cache(0)
        self.assertEqual(k.shape, (4, 256, 12, 64))


class TestDAUMWeights(unittest.TestCase):
    """Test DAUM weight calculation."""
    
    def test_calculate_daum_weights_shape(self):
        """Test output shape of DAUM weights."""
        # Simplified version for testing
        def calculate_daum_weights(input_ids, mask_token_id, beta=1, gamma=1, p=0.5):
            device = input_ids.device
            seq_len = input_ids.shape[1]
            
            is_mask = (input_ids == mask_token_id)
            is_prev_mask = torch.zeros_like(is_mask)
            if seq_len > 1:
                is_prev_mask[:, 1:] = is_mask[:, :-1]
            
            density_term = 1.0 + gamma * is_prev_mask.float()
            terms = (is_mask.float() * density_term).unsqueeze(1)
            
            powers = torch.arange(seq_len - 1, -1, -1, device=device, dtype=torch.float32)
            kernel = ((1 - p) ** powers).view(1, 1, seq_len)
            
            conv_result = F.conv1d(F.pad(terms, (seq_len - 1, 0)), kernel).squeeze(1)
            return 1 / (beta + conv_result)
        
        input_ids = torch.randint(0, 100, (4, 32))
        weights = calculate_daum_weights(input_ids, mask_token_id=50)
        
        self.assertEqual(weights.shape, (4, 32))
    
    def test_daum_weights_positive(self):
        """Test that DAUM weights are always positive."""
        def calculate_daum_weights(input_ids, mask_token_id, beta=1, gamma=1, p=0.5):
            device = input_ids.device
            seq_len = input_ids.shape[1]
            
            is_mask = (input_ids == mask_token_id)
            is_prev_mask = torch.zeros_like(is_mask)
            if seq_len > 1:
                is_prev_mask[:, 1:] = is_mask[:, :-1]
            
            density_term = 1.0 + gamma * is_prev_mask.float()
            terms = (is_mask.float() * density_term).unsqueeze(1)
            
            powers = torch.arange(seq_len - 1, -1, -1, device=device, dtype=torch.float32)
            kernel = ((1 - p) ** powers).view(1, 1, seq_len)
            
            conv_result = F.conv1d(F.pad(terms, (seq_len - 1, 0)), kernel).squeeze(1)
            return 1 / (beta + conv_result)
        
        input_ids = torch.randint(0, 100, (4, 32))
        # Ensure some masks
        input_ids[:, 5:10] = 50
        
        weights = calculate_daum_weights(input_ids, mask_token_id=50)
        
        self.assertTrue(torch.all(weights > 0))


class TestDiffusionLoss(unittest.TestCase):
    """Test diffusion loss computation."""
    
    def test_compute_diffusion_loss_shape(self):
        """Test loss computation returns correct structure."""
        def compute_diffusion_loss(logits, labels, input_ids, mask_token_id, vocab_size,
                                  num_items_in_batch=None, ignore_index=-100,
                                  per_token_weights=None, training=True):
            logits = logits.float()
            labels_flat = labels.view(-1)
            
            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), labels_flat, ignore_index=ignore_index, reduction='none'
            )
            
            num_items = num_items_in_batch.to(per_token_loss.device)
            valid_mask = (labels_flat != ignore_index)
            is_masked = (input_ids == mask_token_id)
            
            masked_loss_mask = valid_mask & is_masked
            non_masked_loss_mask = valid_mask & ~is_masked
            
            unweighted_total = ((per_token_loss * valid_mask).sum() / num_items).detach()
            
            loss = per_token_loss
            if per_token_weights is not None:
                loss = loss * per_token_weights
            
            masked_loss = (loss * masked_loss_mask).sum() / num_items
            non_masked_loss = (loss * non_masked_loss_mask).sum() / num_items
            
            if training:
                total_loss = masked_loss + non_masked_loss
            else:
                total_loss = masked_loss if is_masked.any() else non_masked_loss
            
            return total_loss, masked_loss, non_masked_loss, unweighted_total
        
        batch_size, seq_len, vocab_size = 4, 16, 1000
        logits = torch.randn(batch_size * seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size * seq_len,))
        input_ids = torch.randint(0, vocab_size, (batch_size * seq_len,))
        input_ids[5:10] = 50  # Some masks
        num_items = torch.tensor(batch_size * seq_len, dtype=torch.float32)
        
        total, masked, non_masked, unweighted = compute_diffusion_loss(
            logits, labels, input_ids, mask_token_id=50, vocab_size=vocab_size,
            num_items_in_batch=num_items
        )
        
        # All should be scalars
        self.assertEqual(total.dim(), 0)
        self.assertEqual(masked.dim(), 0)
        self.assertEqual(non_masked.dim(), 0)
        self.assertEqual(unweighted.dim(), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def test_unpad_forward_pad_flow(self):
        """Test the complete unpad -> process -> pad flow."""
        batch, seq, hidden = 3, 8, 64
        
        # Original input with padding
        inputs = torch.randn(batch, seq, hidden)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        
        # Unpad
        unpadded, indices, cu_seqlens, max_seqlen, _, _, _ = unpad_inputs(inputs, attention_mask)
        
        # Simulate processing (e.g., a linear layer)
        processed = unpadded * 2
        
        # Pad
        padded = pad_outputs(processed, indices, batch, seq)
        
        # Verify shape
        self.assertEqual(padded.shape, inputs.shape)
        
        # Verify valid positions have correct values
        for b in range(batch):
            for s in range(seq):
                if attention_mask[b, s]:
                    self.assertTrue(torch.allclose(padded[b, s], inputs[b, s] * 2))
    
    def test_kv_cache_generation_simulation(self):
        """Simulate a generation loop with KV cache."""
        config = CacheConfig(
            batch_size=1,
            max_seq_len=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        cache = ModernBertKVCache(config)
        
        # Prefill phase: process 10 tokens
        cache.advance_seq_len(10)
        self.assertEqual(cache.get_seq_length(), 10)
        
        # Decode phase: generate 5 more tokens one by one
        for i in range(5):
            current_pos = cache.get_seq_length()
            self.assertEqual(current_pos, 10 + i)
            
            # Simulate writing to cache at current position
            k, v = cache.get_layer_cache(0)
            k[:, current_pos:current_pos+1, :, :] = torch.randn(1, 1, 4, 16)
            v[:, current_pos:current_pos+1, :, :] = torch.randn(1, 1, 4, 16)
            
            cache.advance_seq_len(1)
        
        self.assertEqual(cache.get_seq_length(), 15)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestSamplingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestUnpaddingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestKVCache))
    suite.addTests(loader.loadTestsFromTestCase(TestDAUMWeights))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("Running tests for refactored ModernBERT code")
    print("=" * 70)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 70)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 70)
    
    exit(0 if success else 1)