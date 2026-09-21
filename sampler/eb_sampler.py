# samplers/eb_sampler.py
import torch
import torch.nn.functional as F
from typing import List, Literal

ErrorProxyType = Literal['entropy', 'confidence', 'margin']

class EBSampler:
    """
    Implements the Entropy Bounded Sampler (EB-Sampler).

    This is an efficiency sampler that dynamically decides which and how many
    tokens to unmask in a single step to accelerate generation.
    """
    def __init__(self, gamma: float, error_proxy: ErrorProxyType = 'entropy'):
        if error_proxy not in ['entropy', 'confidence', 'margin']:
            raise ValueError("error_proxy must be 'entropy', 'confidence', or 'margin'")
        self.gamma = gamma
        self.error_proxy_type = error_proxy

    def _calculate_error_proxies(self, probs: torch.Tensor) -> torch.Tensor:
        if self.error_proxy_type == 'entropy':
            return torch.distributions.Categorical(probs=probs).entropy()
        elif self.error_proxy_type == 'confidence':
            return -torch.max(probs, dim=-1).values
        elif self.error_proxy_type == 'margin':
            top2_probs = torch.topk(probs, 2, dim=-1).values
            if top2_probs.shape[-1] == 1:
                return -top2_probs[..., 0]
            return -(top2_probs[..., 0] - top2_probs[..., 1])

    @torch.no_grad()
    def plan_step(self, logits: torch.Tensor, masked_tokens_mask: torch.BoolTensor) -> List[torch.Tensor]:
        """
        Determines which tokens to unmask in the current step.
        This is the core planning logic of the EB-Sampler.

        Args:
            logits (torch.Tensor): Raw model output logits. Shape: (batch_size, seq_len, vocab_size).
            masked_tokens_mask (torch.BoolTensor): Mask of currently masked tokens. Shape: (batch_size, seq_len).

        Returns:
            List[torch.Tensor]: A list where each tensor contains the indices to unmask for a batch item.
        """
        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=-1)
        
        entropies = torch.distributions.Categorical(probs=probs).entropy()
        error_proxy = self._calculate_error_proxies(probs)

        error_proxy[~masked_tokens_mask] = float('inf')
        _, sorted_indices = torch.sort(error_proxy, dim=-1)

        sorted_entropies = torch.gather(entropies, 1, sorted_indices)
        
        acc_entropies = torch.cumsum(sorted_entropies, dim=-1)
        cummax_entropies = torch.cummax(sorted_entropies, dim=-1).values

        num_to_unmask = torch.sum(acc_entropies - cummax_entropies <= self.gamma, dim=-1)

        num_masked = masked_tokens_mask.sum(dim=-1)
        num_to_unmask = torch.minimum(num_to_unmask, num_masked)

        indices_to_unmask_list = []
        for i in range(batch_size):
            k = num_to_unmask[i].item()
            final_indices = sorted_indices[i, :k]
            indices_to_unmask_list.append(final_indices)
            
        return indices_to_unmask_list