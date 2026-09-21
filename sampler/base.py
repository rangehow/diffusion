# samplers/base.py
from abc import ABC, abstractmethod
import torch

class BaseSampler(ABC):
    """
    Abstract Base Class for all diffusion samplers.
    It defines the common interface for performing one step of the reverse process.
    """
    def __init__(self, model, noise_scheduler, **kwargs):
        self.model = model
        self.noise_scheduler = noise_scheduler

    @abstractmethod
    @torch.no_grad()
    def step(self, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Performs a single reverse diffusion step from time t to s.

        Args:
            z_t (torch.Tensor): The current noisy state (sequence of tokens) at time t.
                                Shape: (batch_size, seq_len).
            t (torch.Tensor): The current timestep, a scalar tensor for the whole batch.
            s (torch.Tensor): The next timestep to denoise to, a scalar tensor.

        Returns:
            torch.Tensor: The denoised state z_s at time s.
        """
        pass