# samplers/mdlm_sampler.py
import torch
import torch.nn.functional as F
from .base import BaseSampler

class MDLMSampler(BaseSampler):
    """
    Implements a standard Masked Diffusion Language Model (MDLM) sampler.
    
    This sampler denoises from z_t to z_s by predicting x_0 and using the
    standard posterior q(z_s | z_t, x_0). It does not support remasking.
    This is the process that EB-Sampler is designed to accelerate.
    """
    def __init__(self, model, noise_scheduler, mask_token_id: int):
        super().__init__(model, noise_scheduler)
        self.mask_token_id = mask_token_id

    @torch.no_grad()
    def get_posterior_probs(self, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Calculates the posterior probability p(z_s | z_t) for the MDLM process.
        """
        device = z_t.device
        
        # Get schedule values and predict x_0
        alpha_t = self.noise_scheduler.get_alpha(t).to(device)
        alpha_s = self.noise_scheduler.get_alpha(s).to(device)

        x_0_logits = self.model(z_t, t)
        x_0_probs = F.softmax(x_0_logits, dim=-1)

        # Coefficients for the posterior q(z_s | z_t, x_0)
        # This is the posterior from the ReMDM paper with sigma_t = 0.
        coeff1 = (alpha_s - alpha_t) / (1 - alpha_t)
        coeff2 = (1 - alpha_s) / (1 - alpha_t)
        
        # Probabilities for new tokens from the model's prediction
        probs_from_model = coeff1.view(-1, 1, 1) * x_0_probs
        
        # Probability of staying masked
        mask_probs = torch.full_like(x_0_probs, 0)
        mask_probs[..., self.mask_token_id] = coeff2.view(-1, 1, 1)
        
        final_probs = probs_from_model + mask_probs
        return final_probs

    @torch.no_grad()
    def step(self, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Performs a single step, but this is less relevant for EB-Sampler.
        We will primarily use get_posterior_probs.
        """
        # This method would be used for a simple, random-order MDLM sampler.
        # EB-Sampler provides a more intelligent way to select which tokens to update.
        posterior_probs = self.get_posterior_probs(z_t, t, s)
        
        # For a basic sampler, one might just sample all masked positions
        is_masked = (z_t == self.mask_token_id)
        dist = torch.distributions.Categorical(probs=posterior_probs)
        new_tokens = dist.sample()
        
        z_s = z_t.clone()
        z_s[is_masked] = new_tokens[is_masked]
        return z_s