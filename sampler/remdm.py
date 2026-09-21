# samplers/remdm_sampler.py
import torch
import torch.nn.functional as F
from .base import BaseSampler

class ReMDMSampler(BaseSampler):
    """
    Implements the ReMasking Diffusion Model (ReMDM) sampler.

    This sampler allows for iterative refinement by potentially remasking tokens
    that have already been unmasked, correcting errors during generation.
    """
    def __init__(self, model, noise_scheduler, mask_token_id: int,
                 schedule_strategy: str = 'max-capped', **strategy_kwargs):
        """
        Args:
            model: The denoising model (predicts x_0 from z_t).
            noise_scheduler: The noise schedule manager.
            mask_token_id (int): The integer ID for the [MASK] token.
            schedule_strategy (str): The strategy for the remasking schedule sigma_t.
                                     One of ['max-capped', 'rescaled'].
            **strategy_kwargs: Hyperparameters for the chosen strategy.
                               e.g., eta_cap for 'max-capped', eta_rescale for 'rescaled'.
        """
        super().__init__(model, noise_scheduler)
        self.mask_token_id = mask_token_id
        self.schedule_strategy = schedule_strategy
        self.strategy_kwargs = strategy_kwargs

    def _get_sigma_t(self, alpha_t: torch.Tensor, alpha_s: torch.Tensor) -> torch.Tensor:
        """
        Calculates the remasking probability sigma_t based on the chosen strategy.
        Corresponds to Section 4.1 in the paper.
        """
        # Calculate the maximum allowed value for sigma_t (Eq. 7)
        sigma_t_max = torch.minimum(torch.tensor(1.0), (1 - alpha_s) / alpha_t)

        if self.schedule_strategy == 'max-capped':
            eta_cap = self.strategy_kwargs.get('eta_cap', 0.02)
            sigma_t = torch.minimum(torch.tensor(eta_cap), sigma_t_max)
        elif self.schedule_strategy == 'rescaled':
            eta_rescale = self.strategy_kwargs.get('eta_rescale', 0.9)
            sigma_t = eta_rescale * sigma_t_max
        else:
            # Default to no remasking (recovers standard MDLM)
            sigma_t = 0.0

        return sigma_t

    @torch.no_grad()
    def step(self, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Performs one step of ReMDM sampling, implementing Algorithm 1.
        """
        batch_size, seq_len = z_t.shape
        device = z_t.device

        # 1. Get schedule values and predict x_0
        alpha_t = self.noise_scheduler.get_alpha(t).to(device)
        alpha_s = self.noise_scheduler.get_alpha(s).to(device)
        
        # The model predicts the clean data x_0 from the noisy z_t
        # Note: The model should be trained to handle one-hot inputs or token IDs
        # and output logits over the vocabulary.
        x_0_logits = self.model(z_t, t)
        x_0_probs = F.softmax(x_0_logits, dim=-1)

        # 2. Calculate remasking schedule sigma_t
        sigma_t = self._get_sigma_t(alpha_t, alpha_s)

        # 3. Compute the approximate posterior p_theta(z_s | z_t)
        # This involves two cases based on whether z_t is masked or not.
        
        # Create masks for the two cases
        is_masked = (z_t == self.mask_token_id)
        is_unmasked = ~is_masked

        # --- Case 1: z_t is unmasked (z_t != m) ---
        # The token can either be kept or remasked.
        # Prob to keep is (1 - sigma_t), prob to remask is sigma_t
        probs_unmasked = torch.zeros(batch_size, seq_len, device=device)
        probs_unmasked[is_unmasked] = sigma_t
        
        # Sample the remasking decision
        remask_decision = torch.bernoulli(probs_unmasked).bool()
        
        # --- Case 2: z_t is masked (z_t == m) ---
        # The token can be decoded or stay masked.
        # This is the standard denoising part, modified by sigma_t.
        # Coeffs from Algorithm 1 in the paper.
        coeff1 = (alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t)
        coeff2 = (1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t)

        # Probabilities for new tokens come from the model's prediction
        probs_from_model = coeff1.view(-1, 1, 1) * x_0_probs
        
        # Probability of staying masked
        mask_probs = torch.full_like(x_0_probs, 0)
        mask_probs[..., self.mask_token_id] = coeff2.view(-1, 1, 1)
        
        # Combine probabilities for the masked case
        final_probs_masked = probs_from_model + mask_probs
        
        # Sample new tokens for the masked positions
        dist_masked = torch.distributions.Categorical(probs=final_probs_masked)
        sampled_tokens_for_masked = dist_masked.sample()

        # 4. Construct z_s by combining the results
        z_s = torch.zeros_like(z_t)
        
        # Apply Case 1 results: keep original token unless remasked
        z_s[is_unmasked] = torch.where(remask_decision[is_unmasked], self.mask_token_id, z_t[is_unmasked])
        
        # Apply Case 2 results: fill in newly sampled tokens
        z_s[is_masked] = sampled_tokens_for_masked[is_masked]

        return z_s