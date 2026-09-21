# CARD Ablation Experiments

This directory contains all ablation experiment code addressing reviewer concerns.

## Ablation 1: Diffusion Framing vs. Tail-Biased Masking + DAUM (Reviewer 2Bg4)
- **Goal**: Isolate how much gain comes from the diffusion framing itself vs. the tail-biased denoising + local loss weighting (DAUM).
- **Variants**:
  - `full_card`: CARD full model (diffusion + tail-bias + DAUM)
  - `no_daum`: CARD without DAUM weighting
  - `no_tail_bias`: CARD without tail-bias (uniform masking, like MDLM, but causal)
  - `no_both`: CARD without tail-bias AND without DAUM (minimal diffusion, causal mask only)
  - `prefix_lm`: Prefix-LM baseline (random prefix, predict suffix autoregressively — no diffusion)
- **Script**: `ablation/train_ablation1.sh`, `ablation/collator_ablation.py`

## Ablation 2: Sensitivity to Diffusion Steps (Reviewer YsYm)
- **Goal**: Measure how generation quality (PPL) changes with number of diffusion steps at inference.
- **Script**: `ablation/sweep_steps.sh`

## Ablation 3: Block Size vs. Quality Trade-off (Reviewer mkMw)
- **Goal**: Find optimal parallel degree (block size) balancing speed and quality.
- **Script**: `ablation/sweep_blocksize.sh`

## Ablation 4: Efficiency Comparison — Wall-Clock & FLOPs (Reviewer Ruxi)
- **Goal**: Controlled wall-clock timing comparison (tokens/sec) across CARD, AR, and baselines.
- **Script**: `ablation/benchmark_efficiency.py`

## Ablation 5: Tail-Bias Factor Sensitivity
- **Goal**: How does tail_bias_factor affect final model quality?
- **Script**: `ablation/train_tail_bias_sweep.sh`

## Running all ablations
```bash
# 1. Train all ablation variants
bash ablation/train_ablation1.sh

# 2. After training, run inference sweeps
bash ablation/sweep_steps.sh
bash ablation/sweep_blocksize.sh

# 3. Run efficiency benchmarks
python -m ablation.benchmark_efficiency

# 4. Aggregate results
python -m ablation.aggregate_results
```
