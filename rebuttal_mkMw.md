## Response to Reviewer mkMw

We appreciate the reviewer's detailed feedback and address each concern below.

### W1: Accuracy Degradation at Aggressive Acceleration

Yes, the $4\times$ setting does degrade quality, which we reported transparently in Appendix D. That said, this is a smooth, user-controllable knob rather than a binary failure mode. At $1.7\times$, Gen PPL is 12.65 vs. ARM's 11.19 (a gap of only 1.46). The per-model breakdown also confirms this:

| Reference Model | ARM | CARD($1.7\times$) | CARD($2.8\times$) | CARD($4.0\times$) |
|---|---|---|---|---|
| Qwen3-8B | 9.79 | 10.97 | 11.89 | 15.39 |
| Gemma-3-27B | 12.24 | 13.73 | 14.90 | 19.34 |

The looping behavior at $4\times$ happens because the hard step limit forces greedy decoding of whatever masks remain, a limitation that all parallel decoding methods share. What distinguishes CARD is that users can freely adjust $(K, \tau, T_{\max})$ for each deployment scenario, whereas BD3LM is locked to a fixed block size chosen at training time. We also expect that stronger base models and better sampling strategies (e.g., remasking from Wang et al., 2025) will reduce this sensitivity.

### W2: Underperformance vs. ARM on Downstream Tasks

CARD already closes most of the gap between dLLMs and ARM. Note that MMLU and WinoGrande hover near chance level for *all* 1B models including ARM (25.45 and 55.96), so they contribute mostly noise. If we focus on the six benchmarks where models actually differentiate:

| Model | ARC-C | ARC-E | CSQA | HellaSwag | PIQA | SciQ | 6-Task AVG |
|---|---|---|---|---|---|---|---|
| ARM | 34.04 | 64.65 | 52.74 | 61.26 | 75.95 | 81.10 | **61.62** |
| **CARD** | 32.68 | 60.77 | 48.73 | 53.29 | 71.71 | 79.80 | **57.83** |
| BD3LM | 27.30 | 48.06 | 44.06 | 42.48 | 59.79 | 79.60 | 50.22 |
| MDLM | 29.44 | 49.16 | 36.45 | 48.32 | 59.63 | 76.60 | 49.93 |

The gap to ARM shrinks to **3.8 points**, while CARD leads the best prior dLLM by **+7.6**. On the PPL side, CARD actually **beats** ARM (34.40 vs. 38.68 across 8 domains, Table 2), which is a first for any dLLM to our knowledge. We view the small remaining downstream gap as a reasonable price for $1.7\times$ to $4\times$ parallel decoding.

### W3: Limited Scale (1B)

Our goal was a controlled comparison where every model shares the same architecture, data, and hyperparameters. 1B/300B tokens is the standard setup in recent method papers (MDLM at NeurIPS'24 and BD3LM at ICLR'25 both stay at $\leq$1B). Comparing CARD against LLaDA-8B or Dream-7B would mix up the effect of our *training paradigm* with differences in data volume, tokenizer, and model family.

On the practical side, because CARD uses standard causal attention with no block duplication ($3\times$ overhead in BD3LM) or bidirectional attention ($1.5\times$ in MDLM), it scales exactly like ARM. Extending to 3B or 8B is an engineering effort, not a research question, and we plan to release larger checkpoints.

### W4: Comparison with LLaDA, Dream, d1, Fast-dLLM v2

These systems each belong to a different category from CARD:

**LLaDA & Dream** are standard MDLMs; the main differences are training data and scale. Our MDLM baseline already represents this model family under matched conditions. Putting CARD-1B next to LLaDA-8B would be comparing *scales*, not *methods*.

**d1** applies post-training (SFT + RL via diffu-GRPO) on top of a pretrained LLaDA-8B. It tackles reasoning through reinforcement learning, which is orthogonal to the pretraining paradigm we propose. CARD could serve as d1's backbone in future work.

**Fast-dLLM v2** is a training-free inference sampler that retrofits KV caching onto bidirectional dLLMs. CARD achieves KV caching natively through its causal design without the approximations Fast-dLLM introduces, so the two approaches are complementary rather than competing.

### Q1 & Q2: Acceleration Control

CARD exposes three knobs: block size $K$, confidence threshold $\tau$, and step limit $T_{\max}$. Our recommended starting point is $K{=}16, T_{\max}{=}16, \tau{=}0.9$, which gives $1.7\times$ speedup with near-ARM quality. For latency-critical serving, users can increase $K$ or lower $T_{\max}$. The threshold $\tau$ acts as a natural quality gate: when the model is uncertain it falls back to conservative, nearly autoregressive decoding; when confident it decodes multiple tokens at once.

### Q3: Improving Zero/Few-Shot Performance

Two concrete paths: (1) Scaling up, which is straightforward given CARD's ARM-equivalent training cost; (2) Exploiting CARD's data efficiency advantage (Figure 4 shows CARD surpasses ARM at epoch 11 under data repetition, continuing to extract useful signal where ARM saturates). Both improve downstream accuracy without any algorithmic change to CARD itself.

*We will incorporate the 6-task analysis, per-model Gen PPL breakdown, and the clarified positioning relative to concurrent work in the camera-ready revision.*
