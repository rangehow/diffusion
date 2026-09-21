## Response to Reviewer mkMw

We sincerely thank the reviewer for the constructive evaluation and clear articulation of concerns. We address each weakness and question below.

### W1: Accuracy Degradation at Aggressive Acceleration

We agree that aggressive acceleration ($4\times$) causes quality degradation, as transparently reported in Appendix D. However, we emphasize this is a continuous, user-controllable trade-off—not a binary failure. At the recommended $1.7\times$ setting, Gen PPL is 12.65 vs. ARM's 11.19 (Table 5), a gap of only 1.46 PPL. Per-model breakdown confirms consistency:

| Reference Model | ARM | CARD($1.7\times$) | CARD($2.8\times$) | CARD($4.0\times$) |
|---|---|---|---|---|
| Qwen3-8B | 9.79 | 10.97 | 11.89 | 15.39 |
| Gemma-3-27B | 12.24 | 13.73 | 14.90 | 19.34 |

The degradation at $4\times$ (text looping) stems from the hard step limit forcing non-autoregressive decoding of remaining masks, a fundamental limitation shared by all parallel decoding methods (Fast-dLLM, BD3LM). Crucially, CARD's dynamic parallelism lets users tune $(K, \tau, T_{\max})$ per-deployment, unlike BD3LM's fixed block size. We anticipate stronger base models and improved sampling strategies (e.g., remasking) will further alleviate this.

### W2: Underperformance vs. ARM on Downstream Tasks

We respectfully note that CARD already **significantly narrows** the dLLM–ARM gap. MMLU and WinoGrande cluster near chance for *all* models at 1B scale (ARM included: 25.45/55.96). Excluding these two noise-dominated benchmarks:

| Model | ARC-C | ARC-E | CSQA | HellaSwag | PIQA | SciQ | 6-Task AVG |
|---|---|---|---|---|---|---|---|
| ARM | 34.04 | 64.65 | 52.74 | 61.26 | 75.95 | 81.10 | **61.62** |
| **CARD** | 32.68 | 60.77 | 48.73 | 53.29 | 71.71 | 79.80 | **57.83** |
| BD3LM | 27.30 | 48.06 | 44.06 | 42.48 | 59.79 | 79.60 | 50.22 |
| MDLM | 29.44 | 49.16 | 36.45 | 48.32 | 59.63 | 76.60 | 49.93 |

The ARM gap narrows to **3.8 points** while CARD leads the best dLLM baseline by **+7.6 points**. More importantly, CARD is the **first dLLM to surpass ARM on language modeling PPL** (34.40 vs. 38.68 across 8 domains, Table 2), demonstrating superior generative capacity. The remaining downstream gap reflects a modest and expected tax for enabling parallel decoding—a trade-off we believe is worthwhile given CARD's $1.7\times$–$4\times$ inference speedup.

### W3: Limited Scale (1B)

We chose 1B/300B tokens as a controlled testbed where all models (ARM, MDLM, BD3LM, CARD) use identical architecture, data, and hyperparameters—the standard practice in foundational method papers (MDLM@NeurIPS'24, BD3LM@ICLR'25 also evaluate at $\leq$1B). Comparing with LLaDA-8B (trained on multi-trillion tokens) or Dream-7B would conflate the effect of our *method* with differences in data scale, architecture family, and training budget.

That said, CARD's key advantage—ARM-equivalent training cost—means scaling to larger sizes is straightforward: CARD uses the same standard causal attention, the same optimizers, and the same training infrastructure as ARM, with no block duplication overhead (unlike BD3LM at $3\times$) or bidirectional attention (unlike MDLM at $1.5\times$). We view our 1B results as a rigorous proof-of-concept that establishes the methodology; scaling is an engineering effort rather than a research question.

### W4: Comparison with LLaDA, Dream, d1, Fast-dLLM v2

These models occupy different categories from CARD:

- **LLaDA & Dream** are standard MDLMs differing only in training data, initialization, and scale. Our MDLM baseline already represents this class under controlled conditions. Comparing CARD-1B with LLaDA-8B would be comparing *scales*, not *methods*.
- **d1** is a *post-training* framework (SFT + RL via diffu-GRPO) applied on top of a pre-trained LLaDA-8B. It addresses reasoning via reinforcement learning, which is orthogonal to CARD's contribution on pre-training paradigm design. CARD's causal formulation could serve as d1's backbone in future work.
- **Fast-dLLM v2** is an *inference-time sampler* (training-free KV caching + parallel decoding for existing bidirectional dLLMs). It is orthogonal and complementary to CARD—CARD achieves KV caching natively through its causal architecture without requiring the approximations Fast-dLLM introduces.

A fair comparison would require re-training all methods at matched scale and data, which we have done at 1B.

### Q1 & Q2: Acceleration Control and Parallel Degree

CARD provides three user-tunable knobs: block size $K$, confidence threshold $\tau$, and step limit $T_{\max}$. In practice, we recommend starting with $K=16$, $T_{\max}=16$, $\tau=0.9$ for quality-preserving acceleration ($1.7\times$), and increasing $K$ or reducing $T_{\max}$ for latency-critical applications. The confidence threshold $\tau$ provides a natural quality gate: when the model is uncertain, it decodes conservatively ($\approx$ autoregressive); when confident, it parallelizes aggressively. This adaptive behavior is demonstrated in Table 5.

### Q3: Improving Zero/Few-Shot Performance

Two complementary paths exist: (1) **Scaling**—CARD's ARM-equivalent training cost makes scaling to 3B+ straightforward; (2) **Data efficiency**—Figure 4 shows CARD surpasses ARM at epoch 11 under data repetition, suggesting CARD extracts more signal per token in data-constrained regimes. Both paths directly improve downstream performance without algorithmic changes.

---

*We commit to incorporating the 6-task analysis, per-model Gen PPL breakdown, and clarified comparisons with concurrent works in the revision. We hope these clarifications address the reviewer's concerns.*
