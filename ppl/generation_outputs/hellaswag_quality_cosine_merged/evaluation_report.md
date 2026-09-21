# Generation Evaluation Report

*Generated: 2026-01-28 15:02:54*

## Summary Table

| Method | PPL (Qwen3-8B-Base) | PPL (SmolLM3-3B-Base) | PPL (gemma-3-27b-pt) | PPL (gpt2-large) | Dist-1 | Dist-2 | Rep-2 | Rep-3 | SeqRep-2 | AvgLen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 6.30 | 6.47 | 7.88 | 8.15 | 0.0446 | 0.2716 | 0.2686 | 0.1963 | 0.0129 | 89.3 |
| quality_cosine_b16_s16 | 4.15 | 4.33 | 5.08 | 5.22 | 0.0338 | 0.1898 | 0.5127 | 0.4548 | 0.0551 | 91.3 |
| quality_cosine_b16_s8 | 15.32 | 17.82 | 18.47 | 26.83 | 0.0747 | 0.2750 | 0.4304 | 0.3350 | 0.2112 | 84.0 |
| quality_cosine_b32_s8 | 16.45 | 19.44 | 20.90 | 30.11 | 0.0744 | 0.2324 | 0.4619 | 0.3891 | 0.3309 | 79.1 |
| quality_cosine_b64_s4 | 18.95 | 22.51 | 23.92 | 35.72 | 0.0764 | 0.2302 | 0.4783 | 0.4035 | 0.3717 | 79.2 |

## Metric Interpretation

| Metric | Description | Optimal |
| --- | --- | --- |
| PPL | Fluency/naturalness (lower = more predictable) | Lower, but watch for repetition |
| Dist-1/2 | Vocabulary diversity (unique n-grams ratio) | Higher |
| Rep-2/3 | N-gram repetition rate | Lower |
| SeqRep-2 | Consecutive repetition (stuttering) | Lower |

## Quality-Diversity Trade-off

```
                    High Diversity
                          │
            ┌─────────────┼─────────────┐
            │   Diverse   │   Ideal     │
            │  but noisy  │  (target)   │
  High PPL ─┼─────────────┼─────────────┼─ Low PPL
            │    Bad      │  Repetitive │
            │  (garbage)  │  but fluent │
            └─────────────┼─────────────┘
                          │
                    Low Diversity
```

## Per-Method Analysis

### llama

**PPL (Qwen3-8B-Base):** 6.30 ± 2.90
**PPL (SmolLM3-3B-Base):** 6.47 ± 2.96
**PPL (gemma-3-27b-pt):** 7.88 ± 5.03
**PPL (gpt2-large):** 8.15 ± 4.05

**Diversity Metrics:**
- Corpus Distinct-1/2/3: 0.0446 / 0.2716 / 0.5600
- Rep-2/3/4: 0.2686 / 0.1963 / 0.1573
- Vocabulary Size: 39984, Entropy: 10.04 bits

### quality_cosine_b16_s16

**PPL (Qwen3-8B-Base):** 4.15 ± 1.27
**PPL (SmolLM3-3B-Base):** 4.33 ± 1.33
**PPL (gemma-3-27b-pt):** 5.08 ± 1.50
**PPL (gpt2-large):** 5.22 ± 1.93

**Diversity Metrics:**
- Corpus Distinct-1/2/3: 0.0338 / 0.1898 / 0.3788
- Rep-2/3/4: 0.5127 / 0.4548 / 0.4155
- Vocabulary Size: 30955, Entropy: 9.74 bits

### quality_cosine_b16_s8

**PPL (Qwen3-8B-Base):** 15.32 ± 11.01
**PPL (SmolLM3-3B-Base):** 17.82 ± 13.76
**PPL (gemma-3-27b-pt):** 18.47 ± 15.59
**PPL (gpt2-large):** 26.83 ± 19.13

**Diversity Metrics:**
- Corpus Distinct-1/2/3: 0.0747 / 0.2750 / 0.4651
- Rep-2/3/4: 0.4304 / 0.3350 / 0.2776
- Vocabulary Size: 63015, Entropy: 9.84 bits

### quality_cosine_b32_s8

**PPL (Qwen3-8B-Base):** 16.45 ± 18.08
**PPL (SmolLM3-3B-Base):** 19.44 ± 22.39
**PPL (gemma-3-27b-pt):** 20.90 ± 30.52
**PPL (gpt2-large):** 30.11 ± 31.11

**Diversity Metrics:**
- Corpus Distinct-1/2/3: 0.0744 / 0.2324 / 0.3639
- Rep-2/3/4: 0.4619 / 0.3891 / 0.3424
- Vocabulary Size: 59117, Entropy: 8.95 bits

### quality_cosine_b64_s4

**PPL (Qwen3-8B-Base):** 18.95 ± 19.36
**PPL (SmolLM3-3B-Base):** 22.51 ± 24.04
**PPL (gemma-3-27b-pt):** 23.92 ± 28.70
**PPL (gpt2-large):** 35.72 ± 35.22

**Diversity Metrics:**
- Corpus Distinct-1/2/3: 0.0764 / 0.2302 / 0.3555
- Rep-2/3/4: 0.4783 / 0.4035 / 0.3554
- Vocabulary Size: 60720, Entropy: 8.44 bits
