"""
Diversity Metrics Calculator (Distinct-n, Rep-n)
Evaluates generation diversity without requiring reference text or GPU.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


# ============================================================================
# N-GRAM UTILITIES
# ============================================================================

def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def tokenize_text(text: str, method: str = "char") -> List[str]:
    """
    Tokenize text using specified method.

    Methods:
        - char: character-level (recommended for Chinese)
        - word: whitespace split (recommended for English)
    """
    if method == "char":
        return [c for c in text if not c.isspace()]
    elif method == "word":
        return text.split()
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_distinct_n_corpus(all_tokens: List[List[str]], n: int) -> float:
    """
    Calculate corpus-level Distinct-n.
    Distinct-n = unique n-grams / total n-grams (across all texts)

    Higher = more diverse vocabulary usage across the corpus.
    """
    all_ngrams = []
    for tokens in all_tokens:
        all_ngrams.extend(get_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def calculate_distinct_n_sample(tokens: List[str], n: int) -> float:
    """Calculate Distinct-n for a single sample."""
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def calculate_rep_n(tokens: List[str], n: int) -> float:
    """
    Calculate Rep-n for a single sample.
    Rep-n = 1 - (unique n-grams / total n-grams)

    Lower = less repetition = better.
    """
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def calculate_seq_rep_n(tokens: List[str], n: int) -> float:
    """
    Calculate sequential repetition (repeated consecutive n-grams).
    Measures "stuttering" patterns like "I think I think I think".
    """
    ngrams = get_ngrams(tokens, n)
    if len(ngrams) < 2:
        return 0.0

    repeated = sum(1 for i in range(1, len(ngrams)) if ngrams[i] == ngrams[i-1])
    return repeated / (len(ngrams) - 1)


def calculate_metrics_single(
    text: str,
    tokenize_method: str,
) -> Dict:
    """Calculate all metrics for a single text."""
    tokens = tokenize_text(text, tokenize_method)

    if len(tokens) == 0:
        return None

    return {
        "tokens": tokens,
        "length": len(tokens),
        "distinct_1": calculate_distinct_n_sample(tokens, 1),
        "distinct_2": calculate_distinct_n_sample(tokens, 2),
        "distinct_3": calculate_distinct_n_sample(tokens, 3),
        "rep_2": calculate_rep_n(tokens, 2),
        "rep_3": calculate_rep_n(tokens, 3),
        "rep_4": calculate_rep_n(tokens, 4),
        "seq_rep_2": calculate_seq_rep_n(tokens, 2),
        "seq_rep_3": calculate_seq_rep_n(tokens, 3),
    }


def calculate_metrics_for_texts(
    texts: List[str],
    tokenize_method: str = "char",
    num_workers: int = 1,
    show_progress: bool = True,
) -> Dict:
    """
    Calculate all diversity metrics for a list of texts.

    Returns both corpus-level and averaged sample-level metrics.
    """
    # Tokenize and calculate per-sample metrics
    if show_progress:
        print("  Tokenizing and calculating per-sample metrics...")

    if num_workers > 1:
        calc_fn = partial(calculate_metrics_single, tokenize_method=tokenize_method)
        with mp.Pool(num_workers) as pool:
            sample_results = list(tqdm(
                pool.imap(calc_fn, texts),
                total=len(texts),
                desc="  Processing",
                disable=not show_progress
            ))
    else:
        sample_results = [
            calculate_metrics_single(text, tokenize_method)
            for text in tqdm(texts, desc="  Processing", disable=not show_progress)
        ]

    # Filter out None results (empty texts)
    sample_results = [r for r in sample_results if r is not None]

    if not sample_results:
        return {
            "corpus_distinct_1": 0.0,
            "corpus_distinct_2": 0.0,
            "corpus_distinct_3": 0.0,
            "sample_distinct_1": 0.0,
            "sample_distinct_2": 0.0,
            "sample_distinct_3": 0.0,
            "rep_2": 0.0,
            "rep_3": 0.0,
            "rep_4": 0.0,
            "seq_rep_2": 0.0,
            "seq_rep_3": 0.0,
            "avg_length": 0.0,
            "num_samples": 0,
        }

    all_tokens = [r["tokens"] for r in sample_results]
    n_samples = len(sample_results)

    # Corpus-level Distinct-n (across all texts)
    if show_progress:
        print("  Calculating corpus-level Distinct-n...")

    corpus_distinct_1 = calculate_distinct_n_corpus(all_tokens, 1)
    corpus_distinct_2 = calculate_distinct_n_corpus(all_tokens, 2)
    corpus_distinct_3 = calculate_distinct_n_corpus(all_tokens, 3)

    # Sample-level averages
    def avg(key):
        return sum(r[key] for r in sample_results) / n_samples

    def std(key):
        mean = avg(key)
        return (sum((r[key] - mean) ** 2 for r in sample_results) / n_samples) ** 0.5

    # Vocabulary statistics
    all_unigrams = []
    for tokens in all_tokens:
        all_unigrams.extend(tokens)

    vocab_size = len(set(all_unigrams))
    total_tokens = len(all_unigrams)

    # Token frequency for entropy calculation
    token_counts = Counter(all_unigrams)
    entropy = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        entropy -= p * (p > 0 and __import__('math').log2(p) or 0)

    return {
        # Corpus-level (measures diversity across all outputs)
        "corpus_distinct_1": corpus_distinct_1,
        "corpus_distinct_2": corpus_distinct_2,
        "corpus_distinct_3": corpus_distinct_3,

        # Sample-level averages (measures diversity within each output)
        "sample_distinct_1": avg("distinct_1"),
        "sample_distinct_2": avg("distinct_2"),
        "sample_distinct_3": avg("distinct_3"),

        # Repetition metrics (lower is better)
        "rep_2": avg("rep_2"),
        "rep_2_std": std("rep_2"),
        "rep_3": avg("rep_3"),
        "rep_3_std": std("rep_3"),
        "rep_4": avg("rep_4"),
        "rep_4_std": std("rep_4"),

        # Sequential repetition (lower is better)
        "seq_rep_2": avg("seq_rep_2"),
        "seq_rep_3": avg("seq_rep_3"),

        # Statistics
        "avg_length": avg("length"),
        "length_std": std("length"),
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "token_entropy": entropy,
        "num_samples": n_samples,
    }


# ============================================================================
# DATA HELPERS
# ============================================================================

def load_generations(gen_dir: str) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Load all generation files and their metadata."""
    generation_data = {}
    meta_data = {}

    for filename in os.listdir(gen_dir):
        if filename.endswith(".jsonl") and filename != "prompts.jsonl":
            method_name = filename.replace(".jsonl", "")
            filepath = os.path.join(gen_dir, filename)

            samples = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            generation_data[method_name] = samples

            meta_path = os.path.join(gen_dir, f"{method_name}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta_data[method_name] = json.load(f)
            else:
                meta_data[method_name] = {}

    return generation_data, meta_data


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Distinct-n and Rep-n diversity metrics (no GPU required)"
    )

    parser.add_argument(
        "--gen_dir", 
        type=str, 
        required=True, 
        help="Directory containing generation outputs (.jsonl files)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Output directory for results (default: same as gen_dir)"
    )
    parser.add_argument(
        "--tokenize",
        type=str,
        default="word",
        choices=["char", "word"],
        help="Tokenization: 'char' for Chinese, 'word' for English (default: char)"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="generation",
        help="JSON field containing generated text (default: generation)"
    )
    parser.add_argument(
        "--include_prompt",
        action="store_true",
        help="Include prompt in text for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    gen_dir = args.gen_dir
    output_dir = args.output_dir or gen_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DIVERSITY METRICS CALCULATOR (Distinct-n, Rep-n)")
    print("=" * 80)
    print(f"Generation Directory: {gen_dir}")
    print(f"Tokenization Method:  {args.tokenize}")
    print(f"Text Field:           {args.text_field}")
    print(f"Include Prompt:       {args.include_prompt}")
    print(f"Num Workers:          {args.num_workers}")
    print(f"Output Directory:     {output_dir}")
    print("=" * 80)

    # Load generations
    print("\n📂 Loading generations...")
    generation_data, meta_data = load_generations(gen_dir)

    if not generation_data:
        print("❌ No generation files found!")
        return

    print(f"   Found {len(generation_data)} generation methods:")
    for method_name, samples in generation_data.items():
        print(f"      - {method_name}: {len(samples)} samples")

    # Calculate metrics for each method
    print(f"\n📊 Calculating diversity metrics...")

    results = {}

    for method_name, samples in generation_data.items():
        print(f"\n{'='*60}")
        print(f"📈 [{method_name}]")
        print(f"{'='*60}")

        meta = meta_data.get(method_name, {})

        # Prepare texts
        if args.include_prompt:
            texts = [s.get("prompt", "") + s.get(args.text_field, "") for s in samples]
        else:
            texts = [s.get(args.text_field, "") for s in samples]

        # Filter empty texts
        texts = [t for t in texts if t.strip()]

        if not texts:
            print(f"  ⚠️  No valid texts found!")
            continue

        # Calculate metrics
        metrics = calculate_metrics_for_texts(
            texts=texts,
            tokenize_method=args.tokenize,
            num_workers=args.num_workers,
            show_progress=True,
        )

        # Add metadata
        metrics["generation_time_sec"] = meta.get("generation_time", 0)
        metrics["tokens_per_second"] = meta.get("tokens_per_second", 0)

        results[method_name] = metrics

        # Print results
        print(f"\n  📊 Results:")
        print(f"  ┌─────────────────────────────────────────────────────────")
        print(f"  │ Corpus Distinct-1:    {metrics['corpus_distinct_1']:.4f}  (diversity across outputs)")
        print(f"  │ Corpus Distinct-2:    {metrics['corpus_distinct_2']:.4f}")
        print(f"  │ Corpus Distinct-3:    {metrics['corpus_distinct_3']:.4f}")
        print(f"  ├─────────────────────────────────────────────────────────")
        print(f"  │ Sample Distinct-1:    {metrics['sample_distinct_1']:.4f}  (diversity within outputs)")
        print(f"  │ Sample Distinct-2:    {metrics['sample_distinct_2']:.4f}")
        print(f"  │ Sample Distinct-3:    {metrics['sample_distinct_3']:.4f}")
        print(f"  ├─────────────────────────────────────────────────────────")
        print(f"  │ Rep-2:                {metrics['rep_2']:.4f} ± {metrics['rep_2_std']:.4f}  (lower = less repetition)")
        print(f"  │ Rep-3:                {metrics['rep_3']:.4f} ± {metrics['rep_3_std']:.4f}")
        print(f"  │ Rep-4:                {metrics['rep_4']:.4f} ± {metrics['rep_4_std']:.4f}")
        print(f"  ├─────────────────────────────────────────────────────────")
        print(f"  │ Seq-Rep-2:            {metrics['seq_rep_2']:.4f}  (consecutive repetition)")
        print(f"  │ Seq-Rep-3:            {metrics['seq_rep_3']:.4f}")
        print(f"  ├─────────────────────────────────────────────────────────")
        print(f"  │ Avg Length:           {metrics['avg_length']:.1f} ± {metrics['length_std']:.1f} tokens")
        print(f"  │ Vocabulary Size:      {metrics['vocab_size']}")
        print(f"  │ Token Entropy:        {metrics['token_entropy']:.2f} bits")
        print(f"  └─────────────────────────────────────────────────────────")

    # Save results
    output_data = {
        "gen_dir": gen_dir,
        "tokenize_method": args.tokenize,
        "text_field": args.text_field,
        "include_prompt": args.include_prompt,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    json_filename = "diversity_metrics.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = f"{'Method':<20} {'C-Dist1':<8} {'C-Dist2':<8} {'C-Dist3':<8} {'Rep-2':<8} {'Rep-3':<8} {'SeqRep2':<8} {'AvgLen':<8}"
    print(header)
    print("-" * 100)

    for method_name, data in sorted(results.items()):
        row = (
            f"{method_name:<20} "
            f"{data['corpus_distinct_1']:<8.4f} "
            f"{data['corpus_distinct_2']:<8.4f} "
            f"{data['corpus_distinct_3']:<8.4f} "
            f"{data['rep_2']:<8.4f} "
            f"{data['rep_3']:<8.4f} "
            f"{data['seq_rep_2']:<8.4f} "
            f"{data['avg_length']:<8.1f}"
        )
        print(row)

    # Interpretation guide
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("  • Corpus Distinct-n: Higher = more diverse outputs overall (less mode collapse)")
    print("  • Sample Distinct-n: Higher = richer vocabulary within each output")
    print("  • Rep-n:             Lower = less n-gram repetition (less 'looping')")
    print("  • Seq-Rep-n:         Lower = less consecutive repetition (less 'stuttering')")
    print("  • Token Entropy:     Higher = more uniform token distribution")
    print("=" * 100)

    print("\n✅ COMPLETED")
    print(f"   Output: {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()