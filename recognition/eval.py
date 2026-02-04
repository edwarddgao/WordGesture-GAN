"""Evaluate gesture recognition pipeline with optional LLM reranking.

This module uses:
- CTC decoder with beam search for candidate generation
- LLM reranker for final word selection (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from shared import get_device
from ctc import CTCDecoder

from .reranker import OpenRouterReranker, NoopReranker, RerankerResult
from .sentence_data import SentenceData, get_sentence_stats, load_sentence_dataset_subset


def load_wordfreq_vocabulary(n: int) -> List[str]:
    """Load top N most common English words from wordfreq.

    Args:
        n: Number of top words to retrieve (e.g., 10000, 50000, 100000)

    Returns:
        List of words filtered to ASCII alpha-only, length >= 2
    """
    from wordfreq import top_n_list

    words = top_n_list('en', n)
    # Filter to ASCII alpha-only (a-z), length >= 2
    words = [w for w in words if w.isascii() and w.isalpha() and len(w) >= 2]
    return words  # Already sorted by frequency


@dataclass
class ErrorBreakdown:
    """Categorized error counts for evaluation."""
    total_words: int = 0
    correct: int = 0
    oov: int = 0              # GT word not in vocab
    retrieval_fail: int = 0   # GT in vocab, not in top-k
    rerank_fail: int = 0      # GT in top-k, LLM chose wrong


def compute_wer(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute Word Error Rate using Levenshtein distance."""
    if not ground_truth:
        return 0.0 if not predictions else 1.0

    # Dynamic programming for edit distance
    m, n = len(predictions), len(ground_truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predictions[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / n


async def evaluate_sentences(
    ctc_decoder: CTCDecoder,
    sentences: List[SentenceData],
    vocab: List[str],
    k: int = 10,
    beam_width: int = 100,
    reranker=None,
    log_file: Path | None = None,
) -> Dict[str, float]:
    """Evaluate sentence-level accuracy with parallel reranking.

    Args:
        ctc_decoder: CTC decoder with vocabulary set
        sentences: List of SentenceData objects
        vocab: Vocabulary words
        k: Number of candidates for reranking
        beam_width: Beam width for CTC decoding
        reranker: Reranker with rerank_batch_async method
        log_file: Path to write JSONL log with per-sentence details and summary

    Returns:
        Dictionary of metrics
    """
    if reranker is None:
        reranker = NoopReranker()

    print(f"Evaluating {len(sentences)} sentences...")

    # Collect all candidates
    all_candidates = []
    all_ground_truth = []
    recall_at_k_hits = 0
    total_words = 0

    all_greedy_words = []
    for sentence in sentences:
        # Get top-K candidates using CTC beam search
        candidates = ctc_decoder.decode_batch_top_k(
            sentence.gestures, k=k, beam_width=beam_width
        )
        # Get greedy decode for "Decoded" field in reranker prompt
        greedy_words = ctc_decoder.decode_batch(sentence.gestures)

        all_candidates.append(candidates)
        all_ground_truth.append(sentence.words)
        all_greedy_words.append(greedy_words)

        # Check recall@K (oracle upper bound)
        for j, gt_word in enumerate(sentence.words):
            total_words += 1
            candidate_words = {w for w, _ in candidates[j]}
            if gt_word in candidate_words:
                recall_at_k_hits += 1

    # Rerank all sentences in parallel with greedy decode as hint
    all_ctc_words_safe = all_greedy_words

    if hasattr(reranker, "rerank_batch_async"):
        all_results: List[RerankerResult] = await reranker.rerank_batch_async(
            all_candidates, all_ctc_words_safe
        )
    else:
        all_results = [
            reranker.rerank(c, ctc)
            for c, ctc in zip(all_candidates, all_ctc_words_safe)
        ]

    # Compute metrics with error categorization
    vocab_set = set(vocab)
    errors = ErrorBreakdown(total_words=total_words)
    correct_sentences = 0
    total_wer = 0.0
    total_fallbacks = 0
    fallback_reasons = {}
    log_entries = []

    for i, (result, ground_truth, candidates) in enumerate(
        zip(all_results, all_ground_truth, all_candidates)
    ):
        predictions = result.predictions
        # Categorize each word
        for j, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            candidate_words = {w for w, _ in candidates[j]}

            if gt not in vocab_set:
                errors.oov += 1
            elif gt not in candidate_words:
                errors.retrieval_fail += 1
            elif pred != gt:
                errors.rerank_fail += 1
            else:
                errors.correct += 1

        # Sentence accuracy
        is_correct = predictions == ground_truth
        if is_correct:
            correct_sentences += 1

        # WER
        wer = compute_wer(predictions, ground_truth)
        total_wer += wer

        # Count fallbacks from parse details with reasons
        for detail in result.parse_details:
            if detail.get("fallback", False):
                reason = detail.get("fallback_reason", "unknown")
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                total_fallbacks += 1

        # Collect log entry if logging enabled
        if log_file is not None:
            log_entries.append({
                "sentence_idx": i,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "raw_response": result.raw_response,
                "parse_details": result.parse_details,
                "candidates": [[(w, s) for w, s in cands[:5]] for cands in candidates],
                "is_correct": is_correct,
            })

    total_sentences = len(sentences)
    word_accuracy = errors.correct / total_words if total_words > 0 else 0.0
    sentence_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0.0
    avg_wer = total_wer / total_sentences if total_sentences > 0 else 0.0
    recall_at_k = recall_at_k_hits / total_words if total_words > 0 else 0.0

    # Write log file
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "type": "summary",
            "word_accuracy": word_accuracy,
            "sentence_accuracy": sentence_accuracy,
            "wer": avg_wer,
            f"recall@{k}": recall_at_k,
            "total_words": total_words,
            "total_sentences": total_sentences,
            "errors": {
                "oov": errors.oov,
                "retrieval_fail": errors.retrieval_fail,
                "rerank_fail": errors.rerank_fail,
                "correct": errors.correct,
            },
            "total_fallbacks": total_fallbacks,
            "fallback_reasons": fallback_reasons,
        }
        with log_file.open("w") as log_fh:
            for entry in log_entries:
                log_fh.write(json.dumps(entry) + "\n")
            log_fh.write(json.dumps(summary) + "\n")

    results = {
        "word_accuracy": word_accuracy,
        "sentence_accuracy": sentence_accuracy,
        "wer": avg_wer,
        f"recall@{k}": recall_at_k,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "errors": errors,
        "total_fallbacks": total_fallbacks,
        "fallback_reasons": fallback_reasons,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate gesture recognition with CTC beam search and optional LLM reranking."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ctc/checkpoints/ctc_best.pt"),
        help="Path to CTC model checkpoint.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/processed/raw"),
        help="Directory containing raw JSONL files.",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=50,
        help="Maximum sentences to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of candidates for reranking.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=100,
        help="Beam width for CTC decoding.",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable LLM reranking via OpenRouter (default: top-1 only).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="OpenRouter model name (e.g., google/gemini-3-flash-preview, openai/gpt-4o).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sentence sampling. If not specified, uses a random seed.",
    )
    parser.add_argument(
        "--rerank-log",
        type=Path,
        default=None,
        help="Path to write reranker debug log (JSONL format). One entry per sentence.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic sentences (random word combinations) in addition to natural sentences.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent API calls for reranking.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20000,
        help="Use top N most common English words from wordfreq (default: 20000).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Handle random seed
    import random
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    print(f"Using seed: {args.seed}")

    # Build vocabulary from wordfreq
    vocab = load_wordfreq_vocabulary(args.vocab_size)
    print(f"Vocabulary size: {len(vocab)} (wordfreq top-{args.vocab_size})")

    # Load CTC decoder with vocabulary
    print(f"Loading CTC decoder from {args.checkpoint}...")
    ctc_decoder = CTCDecoder.from_checkpoint(str(args.checkpoint), vocabulary=vocab)
    print(f"CTC decoder loaded (trie size: {len(ctc_decoder._trie)})")

    # Load sentence data
    print(f"Loading sentence data from {args.raw_dir}...")
    sentences = load_sentence_dataset_subset(
        args.raw_dir, n_points=128, max_sentences=args.max_sentences, seed=args.seed,
        natural_only=not args.include_synthetic,
    )
    stats = get_sentence_stats(sentences)
    test_words = {w for s in sentences for w in s.words}
    oov_count = len(test_words - set(vocab))
    print(f"Loaded {stats['n_sentences']} sentences ({stats['n_words']} words, {oov_count} OOV)")

    # Select reranker
    if args.reranker:
        try:
            reranker = OpenRouterReranker(
                api_key=args.api_key,
                model=args.model,
                max_concurrent=args.max_concurrent,
                max_candidates_display=args.k,
            )
            reranker_name = f"OpenRouter ({args.model})"
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nTo use the OpenRouter reranker, install required dependencies:")
            print("  pip install openai python-dotenv")
            print("\nAlternatively, run without --reranker for baseline evaluation.")
            return
    else:
        reranker = NoopReranker()
        reranker_name = "Top-1 (no reranking)"

    # Evaluate
    print("\n" + "=" * 60)
    print(f"Evaluation: CTC Beam Search + {reranker_name}")
    print("=" * 60)
    results = asyncio.run(evaluate_sentences(
        ctc_decoder, sentences, vocab,
        k=args.k, beam_width=args.beam_width,
        reranker=reranker,
        log_file=args.rerank_log,
    ))

    print(f"\n  Word Accuracy:     {results['word_accuracy']:.1%}")
    print(f"  Sentence Accuracy: {results['sentence_accuracy']:.1%}")
    print(f"  WER:               {results['wer']:.3f}")
    print(f"  Recall@{args.k}:          {results[f'recall@{args.k}']:.1%} (oracle upper bound)")

    # Print error breakdown
    err = results["errors"]
    total_errors = err.oov + err.retrieval_fail + err.rerank_fail
    print(f"\n  Error Breakdown ({total_errors} errors):")
    print(f"    OOV:        {err.oov:4d} ({100*err.oov/err.total_words:.1f}%) - word not in vocab")
    print(f"    Retrieval:  {err.retrieval_fail:4d} ({100*err.retrieval_fail/err.total_words:.1f}%) - not in top-{args.k}")
    print(f"    Reranking:  {err.rerank_fail:4d} ({100*err.rerank_fail/err.total_words:.1f}%) - in top-{args.k}, wrong")
    print(f"    Fallbacks:  {results['total_fallbacks']:4d} ({100*results['total_fallbacks']/err.total_words:.1f}%) - invalid output, used top-1")
    if results.get("fallback_reasons"):
        for reason, count in sorted(results["fallback_reasons"].items(), key=lambda x: -x[1]):
            print(f"      - {reason}: {count}")

    if args.rerank_log:
        print(f"\n  Log written to: {args.rerank_log}")


if __name__ == "__main__":
    main()
