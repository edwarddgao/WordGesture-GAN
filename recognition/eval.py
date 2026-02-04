"""Evaluate gesture recognition pipeline with optional LLM reranking.

This module combines:
- Contrastive model for embedding-based retrieval
- CTC decoder for OOV recovery
- LLM reranker for final word selection
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from shared import KeyboardLayout, build_word_prototype, get_device
from features import GestureFeatureExtractor
from contrastive.models import TwoTowerModel

from .reranker import GeminiReranker, NoopReranker, RerankerResult
from .sentence_data import SentenceData, get_sentence_stats, load_sentence_dataset_subset

# Optional CTC decoder import
try:
    from ctc import CTCDecoder
    CTC_AVAILABLE = True
except ImportError:
    CTCDecoder = None
    CTC_AVAILABLE = False


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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[TwoTowerModel, Dict, GestureFeatureExtractor]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    # Get gesture_in_channels from config (default 31 for new models, 3 for old)
    gesture_in_channels = cfg["model"].get("gesture_in_channels", 3)

    model = TwoTowerModel(
        gesture_in_channels=gesture_in_channels,
        embedding_dim=cfg["model"]["embedding_dim"],
        projection_dim=cfg["model"]["projection_dim"],
        temperature=cfg["contrastive"]["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Create feature extractor if model uses extended features
    features_cfg = cfg.get("features", {})
    use_key_proximity = features_cfg.get("use_key_proximity", gesture_in_channels > 3)
    use_velocity = features_cfg.get("use_velocity", gesture_in_channels > 3)
    key_proximity_sigma = features_cfg.get("key_proximity_sigma", 0.3)

    feature_extractor = GestureFeatureExtractor(
        layout=KeyboardLayout(),
        sigma=key_proximity_sigma,
        use_key_proximity=use_key_proximity,
        use_velocity=use_velocity,
    )

    return model, cfg, feature_extractor


def build_vocabulary_embeddings(
    model: TwoTowerModel,
    vocab: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Precompute prototype embeddings for entire vocabulary.

    Returns:
        proto_embs: (V, D) tensor of prototype embeddings
        word_to_idx: mapping from word to vocabulary index
    """
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Build prototype arrays
    proto_list = []
    for w in vocab:
        proto = build_word_prototype(w, n_points, layout)[:, :2]  # (n_points, 2)
        proto_list.append(proto)
    protos = np.stack(proto_list, axis=0)  # (V, n_points, 2)

    # Encode in batches to avoid MPS numerical precision issues with large tensors
    proto_embs_list = []
    with torch.no_grad():
        for i in range(0, len(protos), batch_size):
            batch = torch.from_numpy(protos[i : i + batch_size]).to(device)
            emb = model.encode_prototype(batch)
            proto_embs_list.append(emb)
        proto_embs = torch.cat(proto_embs_list, dim=0)  # (V, D)

    return proto_embs, word_to_idx


def get_top_k_candidates(
    gesture_embs: torch.Tensor,
    proto_embs: torch.Tensor,
    vocab: List[str],
    k: int,
) -> List[List[Tuple[str, float]]]:
    """Get top-K candidate words for each gesture.

    Args:
        gesture_embs: (N, D) gesture embeddings
        proto_embs: (V, D) prototype embeddings
        vocab: List of vocabulary words
        k: Number of candidates per position

    Returns:
        List of candidate lists, each containing (word, score) tuples
    """
    # Compute similarities
    similarities = gesture_embs @ proto_embs.T  # (N, V)

    # Get top-K indices and scores
    top_scores, top_indices = similarities.topk(k, dim=1)  # (N, k)

    candidates = []
    for i in range(len(gesture_embs)):
        position_candidates = [
            (vocab[top_indices[i, j].item()], top_scores[i, j].item())
            for j in range(k)
        ]
        candidates.append(position_candidates)

    return candidates


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
    model: TwoTowerModel,
    sentences: List[SentenceData],
    vocab: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
    k: int = 10,
    reranker=None,
    ctc_decoder=None,
    feature_extractor: GestureFeatureExtractor | None = None,
    log_file: Path | None = None,
) -> Dict[str, float]:
    """Evaluate sentence-level accuracy with parallel reranking.

    Args:
        model: Trained TwoTowerModel
        sentences: List of SentenceData objects
        vocab: Vocabulary words
        layout: Keyboard layout
        n_points: Points per gesture
        device: Torch device
        k: Number of candidates for reranking
        reranker: Reranker with rerank_batch_async method
        ctc_decoder: Optional CTC decoder for OOV recovery
        log_file: Path to write JSONL log with per-sentence details and summary

    Returns:
        Dictionary of metrics
    """
    if reranker is None:
        reranker = NoopReranker()

    # Precompute vocabulary embeddings
    print("Building vocabulary embeddings...")
    proto_embs, word_to_idx = build_vocabulary_embeddings(
        model, vocab, layout, n_points, device
    )

    print(f"Evaluating {len(sentences)} sentences...")

    # Collect all candidates
    all_candidates = []
    all_ground_truth = []
    all_ctc_words = []
    recall_at_k_hits = 0
    total_words = 0

    for sentence in sentences:
        # Encode gestures
        gestures_np = np.stack(sentence.gestures, axis=0)

        # Apply feature extraction if available (key proximity, velocity)
        if feature_extractor is not None:
            gestures_np = np.stack(
                [feature_extractor(g) for g in gestures_np], axis=0
            )  # (n_words, n_points, 31)

        gestures_t = torch.from_numpy(gestures_np).to(device)

        with torch.no_grad():
            gesture_embs = model.encode_gesture(gestures_t)

        # Get top-K candidates
        candidates = get_top_k_candidates(gesture_embs, proto_embs, vocab, k)

        # Get CTC decoded words if decoder is available (passed separately to reranker)
        ctc_words = None
        if ctc_decoder is not None:
            ctc_words = ctc_decoder.decode_batch(sentence.gestures)

        all_candidates.append(candidates)
        all_ground_truth.append(sentence.words)
        all_ctc_words.append(ctc_words)

        # Check recall@K (oracle upper bound) - includes CTC decoded words
        for j, gt_word in enumerate(sentence.words):
            total_words += 1
            candidate_words = {w for w, _ in candidates[j]}
            # Also count CTC decoded word as a valid candidate
            if ctc_words is not None and j < len(ctc_words) and ctc_words[j]:
                candidate_words.add(ctc_words[j])
            if gt_word in candidate_words:
                recall_at_k_hits += 1

    # Rerank all sentences in parallel
    # Ensure ctc_words is always a list (empty strings if no decoder)
    all_ctc_words_safe = [
        ctc if ctc is not None else [""] * len(cands)
        for ctc, cands in zip(all_ctc_words, all_candidates)
    ]

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
    log_entries = []  # Collect log entries to write at end

    for i, (result, ground_truth, candidates, ctc_words) in enumerate(
        zip(all_results, all_ground_truth, all_candidates, all_ctc_words_safe)
    ):
        predictions = result.predictions
        # Categorize each word
        for j, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            candidate_words = {w for w, _ in candidates[j]}
            # Include CTC decoded word as valid candidate
            ctc_word = ctc_words[j] if ctc_words and j < len(ctc_words) else None
            if ctc_word:
                candidate_words.add(ctc_word)

            if gt not in vocab_set and (ctc_word is None or gt != ctc_word):
                # OOV and CTC didn't decode it correctly
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
                "ctc_words": ctc_words,
                "is_correct": is_correct,
            })

    total_sentences = len(sentences)
    word_accuracy = errors.correct / total_words if total_words > 0 else 0.0
    sentence_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0.0
    avg_wer = total_wer / total_sentences if total_sentences > 0 else 0.0
    recall_at_k = recall_at_k_hits / total_words if total_words > 0 else 0.0

    # Write log file with context manager for proper resource cleanup
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
        description="Evaluate gesture recognition pipeline with optional LLM reranking."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to contrastive model checkpoint.",
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
        "--reranker",
        choices=["none", "gemini"],
        default="none",
        help="Reranker to use. 'none' uses top-1 candidates, 'gemini' uses LLM reranking.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project ID for Gemini (or set GOOGLE_CLOUD_PROJECT).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="GCP location for Gemini (default: GOOGLE_CLOUD_LOCATION or 'global').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name.",
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
        "--max-candidates-display",
        type=int,
        default=10,
        help="Maximum candidates to show LLM per position (default: 10).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Use top N most common English words from wordfreq (default: 10000).",
    )
    parser.add_argument(
        "--ctc-checkpoint",
        type=Path,
        default=None,
        help="Path to CTC decoder checkpoint. If provided, CTC decoded words are added as candidates.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Handle random seed
    import random
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    print(f"Using seed: {args.seed}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg, feature_extractor = load_model(args.checkpoint, device)
    n_points = cfg["data"]["n_points"]
    layout = KeyboardLayout()
    gesture_in_channels = cfg["model"].get("gesture_in_channels", 3)
    print(f"Gesture input channels: {gesture_in_channels}")

    # Load sentence data first (needed to find OOV words)
    print(f"Loading sentence data from {args.raw_dir}...")
    sentences = load_sentence_dataset_subset(
        args.raw_dir, n_points, args.max_sentences, args.seed,
        natural_only=not args.include_synthetic,
    )
    stats = get_sentence_stats(sentences)
    print(f"Loaded {stats['n_sentences']} sentences ({stats['n_words']} words)")

    # Build vocabulary from wordfreq
    test_words = {w for s in sentences for w in s.words}
    vocab = load_wordfreq_vocabulary(args.vocab_size)
    oov_count = len(test_words - set(vocab))
    print(f"Vocabulary size: {len(vocab)} (wordfreq top-{args.vocab_size}, {oov_count}/{len(test_words)} test words OOV)")

    # Load CTC decoder if provided
    ctc_decoder = None
    if args.ctc_checkpoint is not None:
        if not CTC_AVAILABLE:
            print("\nWarning: CTC module not available. Install with: pip install -e .")
            print("Continuing without CTC decoder.\n")
        else:
            print(f"Loading CTC decoder from {args.ctc_checkpoint}...")
            ctc_decoder = CTCDecoder.from_checkpoint(str(args.ctc_checkpoint))
            print("CTC decoder loaded.")

    # Select reranker
    if args.reranker == "gemini":
        try:
            reranker = GeminiReranker(
                project=args.project,
                location=args.location,
                model=args.model,
                max_concurrent=args.max_concurrent,
                max_candidates_display=args.max_candidates_display,
            )
            reranker_name = f"Gemini ({args.model})"
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nTo use the Gemini reranker, install required dependencies:")
            print("  pip install google-genai>=1.51.0 python-dotenv")
            print("\nSee README.md for full setup instructions.")
            print("\nAlternatively, run with --reranker none for baseline evaluation.")
            return
    else:
        reranker = NoopReranker()
        reranker_name = "Top-1 (no reranking)"

    # Evaluate
    print("\n" + "=" * 60)
    print(f"Evaluation: {reranker_name}")
    print("=" * 60)
    results = asyncio.run(evaluate_sentences(
        model, sentences, vocab, layout, n_points, device,
        k=args.k, reranker=reranker,
        ctc_decoder=ctc_decoder,
        feature_extractor=feature_extractor,
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
    print(f"    Retrieval:  {err.retrieval_fail:4d} ({100*err.retrieval_fail/err.total_words:.1f}%) - word in vocab, not top-{args.k}")
    print(f"    Reranking:  {err.rerank_fail:4d} ({100*err.rerank_fail/err.total_words:.1f}%) - word in top-{args.k}, wrong")
    print(f"    Fallbacks:  {results['total_fallbacks']:4d} ({100*results['total_fallbacks']/err.total_words:.1f}%) - invalid output, used top-1")
    if results.get("fallback_reasons"):
        for reason, count in sorted(results["fallback_reasons"].items(), key=lambda x: -x[1]):
            print(f"      - {reason}: {count}")

    if args.rerank_log:
        print(f"\n  Log written to: {args.rerank_log}")


if __name__ == "__main__":
    main()
