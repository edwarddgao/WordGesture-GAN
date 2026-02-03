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

from .reranker import GeminiReranker, NoopReranker
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


def evaluate_sentences(
    model: TwoTowerModel,
    sentences: List[SentenceData],
    vocab: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
    k: int = 10,
    reranker=None,
    verbose: bool = False,
    feature_extractor: GestureFeatureExtractor | None = None,
) -> Dict[str, float]:
    """Evaluate sentence-level accuracy with optional reranking.

    Args:
        model: Trained TwoTowerModel
        sentences: List of SentenceData objects
        vocab: Vocabulary words
        layout: Keyboard layout
        n_points: Points per gesture
        device: Torch device
        k: Number of candidates for reranking
        reranker: Optional reranker (GeminiReranker or NoopReranker)
        verbose: Print progress

    Returns:
        Dictionary of metrics
    """
    if reranker is None:
        reranker = NoopReranker()

    # Precompute vocabulary embeddings
    if verbose:
        print("Building vocabulary embeddings...")
    proto_embs, word_to_idx = build_vocabulary_embeddings(
        model, vocab, layout, n_points, device
    )

    # Metrics accumulators
    total_words = 0
    correct_words = 0
    total_sentences = 0
    correct_sentences = 0
    total_wer = 0.0
    recall_at_k_hits = 0

    if verbose:
        print(f"Evaluating {len(sentences)} sentences...")

    for i, sentence in enumerate(sentences):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sentences)} sentences")

        # Encode gestures
        gestures_np = np.stack(sentence.gestures, axis=0)  # (n_words, n_points, 3)

        # Apply feature extraction if available (key proximity, velocity)
        if feature_extractor is not None:
            gestures_np = np.stack(
                [feature_extractor(g) for g in gestures_np], axis=0
            )  # (n_words, n_points, 31)

        gestures_t = torch.from_numpy(gestures_np).to(device)

        with torch.no_grad():
            gesture_embs = model.encode_gesture(gestures_t)  # (n_words, D)

        # Get top-K candidates
        candidates = get_top_k_candidates(gesture_embs, proto_embs, vocab, k)

        # Check recall@K (oracle upper bound)
        for j, gt_word in enumerate(sentence.words):
            candidate_words = {w for w, _ in candidates[j]}
            if gt_word in candidate_words:
                recall_at_k_hits += 1

        # Rerank
        predictions = reranker.rerank(candidates)

        # Compute metrics
        ground_truth = sentence.words

        # Word accuracy
        for pred, gt in zip(predictions, ground_truth):
            total_words += 1
            if pred == gt:
                correct_words += 1

        # Sentence accuracy
        total_sentences += 1
        if predictions == ground_truth:
            correct_sentences += 1

        # WER
        total_wer += compute_wer(predictions, ground_truth)

    # Aggregate metrics
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0
    sentence_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0.0
    avg_wer = total_wer / total_sentences if total_sentences > 0 else 0.0
    recall_at_k = recall_at_k_hits / total_words if total_words > 0 else 0.0

    return {
        "word_accuracy": word_accuracy,
        "sentence_accuracy": sentence_accuracy,
        "wer": avg_wer,
        f"recall@{k}": recall_at_k,
        "total_words": total_words,
        "total_sentences": total_sentences,
    }


async def evaluate_sentences_async(
    model: TwoTowerModel,
    sentences: List[SentenceData],
    vocab: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
    k: int = 10,
    reranker=None,
    verbose: bool = False,
    return_details: bool = False,
    ctc_decoder=None,
    feature_extractor: GestureFeatureExtractor | None = None,
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
        verbose: Print progress
        return_details: Return per-sentence details
        ctc_decoder: Optional CTC decoder for OOV recovery

    Returns:
        Dictionary of metrics
    """
    if reranker is None:
        reranker = NoopReranker()

    # Precompute vocabulary embeddings
    if verbose:
        print("Building vocabulary embeddings...")
    proto_embs, word_to_idx = build_vocabulary_embeddings(
        model, vocab, layout, n_points, device
    )

    if verbose:
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
    if verbose:
        print(f"Reranking {len(sentences)} sentences...")

    if hasattr(reranker, "rerank_batch_async"):
        all_predictions = await reranker.rerank_batch_async(all_candidates, all_ctc_words)
    else:
        all_predictions = [
            reranker.rerank(c, ctc)
            for c, ctc in zip(all_candidates, all_ctc_words)
        ]

    # Compute metrics with error categorization
    vocab_set = set(vocab)
    errors = ErrorBreakdown(total_words=total_words)
    correct_sentences = 0
    total_wer = 0.0
    sentence_details = []

    for i, (predictions, ground_truth, candidates, ctc_words) in enumerate(
        zip(all_predictions, all_ground_truth, all_candidates, all_ctc_words)
    ):
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

        if return_details:
            sentence_details.append({
                "index": i,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "candidates": candidates,
                "ctc_words": ctc_words,
                "is_correct": is_correct,
                "word_accuracy": sum(1 for p, g in zip(predictions, ground_truth) if p == g) / len(ground_truth) if ground_truth else 0,
                "wer": wer,
            })

    total_sentences = len(sentences)
    word_accuracy = errors.correct / total_words if total_words > 0 else 0.0
    sentence_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0.0
    avg_wer = total_wer / total_sentences if total_sentences > 0 else 0.0
    recall_at_k = recall_at_k_hits / total_words if total_words > 0 else 0.0

    results = {
        "word_accuracy": word_accuracy,
        "sentence_accuracy": sentence_accuracy,
        "wer": avg_wer,
        f"recall@{k}": recall_at_k,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "errors": errors,
    }

    if return_details:
        results["details"] = sentence_details

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
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed data.",
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
        default="gemini-3-flash-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sentence sampling.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic sentences (random word combinations) in addition to natural sentences.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls for reranking.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show sentences that failed after reranking.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show detailed breakdown of OOV, retrieval, and reranking failures.",
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

    # Evaluate baseline (top-1, no reranking)
    print("\n" + "=" * 60)
    print("Baseline (Top-1, no reranking)")
    print("=" * 60)
    baseline_results = evaluate_sentences(
        model, sentences, vocab, layout, n_points, device,
        k=args.k, reranker=NoopReranker(), verbose=True,
        feature_extractor=feature_extractor,
    )
    print(f"\n  Word Accuracy:     {baseline_results['word_accuracy']:.1%}")
    print(f"  Sentence Accuracy: {baseline_results['sentence_accuracy']:.1%}")
    print(f"  WER:               {baseline_results['wer']:.3f}")
    print(f"  Recall@{args.k}:          {baseline_results[f'recall@{args.k}']:.1%} (oracle upper bound)")

    # Evaluate with reranker if requested
    reranker_results = None
    if args.reranker == "gemini":
        print("\n" + "=" * 60)
        print(f"With Gemini Reranker ({args.model})")
        print("=" * 60)

        try:
            reranker = GeminiReranker(
                project=args.project,
                location=args.location,
                model=args.model,
                max_concurrent=args.max_concurrent,
                max_candidates_display=args.max_candidates_display,
            )
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nTo use the Gemini reranker, install required dependencies:")
            print("  pip install google-genai>=1.51.0 python-dotenv")
            print("\nSee README.md for full setup instructions.")
            print("\nAlternatively, run with --reranker none for baseline evaluation.")
            return
        # Use async evaluation for parallel API calls
        reranker_results = asyncio.run(evaluate_sentences_async(
            model, sentences, vocab, layout, n_points, device,
            k=args.k, reranker=reranker, verbose=True,
            return_details=args.show_errors or args.show_failures,
            ctc_decoder=ctc_decoder,
            feature_extractor=feature_extractor,
        ))

        # Compute improvements
        word_acc_delta = reranker_results["word_accuracy"] - baseline_results["word_accuracy"]
        sent_acc_delta = reranker_results["sentence_accuracy"] - baseline_results["sentence_accuracy"]
        wer_delta = reranker_results["wer"] - baseline_results["wer"]
        wer_reduction = -wer_delta / baseline_results["wer"] if baseline_results["wer"] > 0 else 0

        print(f"\n  Word Accuracy:     {reranker_results['word_accuracy']:.1%} ({word_acc_delta:+.1%})")
        print(f"  Sentence Accuracy: {reranker_results['sentence_accuracy']:.1%} ({sent_acc_delta:+.1%})")
        print(f"  WER:               {reranker_results['wer']:.3f} ({wer_reduction:.1%} reduction)")

        # Print error breakdown
        err = reranker_results["errors"]
        total_errors = err.oov + err.retrieval_fail + err.rerank_fail
        print(f"\n  Error Breakdown ({total_errors} errors):")
        print(f"    OOV:        {err.oov:4d} ({100*err.oov/err.total_words:.1f}%) - word not in vocab")
        print(f"    Retrieval:  {err.retrieval_fail:4d} ({100*err.retrieval_fail/err.total_words:.1f}%) - word in vocab, not top-{args.k}")
        print(f"    Reranking:  {err.rerank_fail:4d} ({100*err.rerank_fail/err.total_words:.1f}%) - word in top-{args.k}, LLM wrong")

        # Show errors if requested
        if args.show_errors and "details" in reranker_results:
            errors = [d for d in reranker_results["details"] if not d["is_correct"]]
            if errors:
                print(f"\n" + "=" * 60)
                print(f"Failed Sentences ({len(errors)} / {len(reranker_results['details'])})")
                print("=" * 60)
                for err in errors:
                    print(f"\n[{err['index']}] Ground truth: {' '.join(err['ground_truth'])}")
                    print(f"    Prediction:   {' '.join(err['predictions'])}")
                    # Show word-level diff
                    diffs = []
                    for j, (gt, pred) in enumerate(zip(err['ground_truth'], err['predictions'])):
                        if gt != pred:
                            # Show top candidates for this position
                            cands = err['candidates'][j][:5]
                            cand_str = ", ".join(f"{w}({s:.2f})" for w, s in cands)
                            gt_in_cands = any(w == gt for w, _ in err['candidates'][j])
                            marker = "✓" if gt_in_cands else "✗"
                            diffs.append(f"    Position {j+1}: '{pred}' should be '{gt}' {marker} [{cand_str}]")
                    for diff in diffs:
                        print(diff)

        # Show reranking failures only
        if args.show_failures and "details" in reranker_results:
            from collections import Counter
            vocab_set = set(vocab)

            oov_failures = []
            retrieval_failures = []
            rerank_failures = []

            for detail in reranker_results["details"]:
                ctc_words = detail.get('ctc_words')
                for j, (gt, pred) in enumerate(zip(detail['ground_truth'], detail['predictions'])):
                    if gt != pred:
                        cand_words = [w for w, _ in detail['candidates'][j]]
                        cands = detail['candidates'][j][:5]
                        cand_str = ", ".join(f"{w}({s:.2f})" for w, s in cands)
                        context = " ".join(detail['ground_truth'])
                        ctc_word = ctc_words[j] if ctc_words else None

                        failure = {
                            "gt": gt,
                            "pred": pred,
                            "context": context,
                            "cands": cand_str,
                            "ctc": ctc_word,
                        }

                        if gt not in vocab_set:
                            oov_failures.append(failure)
                        elif gt not in cand_words:
                            retrieval_failures.append(failure)
                        else:
                            gt_rank = cand_words.index(gt) + 1
                            failure["gt_rank"] = gt_rank
                            rerank_failures.append(failure)

            # OOV failures
            print(f"\n" + "=" * 60)
            print(f"OOV Failures ({len(oov_failures)}) - word not in vocabulary")
            print("=" * 60)
            for f in oov_failures:
                ctc_str = f" [CTC: '{f['ctc']}']" if f['ctc'] else ""
                print(f"  '{f['gt']}' (OOV) -> '{f['pred']}'{ctc_str} | {f['context']}")

            # Retrieval failures
            print(f"\n" + "=" * 60)
            print(f"Retrieval Failures ({len(retrieval_failures)}) - word in vocab, not top-{args.k}")
            print("=" * 60)
            for f in retrieval_failures:
                ctc_str = f" [CTC: '{f['ctc']}']" if f['ctc'] else ""
                print(f"  '{f['gt']}' -> '{f['pred']}'{ctc_str} | {f['context']}")
                print(f"      top-5: {f['cands']}")

            # Reranking failures
            print(f"\n" + "=" * 60)
            print(f"Reranking Failures ({len(rerank_failures)}) - word in top-{args.k}, LLM wrong")
            print("=" * 60)

            pair_counts = Counter((f["pred"], f["gt"]) for f in rerank_failures)
            print(f"\nMost common confusions (pred <- gt):")
            for (pred, gt), count in pair_counts.most_common(30):
                print(f"  {pred:15} <- {gt:15} ({count}x)")

            print(f"\nDetailed:")
            for f in rerank_failures[:50]:
                ctc_str = f" [CTC: '{f['ctc']}']" if f['ctc'] else ""
                print(f"  '{f['pred']}' <- '{f['gt']}' (rank {f['gt_rank']}){ctc_str} | {f['context']}")
                print(f"      top-5: {f['cands']}")

    # Save results
    if args.output:
        results = {
            "config": {
                "checkpoint": str(args.checkpoint),
                "max_sentences": args.max_sentences,
                "k": args.k,
                "reranker": args.reranker,
                "seed": args.seed,
            },
            "baseline": baseline_results,
        }
        if reranker_results:
            results["reranker"] = reranker_results
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
