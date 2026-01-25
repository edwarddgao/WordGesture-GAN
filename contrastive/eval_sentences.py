"""Sentence-level evaluation with optional LLM reranking."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from shared import KeyboardLayout, build_word_prototype, get_device, load_dataset

from .models import TwoTowerModel
from .reranker import GeminiReranker, NoopReranker
from .sentence_data import SentenceData, get_sentence_stats, load_sentence_dataset_subset


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[TwoTowerModel, Dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = TwoTowerModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        projection_dim=cfg["model"]["projection_dim"],
        temperature=cfg["contrastive"]["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, cfg


def build_vocabulary_embeddings(
    model: TwoTowerModel,
    vocab: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Precompute prototype embeddings for entire vocabulary.

    Returns:
        proto_embs: (V, D) tensor of prototype embeddings
        word_to_idx: mapping from word to vocabulary index
    """
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    with torch.no_grad():
        proto_list = []
        for w in vocab:
            proto = build_word_prototype(w, n_points, layout)[:, :2]  # (n_points, 2)
            proto_list.append(proto)
        protos = torch.from_numpy(np.stack(proto_list, axis=0)).to(device)  # (V, n_points, 2)
        proto_embs = model.encode_prototype(protos)  # (V, D)

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
    recall_at_k_hits = 0
    total_words = 0

    for sentence in sentences:
        # Encode gestures
        gestures_np = np.stack(sentence.gestures, axis=0)
        gestures_t = torch.from_numpy(gestures_np).to(device)

        with torch.no_grad():
            gesture_embs = model.encode_gesture(gestures_t)

        # Get top-K candidates
        candidates = get_top_k_candidates(gesture_embs, proto_embs, vocab, k)
        all_candidates.append(candidates)
        all_ground_truth.append(sentence.words)

        # Check recall@K (oracle upper bound)
        for j, gt_word in enumerate(sentence.words):
            total_words += 1
            candidate_words = {w for w, _ in candidates[j]}
            if gt_word in candidate_words:
                recall_at_k_hits += 1

    # Rerank all sentences in parallel
    if verbose:
        print(f"Reranking {len(sentences)} sentences in parallel...")

    if hasattr(reranker, 'rerank_batch_async'):
        all_predictions = await reranker.rerank_batch_async(all_candidates)
    else:
        all_predictions = [reranker.rerank(c) for c in all_candidates]

    # Compute metrics
    correct_words = 0
    correct_sentences = 0
    total_wer = 0.0
    sentence_details = []

    for i, (predictions, ground_truth, candidates) in enumerate(
        zip(all_predictions, all_ground_truth, all_candidates)
    ):
        # Word accuracy
        word_correct = 0
        for pred, gt in zip(predictions, ground_truth):
            if pred == gt:
                correct_words += 1
                word_correct += 1

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
                "is_correct": is_correct,
                "word_accuracy": word_correct / len(ground_truth) if ground_truth else 0,
                "wer": wer,
            })

    total_sentences = len(sentences)
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0
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
    }

    if return_details:
        results["details"] = sentence_details

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sentence-level accuracy with optional LLM reranking."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
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
        default=None,
        help="Maximum sentences to evaluate (for quick testing).",
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
        help="Reranker to use.",
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
        "--natural-only",
        action="store_true",
        help="Only use natural sentences (enron dataset) instead of random word combinations.",
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
        "--max-candidates-display",
        type=int,
        default=10,
        help="Maximum candidates to show LLM per position (default: 10).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device)
    n_points = cfg["data"]["n_points"]
    layout = KeyboardLayout()

    # Load sentence data first (needed to find OOV words)
    print(f"Loading sentence data from {args.raw_dir}...")
    sentences = load_sentence_dataset_subset(
        args.raw_dir, n_points, args.max_sentences, args.seed,
        natural_only=args.natural_only,
    )
    stats = get_sentence_stats(sentences)
    print(f"Loaded {stats['n_sentences']} sentences ({stats['n_words']} words)")

    # Build vocabulary from training data + OOV words from test sentences
    _, train_words = load_dataset(args.data_dir / "train.npz")
    train_vocab = set(train_words)
    test_words = {w for s in sentences for w in s.words}
    oov_words = test_words - train_vocab
    vocab = sorted(train_vocab | test_words)
    print(f"Vocabulary size: {len(vocab)} ({len(train_vocab)} train + {len(oov_words)} OOV)")

    # Evaluate baseline (top-1, no reranking)
    print("\n" + "=" * 60)
    print("Baseline (Top-1, no reranking)")
    print("=" * 60)
    baseline_results = evaluate_sentences(
        model, sentences, vocab, layout, n_points, device,
        k=args.k, reranker=NoopReranker(), verbose=True
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

        reranker = GeminiReranker(
            project=args.project,
            location=args.location,
            model=args.model,
            max_concurrent=args.max_concurrent,
            max_candidates_display=args.max_candidates_display,
        )
        # Use async evaluation for parallel API calls
        reranker_results = asyncio.run(evaluate_sentences_async(
            model, sentences, vocab, layout, n_points, device,
            k=args.k, reranker=reranker, verbose=True,
            return_details=args.show_errors,
        ))

        # Compute improvements
        word_acc_delta = reranker_results["word_accuracy"] - baseline_results["word_accuracy"]
        sent_acc_delta = reranker_results["sentence_accuracy"] - baseline_results["sentence_accuracy"]
        wer_delta = reranker_results["wer"] - baseline_results["wer"]
        wer_reduction = -wer_delta / baseline_results["wer"] if baseline_results["wer"] > 0 else 0

        print(f"\n  Word Accuracy:     {reranker_results['word_accuracy']:.1%} ({word_acc_delta:+.1%})")
        print(f"  Sentence Accuracy: {reranker_results['sentence_accuracy']:.1%} ({sent_acc_delta:+.1%})")
        print(f"  WER:               {reranker_results['wer']:.3f} ({wer_reduction:.1%} reduction)")

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
