"""Evaluate the two-tower contrastive model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from shared import KeyboardLayout, build_word_prototype, get_device, load_dataset

from .models import TwoTowerModel


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def evaluate_retrieval(
    model: TwoTowerModel,
    gestures: np.ndarray,
    words: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 256,
    return_details: bool = False,
) -> Dict:
    """Evaluate retrieval accuracy.

    For each gesture, rank all vocabulary words and compute Recall@K.

    Args:
        return_details: If True, also return vocab, proto_embs, predictions, etc.
    """
    model.eval()
    vocab = sorted(set(words))

    # Precompute prototype embeddings for entire vocabulary
    with torch.no_grad():
        proto_list = []
        for w in vocab:
            proto = build_word_prototype(w, n_points, layout)[:, :2]  # (n_points, 2)
            proto_list.append(proto)
        protos = torch.from_numpy(np.stack(proto_list, axis=0)).to(device)  # (V, n_points, 2)
        proto_embs = model.encode_prototype(protos)  # (V, D)

    # Compute gesture embeddings in batches
    gesture_embs_list = []
    with torch.no_grad():
        for i in range(0, len(gestures), batch_size):
            batch = torch.from_numpy(gestures[i : i + batch_size]).to(device)
            emb = model.encode_gesture(batch)
            gesture_embs_list.append(emb)
        gesture_embs = torch.cat(gesture_embs_list, dim=0)  # (N, D)

    # Compute similarity matrix
    similarities = gesture_embs @ proto_embs.T  # (N, V)

    # Get rankings
    rankings = similarities.argsort(dim=1, descending=True)  # (N, V)

    # Compute Recall@K
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    ground_truth = torch.tensor([word_to_idx[w] for w in words], device=device)

    results = {}
    for k in k_values:
        top_k = rankings[:, :k]
        hits = (top_k == ground_truth.unsqueeze(1)).any(dim=1)
        recall = hits.float().mean().item()
        results[f"recall@{k}"] = recall

    # Mean Reciprocal Rank
    ranks = (rankings == ground_truth.unsqueeze(1)).float().argmax(dim=1) + 1
    mrr = (1.0 / ranks.float()).mean().item()
    results["mrr"] = mrr

    if return_details:
        # Top-1 predictions
        top1_indices = rankings[:, 0]
        predictions = [vocab[idx.item()] for idx in top1_indices]
        # Prediction scores
        pred_scores = similarities[torch.arange(len(similarities)), top1_indices].cpu().numpy()
        # Ground truth scores
        gt_scores = similarities[torch.arange(len(similarities)), ground_truth].cpu().numpy()

        results["_details"] = {
            "vocab": vocab,
            "word_to_idx": word_to_idx,
            "proto_embs": proto_embs,
            "predictions": predictions,
            "ground_truth_words": words,
            "pred_scores": pred_scores,
            "gt_scores": gt_scores,
            "ranks": ranks.cpu().numpy(),
        }

    return results


def analyze_confusions(
    results: Dict,
    top_n: int = 20,
) -> Dict:
    """Analyze prediction confusions.

    Args:
        results: Results from evaluate_retrieval with return_details=True
        top_n: Number of top confusions to return

    Returns:
        Dictionary with confusion analysis
    """
    details = results["_details"]
    predictions = details["predictions"]
    ground_truth = details["ground_truth_words"]
    pred_scores = details["pred_scores"]
    gt_scores = details["gt_scores"]

    # Count confusions (gt -> pred)
    confusion_counts = Counter()
    confusion_examples = defaultdict(list)

    for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        if gt != pred:
            confusion_counts[(gt, pred)] += 1
            if len(confusion_examples[(gt, pred)]) < 3:  # Keep up to 3 examples
                confusion_examples[(gt, pred)].append({
                    "gt_score": float(gt_scores[i]),
                    "pred_score": float(pred_scores[i]),
                    "margin": float(pred_scores[i] - gt_scores[i]),
                })

    # Get top confusions
    top_confusions = []
    for (gt, pred), count in confusion_counts.most_common(top_n):
        examples = confusion_examples[(gt, pred)]
        avg_margin = np.mean([ex["margin"] for ex in examples])
        top_confusions.append({
            "ground_truth": gt,
            "predicted": pred,
            "count": count,
            "avg_margin": float(avg_margin),
            "examples": examples,
        })

    # Per-word recall
    word_correct = defaultdict(int)
    word_total = defaultdict(int)
    for gt, pred in zip(ground_truth, predictions):
        word_total[gt] += 1
        if gt == pred:
            word_correct[gt] += 1

    word_recall = {
        w: word_correct[w] / word_total[w]
        for w in word_total
    }
    worst_words = sorted(word_recall.items(), key=lambda x: (x[1], -word_total[x[0]]))[:top_n]

    # Similarity statistics
    correct_mask = np.array([gt == pred for gt, pred in zip(ground_truth, predictions)])
    correct_scores = gt_scores[correct_mask]
    incorrect_gt_scores = gt_scores[~correct_mask]
    incorrect_pred_scores = pred_scores[~correct_mask]

    return {
        "top_confusions": top_confusions,
        "worst_words": [
            {"word": w, "recall": r, "count": word_total[w]}
            for w, r in worst_words
        ],
        "similarity_stats": {
            "correct_mean": float(np.mean(correct_scores)) if len(correct_scores) > 0 else 0,
            "correct_std": float(np.std(correct_scores)) if len(correct_scores) > 0 else 0,
            "incorrect_gt_mean": float(np.mean(incorrect_gt_scores)) if len(incorrect_gt_scores) > 0 else 0,
            "incorrect_pred_mean": float(np.mean(incorrect_pred_scores)) if len(incorrect_pred_scores) > 0 else 0,
            "avg_margin_when_wrong": float(np.mean(incorrect_pred_scores - incorrect_gt_scores)) if len(incorrect_gt_scores) > 0 else 0,
        },
        "n_errors": int((~correct_mask).sum()),
        "n_total": len(ground_truth),
    }


def analyze_hard_negatives(
    results: Dict,
    top_n: int = 20,
) -> Dict:
    """Analyze prototype similarities to find potential hard negatives.

    Args:
        results: Results from evaluate_retrieval with return_details=True
        top_n: Number of hard negative pairs to return

    Returns:
        Dictionary with hard negative analysis
    """
    details = results["_details"]
    vocab = details["vocab"]
    proto_embs = details["proto_embs"]

    # Compute prototype-to-prototype similarities
    proto_sim = proto_embs @ proto_embs.T  # (V, V)

    # Zero out diagonal
    proto_sim.fill_diagonal_(-float("inf"))

    # Find most similar pairs
    V = len(vocab)
    similarities_flat = proto_sim.cpu().numpy().flatten()
    top_indices = np.argsort(similarities_flat)[::-1][:top_n * 2]  # Get more to filter duplicates

    hard_pairs = []
    seen = set()
    for idx in top_indices:
        i, j = idx // V, idx % V
        if (j, i) in seen:  # Skip duplicate pairs
            continue
        seen.add((i, j))
        hard_pairs.append({
            "word1": vocab[i],
            "word2": vocab[j],
            "similarity": float(proto_sim[i, j]),
        })
        if len(hard_pairs) >= top_n:
            break

    # Similarity distribution stats
    upper_tri = proto_sim.cpu().numpy()[np.triu_indices(V, k=1)]

    return {
        "hard_pairs": hard_pairs,
        "proto_similarity_stats": {
            "mean": float(np.mean(upper_tri)),
            "std": float(np.std(upper_tri)),
            "max": float(np.max(upper_tri)),
            "p95": float(np.percentile(upper_tri, 95)),
            "p99": float(np.percentile(upper_tri, 99)),
        },
    }


def evaluate_oov(
    model: TwoTowerModel,
    test_gestures: np.ndarray,
    test_words: List[str],
    train_words: List[str],
    layout: KeyboardLayout,
    n_points: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on OOV words (words not in training set)."""
    train_vocab = set(train_words)
    oov_mask = np.array([w not in train_vocab for w in test_words])

    oov_gestures = test_gestures[oov_mask]
    oov_words = [w for w, is_oov in zip(test_words, oov_mask) if is_oov]

    if not oov_words:
        return {"oov_count": 0}

    results = evaluate_retrieval(
        model, oov_gestures, oov_words, layout, n_points, device
    )
    results["oov_count"] = len(oov_words)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-tower contrastive model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--analyze", action="store_true", help="Run detailed confusion analysis")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top items in analysis")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    n_points = cfg["data"]["n_points"]
    layout = KeyboardLayout()

    # Load test data
    test_gestures, test_words = load_dataset(args.data_dir / "test.npz")
    train_gestures, train_words = load_dataset(args.data_dir / "train.npz")

    print(f"Test samples: {len(test_gestures)}, vocab size: {len(set(test_words))}")

    # Evaluate on all test data
    print("\nEvaluating on all test data...")
    all_results = evaluate_retrieval(
        model, test_gestures, test_words, layout, n_points, device,
        return_details=args.analyze,
    )
    print(f"  Recall@1:  {all_results['recall@1']:.4f}")
    print(f"  Recall@5:  {all_results['recall@5']:.4f}")
    print(f"  Recall@10: {all_results['recall@10']:.4f}")
    print(f"  MRR:       {all_results['mrr']:.4f}")

    # Evaluate on OOV words
    print("\nEvaluating on OOV words...")
    oov_results = evaluate_oov(model, test_gestures, test_words, train_words, layout, n_points, device)
    if oov_results["oov_count"] > 0:
        print(f"  OOV count: {oov_results['oov_count']}")
        print(f"  Recall@1:  {oov_results['recall@1']:.4f}")
        print(f"  Recall@5:  {oov_results['recall@5']:.4f}")
        print(f"  Recall@10: {oov_results['recall@10']:.4f}")
        print(f"  MRR:       {oov_results['mrr']:.4f}")
    else:
        print("  No OOV words in test set")

    # Run detailed analysis if requested
    if args.analyze:
        print("\n" + "=" * 60)
        print("CONFUSION ANALYSIS")
        print("=" * 60)
        confusion = analyze_confusions(all_results, top_n=args.top_n)

        print(f"\nTotal errors: {confusion['n_errors']} / {confusion['n_total']} "
              f"({confusion['n_errors']/confusion['n_total']:.1%})")

        print("\n--- Similarity Statistics ---")
        stats = confusion["similarity_stats"]
        print(f"  Correct predictions - mean score: {stats['correct_mean']:.4f} ± {stats['correct_std']:.4f}")
        print(f"  Wrong predictions   - GT score:   {stats['incorrect_gt_mean']:.4f}")
        print(f"  Wrong predictions   - Pred score: {stats['incorrect_pred_mean']:.4f}")
        print(f"  Avg margin when wrong: {stats['avg_margin_when_wrong']:.4f}")

        print(f"\n--- Top {args.top_n} Confusions (GT -> Predicted) ---")
        for i, conf in enumerate(confusion["top_confusions"], 1):
            print(f"  {i:2d}. '{conf['ground_truth']}' -> '{conf['predicted']}' "
                  f"(n={conf['count']}, margin={conf['avg_margin']:.4f})")

        print(f"\n--- Words with Worst Recall ---")
        for i, w in enumerate(confusion["worst_words"], 1):
            print(f"  {i:2d}. '{w['word']}': {w['recall']:.1%} ({w['count']} samples)")

        print("\n" + "=" * 60)
        print("HARD NEGATIVE ANALYSIS (Prototype Similarities)")
        print("=" * 60)
        hard_neg = analyze_hard_negatives(all_results, top_n=args.top_n)

        print("\n--- Prototype Similarity Distribution ---")
        pstats = hard_neg["proto_similarity_stats"]
        print(f"  Mean: {pstats['mean']:.4f}, Std: {pstats['std']:.4f}")
        print(f"  Max:  {pstats['max']:.4f}, P95: {pstats['p95']:.4f}, P99: {pstats['p99']:.4f}")

        print(f"\n--- Most Similar Word Pairs (Potential Hard Negatives) ---")
        for i, pair in enumerate(hard_neg["hard_pairs"], 1):
            print(f"  {i:2d}. '{pair['word1']}' <-> '{pair['word2']}' (sim={pair['similarity']:.4f})")

        # Add analysis to results for JSON output
        all_results["confusion_analysis"] = confusion
        all_results["hard_negative_analysis"] = hard_neg

    # Save results
    if args.output:
        # Remove internal details before saving
        save_results = {k: v for k, v in all_results.items() if not k.startswith("_")}
        results = {"all": save_results, "oov": oov_results}
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
