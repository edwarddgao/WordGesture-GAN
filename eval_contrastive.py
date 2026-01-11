#!/usr/bin/env python3
"""
Evaluate contrastive gesture encoder on Modal.

Usage:
    python eval_contrastive.py                    # recall@k, mAP
    python eval_contrastive.py --centroids        # centroid comparison
    python eval_contrastive.py --tsne             # t-SNE visualization
    python eval_contrastive.py --query hello      # similarity search
    python eval_contrastive.py --gpu L4           # Use different GPU
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('contrastive-eval')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'matplotlib', 'scikit-learn')
    .add_local_python_source('src')
)


# ============================================================================
# Evaluation Script (embedded for Modal Sandbox)
# ============================================================================

EVAL_SCRIPT = '''
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter

from src.shared.config import ModelConfig, TrainingConfig, ModalConfig
from src.shared.keyboard import QWERTYKeyboard, MinimumJerkModel
from src.shared.data import load_dataset_from_zip
from src.contrastive.model import ContrastiveConfig, ContrastiveEncoder
from src.contrastive.dataset import create_contrastive_datasets, create_contrastive_data_loader


def log(msg):
    print(msg, flush=True)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained contrastive encoder."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', ContrastiveConfig())
    encoder = ContrastiveEncoder(config).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    return encoder, config


def evaluate_recall(encoder, test_loader, device, k_values=(1, 5, 10, 20)):
    """Evaluate recall@k on test set."""
    encoder.eval()
    all_embeddings = []
    all_labels = []
    all_words = []

    with torch.no_grad():
        for gestures, labels, words in test_loader:
            gestures = gestures.to(device)
            embeddings = encoder(gestures).cpu()
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            all_words.extend(words)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    n = embeddings.size(0)

    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-float('inf'))

    max_k = max(k_values)
    _, topk_indices = similarity.topk(max_k, dim=1)

    results = {}
    for k in k_values:
        topk = topk_indices[:, :k]
        query_labels = labels.unsqueeze(1).expand(-1, k)
        neighbor_labels = labels[topk]
        matches = (query_labels == neighbor_labels).any(dim=1)
        results[f'recall@{k}'] = matches.float().mean().item()

    # mAP
    ap_scores = []
    for i in range(n):
        query_label = labels[i].item()
        neighbors = topk_indices[i]
        neighbor_labels = labels[neighbors]
        correct = (neighbor_labels == query_label).float()
        precision_at_k = torch.cumsum(correct, dim=0) / torch.arange(1, max_k + 1).float()
        ap = (precision_at_k * correct).sum() / correct.sum() if correct.sum() > 0 else 0
        ap_scores.append(ap)
    results['mAP'] = float(np.mean(ap_scores))

    return results, embeddings, labels, all_words


def similarity_search(encoder, query_gesture, database_embeddings, database_words, top_k=10, device='cuda'):
    """Find most similar gestures to query."""
    encoder.eval()
    with torch.no_grad():
        query_tensor = torch.FloatTensor(query_gesture).unsqueeze(0).to(device)
        query_embedding = encoder(query_tensor).cpu()

    similarities = (query_embedding @ database_embeddings.T).squeeze(0)
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'index': idx.item(),
            'word': database_words[idx],
            'similarity': similarities[idx].item()
        })
    return results


def create_tsne_plot(embeddings, words, output_path, n_samples=2000, top_n_words=20):
    """Create t-SNE visualization of embeddings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        words = [words[i] for i in indices]

    word_counts = Counter(words)
    top_words = [w for w, _ in word_counts.most_common(top_n_words)]
    word_to_color = {w: i for i, w in enumerate(top_words)}

    log(f'Running t-SNE on {len(embeddings)} samples...')
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 12))

    other_mask = np.array([w not in word_to_color for w in words])
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c='lightgray', alpha=0.3, s=5, label='other')

    for word, color_idx in word_to_color.items():
        mask = np.array([w == word for w in words])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[plt.cm.tab20(color_idx)], alpha=0.7, s=20, label=word)

    ax.set_title(f't-SNE of Gesture Embeddings (n={len(embeddings)}, top {top_n_words} words colored)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    log(f'Saved t-SNE plot to {output_path}')
    plt.close(fig)


def evaluate_centroids(encoder, gestures_by_word, keyboard, device, sample_counts=(5, 10, 20, 50)):
    """Compare real centroids vs min jerk centroids."""
    encoder.eval()

    min_gestures = 2
    eligible_words = [w for w, g in gestures_by_word.items() if len(g) >= min_gestures]
    random.seed(42)
    random.shuffle(eligible_words)

    split_idx = int(len(eligible_words) * 0.8)
    train_words = set(eligible_words[:split_idx])
    test_words = eligible_words[split_idx:]
    log(f'  Train words: {len(train_words)}, Test words: {len(test_words)}')

    log('Fitting MinimumJerkModel on training data...')
    train_gestures_by_word = {w: gestures_by_word[w] for w in train_words}
    min_jerk_model = MinimumJerkModel(keyboard)
    min_jerk_model.fit(train_gestures_by_word, verbose=True)

    log('Embedding test gestures...')
    query_embeddings = []
    query_words = []

    with torch.no_grad():
        for word in test_words:
            for g in gestures_by_word[word]:
                tensor = torch.FloatTensor(g).unsqueeze(0).to(device)
                emb = encoder(tensor).squeeze(0)
                query_embeddings.append(emb)
                query_words.append(word)

    query_embeddings = torch.stack(query_embeddings)
    log(f'  Embedded {len(query_embeddings)} gestures')

    log('Computing real centroids...')
    real_centroids = {}
    with torch.no_grad():
        for word in test_words:
            embeds = []
            for g in gestures_by_word[word]:
                tensor = torch.FloatTensor(g).unsqueeze(0).to(device)
                emb = encoder(tensor).squeeze(0)
                embeds.append(emb)
            centroid = torch.stack(embeds).mean(dim=0)
            real_centroids[word] = F.normalize(centroid, p=2, dim=0)

    word_list = list(test_words)
    real_matrix = torch.stack([real_centroids[w] for w in word_list])

    log('Computing metrics...')
    sim_real = query_embeddings @ real_matrix.T
    _, topk_real = sim_real.topk(1, dim=1)
    correct_real = sum(1 for i, word in enumerate(query_words)
                       if word_list.index(word) in topk_real[i].cpu().numpy())
    real_recall1 = correct_real / len(query_words)

    results = {'real_recall@1': real_recall1}

    log('')
    log('=' * 60)
    log('Centroid Quality: Real vs Min Jerk')
    log('=' * 60)
    log(f'  Real centroids recall@1: {real_recall1:.4f}')
    log('')
    log('  Samples    recall@1    Gap vs Real')

    for num_samples in sample_counts:
        minjerk_centroids = {}
        with torch.no_grad():
            for word in test_words:
                trajs = []
                for _ in range(num_samples):
                    traj = min_jerk_model.generate_trajectory(
                        word, num_points=128, include_midpoints=True
                    )
                    trajs.append(torch.FloatTensor(traj).unsqueeze(0).to(device))
                trajs = torch.cat(trajs, dim=0)
                embeddings = encoder(trajs)
                centroid = embeddings.mean(dim=0)
                minjerk_centroids[word] = F.normalize(centroid, p=2, dim=0)

        minjerk_matrix = torch.stack([minjerk_centroids[w] for w in word_list])
        sim_minjerk = query_embeddings @ minjerk_matrix.T
        _, topk_minjerk = sim_minjerk.topk(1, dim=1)
        correct_mj = sum(1 for i, word in enumerate(query_words)
                         if word_list.index(word) in topk_minjerk[i].cpu().numpy())
        mj_recall1 = correct_mj / len(query_words)
        gap = real_recall1 - mj_recall1
        log(f'  {num_samples:3d}         {mj_recall1:.4f}      {gap:+.4f}')
        results[f'minjerk_{num_samples}_recall@1'] = mj_recall1

    log('=' * 60)
    return results


# Parse args
do_centroids = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
do_tsne = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
query_word = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'None' else None

device = 'cuda'
modal_config = ModalConfig()

log(f'GPU: {torch.cuda.get_device_name(0)}')
torch.backends.cudnn.benchmark = True

# Load model
log('Loading model...')
checkpoint_path = Path(modal_config.checkpoint_dir) / 'contrastive_latest.pt'
encoder, config = load_model(str(checkpoint_path), device)
log(f'  Embedding dim: {config.embedding_dim}')

# Load data
log('Loading data...')
keyboard = QWERTYKeyboard()
model_config = ModelConfig()
training_config = TrainingConfig()
gestures_by_word, _ = load_dataset_from_zip(modal_config.data_path, keyboard, model_config, training_config)

_, test_dataset = create_contrastive_datasets(
    gestures_by_word, train_ratio=0.8, min_gestures_per_word=2, seed=42
)
test_loader = create_contrastive_data_loader(test_dataset, config, shuffle=False, num_workers=8)
log(f'  Test set: {len(test_dataset)} gestures')

# Evaluate
log('Evaluating...')
metrics, embeddings, labels, words = evaluate_recall(encoder, test_loader, device)

log('')
log('=' * 50)
log('Evaluation Results:')
log('=' * 50)
for key, value in sorted(metrics.items()):
    log(f'  {key}: {value:.4f}')
log('=' * 50)

# Similarity search
if query_word:
    log(f'\\nSimilarity search for word: "{query_word}"')
    query_indices = [i for i, w in enumerate(words) if w == query_word]
    if query_indices:
        query_idx = query_indices[0]
        query_gesture = test_dataset.gestures[query_idx]
        results = similarity_search(encoder, query_gesture, embeddings, words, top_k=10, device=device)
        log(f'Query word: {query_word}')
        log('Top 10 nearest neighbors:')
        for i, r in enumerate(results):
            match = 'Y' if r['word'] == query_word else ' '
            log(f"  {i+1}. [{match}] {r['word']:15} (sim: {r['similarity']:.4f})")
    else:
        log(f'  Word "{query_word}" not found in test set')

# t-SNE
if do_tsne:
    log('\\nGenerating t-SNE visualization...')
    embeddings_np = embeddings.numpy()
    output_path = str(Path(modal_config.checkpoint_dir) / 'contrastive_tsne.png')
    create_tsne_plot(embeddings_np, words, output_path)

# Centroids
if do_centroids:
    log('\\nEvaluating centroid quality...')
    evaluate_centroids(encoder, gestures_by_word, keyboard, device)

log('\\nDone.')
'''


async def run_eval_sandbox(
    do_centroids: bool = False,
    do_tsne: bool = False,
    query_word: str = None,
    gpu: str = 'T4'
):
    """Run evaluation in a Sandbox with real-time stdout streaming."""
    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT,
        str(int(do_centroids)), str(int(do_tsne)), str(query_word),
        app=app,
        image=image,
        gpu=gpu,
        volumes={'/data': volume},
        timeout=3600,
    )

    for line in sb.stdout:
        print(line, end='', flush=True)

    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


async def main():
    parser = argparse.ArgumentParser(description='Evaluate contrastive gesture encoder on Modal')
    parser.add_argument('--centroids', action='store_true', help='Evaluate centroid quality (real vs min jerk)')
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization')
    parser.add_argument('--query', type=str, help='Word to query (demo similarity search)')
    parser.add_argument('--gpu', type=str, default='T4',
                        choices=['T4', 'L4', 'A10G', 'L40S', 'A100'],
                        help='GPU type (default: T4, cheapest for short tasks)')
    args = parser.parse_args()

    async with app.run():
        opts = []
        if args.centroids:
            opts.append('centroids')
        if args.tsne:
            opts.append('t-SNE')
        if args.query:
            opts.append(f'query={args.query}')
        opts_str = f" ({', '.join(opts)})" if opts else ""
        print(f"Evaluating contrastive encoder on {args.gpu}{opts_str}...")

        return_code = await run_eval_sandbox(
            do_centroids=args.centroids,
            do_tsne=args.tsne,
            query_word=args.query,
            gpu=args.gpu
        )

    print(f"\nCompleted with return code: {return_code}")


if __name__ == '__main__':
    asyncio.run(main())
