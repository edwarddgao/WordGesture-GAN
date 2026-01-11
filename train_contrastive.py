#!/usr/bin/env python3
"""
Contrastive Gesture Encoder Training on Modal

Trains embeddings where same-word gestures are close, different-word gestures are far.

Usage:
    python train_contrastive.py                    # Train 100 epochs
    python train_contrastive.py --epochs 50        # Train 50 epochs
    python train_contrastive.py --eval-only        # Run evaluation (reports both real and min jerk centroids)
    python train_contrastive.py --no-resume        # Start fresh
    python train_contrastive.py --augment-min-jerk # Train with min jerk augmentation
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('contrastive-gesture-encoder')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'matplotlib', 'scikit-learn')
    .add_local_python_source('src')
)


# ============================================================================
# Training Script (embedded for Modal Sandbox)
# ============================================================================

TRAIN_SCRIPT = '''
import sys
import torch
import numpy as np
from pathlib import Path

from src.config import ModelConfig, TrainingConfig, ModalConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip
from src.contrastive_model import ContrastiveConfig, ContrastiveEncoder
from src.contrastive_dataset import create_contrastive_datasets, create_contrastive_data_loader
from src.contrastive_trainer import ContrastiveTrainer

def log(msg):
    print(msg, flush=True)

# Parse args
num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
resume = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
augment_min_jerk = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
min_jerk_noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
min_jerk_augmentations = int(sys.argv[5]) if len(sys.argv) > 5 else 2

device = 'cuda'
modal_config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()

# Contrastive config
contrastive_config = ContrastiveConfig(
    embedding_dim=64,
    lstm_hidden_dim=64,
    num_lstm_layers=2,
    temperature=0.07,
    learning_rate=1e-3,
    batch_words=32,
    gestures_per_word=2,
    num_epochs=num_epochs,
)

log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'Training contrastive encoder for {num_epochs} epochs (resume={resume})')
log(f'Config: embedding_dim={contrastive_config.embedding_dim}, batch={contrastive_config.batch_words}x{contrastive_config.gestures_per_word}')
if augment_min_jerk:
    log(f'Min jerk augmentation: ENABLED ({min_jerk_augmentations}x per word, noise={min_jerk_noise})')

# Load data
log('[1/4] Loading data...')
keyboard = QWERTYKeyboard()
gestures_by_word, _ = load_dataset_from_zip(
    modal_config.data_path, keyboard, model_config, training_config
)
log(f'  Loaded {sum(len(g) for g in gestures_by_word.values())} gestures from {len(gestures_by_word)} words')

# Create datasets
log('[2/4] Creating datasets...')
train_dataset, test_dataset = create_contrastive_datasets(
    gestures_by_word,
    train_ratio=0.8,
    min_gestures_per_word=2,
    seed=42,
    augment_min_jerk=augment_min_jerk,
    keyboard=keyboard if augment_min_jerk else None,
    min_jerk_augmentations=min_jerk_augmentations,
    min_jerk_noise=min_jerk_noise
)
log(f'  Train: {len(train_dataset)} gestures from {train_dataset.get_num_words()} words')
log(f'  Test: {len(test_dataset)} gestures from {test_dataset.get_num_words()} words')

# Create data loaders
train_loader = create_contrastive_data_loader(train_dataset, contrastive_config, shuffle=True, num_workers=2)
test_loader = create_contrastive_data_loader(test_dataset, contrastive_config, shuffle=False, num_workers=2)

# Create trainer
log('[3/4] Creating trainer...')
trainer = ContrastiveTrainer(contrastive_config, device=device)

# Resume from checkpoint
checkpoint_dir = Path(modal_config.checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / 'contrastive_latest.pt'

if resume and checkpoint_path.exists():
    log(f'  Loading checkpoint from {checkpoint_path}...')
    trainer.load_checkpoint(str(checkpoint_path))
    log(f'  Resumed from epoch {trainer.current_epoch}, best recall@1={trainer.best_recall:.4f}')

# Save callback
def save_callback(trainer_obj, epoch, metrics):
    torch.save({
        'encoder_state_dict': trainer_obj.encoder.state_dict(),
        'optimizer_state_dict': trainer_obj.optimizer.state_dict(),
        'config': trainer_obj.config,
        'epoch': epoch,
        'best_recall': trainer_obj.best_recall,
        'metrics': metrics
    }, checkpoint_path)
    log(f'  Saved checkpoint to {checkpoint_path}')

# Train
log('[4/4] Training...')
history = trainer.fit(
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    log_every=20,
    eval_every=5,
    save_callback=save_callback
)

# Final save
trainer.save_checkpoint(str(checkpoint_path))
log(f'Training complete. Best recall@1: {trainer.best_recall:.4f}')

# Print final metrics
log('')
log('=' * 60)
log('Final Results:')
log('=' * 60)
if 'test_recall@1' in history:
    log(f"  Recall@1:  {history['test_recall@1'][-1]:.4f}")
if 'test_recall@5' in history:
    log(f"  Recall@5:  {history['test_recall@5'][-1]:.4f}")
if 'test_recall@10' in history:
    log(f"  Recall@10: {history['test_recall@10'][-1]:.4f}")
if 'test_mAP' in history:
    log(f"  mAP:       {history['test_mAP'][-1]:.4f}")
log('=' * 60)
'''


# ============================================================================
# Evaluation Script
# ============================================================================

EVAL_SCRIPT = '''
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random

from src.config import ModelConfig, TrainingConfig, ModalConfig
from src.keyboard import QWERTYKeyboard, MinimumJerkModel
from src.data import load_dataset_from_zip
from src.contrastive_model import ContrastiveConfig, ContrastiveEncoder
from src.contrastive_trainer import ContrastiveTrainer

def log(msg):
    print(msg, flush=True)

device = 'cuda'
modal_config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()

contrastive_config = ContrastiveConfig(
    embedding_dim=64,
    lstm_hidden_dim=64,
    num_lstm_layers=2,
)

log(f'GPU: {torch.cuda.get_device_name(0)}')

# Load checkpoint
checkpoint_path = Path(f'{modal_config.checkpoint_dir}/contrastive_latest.pt')
if not checkpoint_path.exists():
    log(f'ERROR: No checkpoint found at {checkpoint_path}')
    sys.exit(1)

log('[1/6] Loading model...')
trainer = ContrastiveTrainer(contrastive_config, device=device)
trainer.load_checkpoint(str(checkpoint_path))
encoder = trainer.get_encoder()
encoder.eval()
log(f'  Loaded checkpoint from epoch {trainer.current_epoch}')

# Load data
log('[2/6] Loading data...')
keyboard = QWERTYKeyboard()
gestures_by_word, _ = load_dataset_from_zip(
    modal_config.data_path, keyboard, model_config, training_config
)

# Split into train/test (same as training: 80/20)
min_gestures = 2
eligible_words = [w for w, g in gestures_by_word.items() if len(g) >= min_gestures]
random.seed(42)
random.shuffle(eligible_words)

split_idx = int(len(eligible_words) * 0.8)
train_words = set(eligible_words[:split_idx])
test_words = eligible_words[split_idx:]
log(f'  Train words: {len(train_words)}, Test words: {len(test_words)}')

# Fit MinimumJerkModel on training data
log('[3/6] Fitting MinimumJerkModel on training data...')
train_gestures_by_word = {w: gestures_by_word[w] for w in train_words}
min_jerk_model = MinimumJerkModel(keyboard)
min_jerk_model.fit(train_gestures_by_word, verbose=True)

# Embed all test gestures
log('[4/6] Embedding test gestures...')
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

# Compute REAL centroids
log('[5/6] Computing real centroids...')
real_centroids = {}
for word in test_words:
    embeds = []
    for g in gestures_by_word[word]:
        tensor = torch.FloatTensor(g).unsqueeze(0).to(device)
        emb = encoder(tensor).squeeze(0)
        embeds.append(emb)
    centroid = torch.stack(embeds).mean(dim=0)
    real_centroids[word] = F.normalize(centroid, p=2, dim=0)

# Test different numbers of min jerk samples
word_list = list(test_words)
real_matrix = torch.stack([real_centroids[w] for w in word_list])

# Compute real recall@1 once
log('[6/6] Computing metrics...')
sim_real = query_embeddings @ real_matrix.T
_, topk_real = sim_real.topk(1, dim=1)
correct_real = sum(1 for i, word in enumerate(query_words) if word_list.index(word) in topk_real[i].cpu().numpy())
real_recall1 = correct_real / len(query_words)
log(f'  Real centroids recall@1: {real_recall1:.4f}')

# Test FITTED min jerk model with different sample counts
sample_counts = [5, 10, 20, 50]
log('')
log('=' * 60)
log('Fitted MinimumJerkModel Centroid Quality vs Sample Count:')
log('=' * 60)
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
    correct_mj = sum(1 for i, word in enumerate(query_words) if word_list.index(word) in topk_minjerk[i].cpu().numpy())
    mj_recall1 = correct_mj / len(query_words)
    gap = real_recall1 - mj_recall1
    log(f'  {num_samples:3d}         {mj_recall1:.4f}      {gap:+.4f}')

log('=' * 60)
'''


async def run_train_sandbox(
    num_epochs: int = 100,
    resume: bool = True,
    augment_min_jerk: bool = False,
    min_jerk_noise: float = 0.02,
    min_jerk_augmentations: int = 2
):
    """Run training in a Sandbox with real-time stdout streaming."""
    sb = modal.Sandbox.create(
        "python", "-c", TRAIN_SCRIPT,
        str(num_epochs), str(int(resume)), str(int(augment_min_jerk)), str(min_jerk_noise), str(min_jerk_augmentations),
        app=app,
        image=image,
        gpu='T4',
        volumes={'/data': volume},
        timeout=7200,
    )

    for line in sb.stdout:
        print(line, end='', flush=True)

    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


async def run_eval_sandbox():
    """Run evaluation in a Sandbox with real-time stdout streaming."""
    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT,
        app=app,
        image=image,
        gpu='T4',
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
    parser = argparse.ArgumentParser(description='Train contrastive gesture encoder on Modal')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--augment-min-jerk', action='store_true',
                        help='Augment training data with minimum jerk trajectories')
    parser.add_argument('--min-jerk-noise', type=float, default=0.02,
                        help='Std dev of Gaussian noise on key positions for min jerk (default: 0.02)')
    parser.add_argument('--min-jerk-augmentations', type=int, default=2,
                        help='Number of min jerk samples per word (default: 2)')
    args = parser.parse_args()

    async with app.run():
        if args.eval_only:
            print("Running evaluation...")
            return_code = await run_eval_sandbox()
        else:
            aug_str = f" with min jerk augmentation ({args.min_jerk_augmentations}x)" if args.augment_min_jerk else ""
            print(f"Training for {args.epochs} epochs (resume={not args.no_resume}){aug_str}...")
            return_code = await run_train_sandbox(
                args.epochs,
                resume=not args.no_resume,
                augment_min_jerk=args.augment_min_jerk,
                min_jerk_noise=args.min_jerk_noise,
                min_jerk_augmentations=args.min_jerk_augmentations
            )

    print(f"\nCompleted with return code: {return_code}")


if __name__ == '__main__':
    asyncio.run(main())
