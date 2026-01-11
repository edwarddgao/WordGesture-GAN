#!/usr/bin/env python3
"""
Contrastive Gesture Encoder Training on Modal

Trains embeddings where same-word gestures are close, different-word gestures are far.

Usage:
    python train_contrastive.py                    # Train 100 epochs (T4 default)
    python train_contrastive.py --epochs 50        # Train 50 epochs
    python train_contrastive.py --no-resume        # Start fresh
    python train_contrastive.py --augment-min-jerk # Train with min jerk augmentation
    python train_contrastive.py --gpu L40S         # Use faster GPU (higher cost)
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

from src.shared.config import ModelConfig, TrainingConfig, ModalConfig
from src.shared.keyboard import QWERTYKeyboard
from src.shared.data import load_dataset_from_zip
from src.contrastive.model import ContrastiveConfig, ContrastiveEncoder
from src.contrastive.dataset import create_contrastive_datasets, create_contrastive_data_loader
from src.contrastive.trainer import ContrastiveTrainer

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

# Enable cuDNN benchmarking for faster convolutions
torch.backends.cudnn.benchmark = True

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
train_loader = create_contrastive_data_loader(train_dataset, contrastive_config, shuffle=True, num_workers=8)
test_loader = create_contrastive_data_loader(test_dataset, contrastive_config, shuffle=False, num_workers=8)

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


async def run_train_sandbox(
    num_epochs: int = 100,
    resume: bool = True,
    augment_min_jerk: bool = False,
    min_jerk_noise: float = 0.02,
    min_jerk_augmentations: int = 2,
    gpu: str = 'T4'
):
    """Run training in a Sandbox with real-time stdout streaming."""
    sb = modal.Sandbox.create(
        "python", "-c", TRAIN_SCRIPT,
        str(num_epochs), str(int(resume)), str(int(augment_min_jerk)), str(min_jerk_noise), str(min_jerk_augmentations),
        app=app,
        image=image,
        gpu=gpu,
        volumes={'/data': volume},
        timeout=7200,
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
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--augment-min-jerk', action='store_true',
                        help='Augment training data with minimum jerk trajectories')
    parser.add_argument('--min-jerk-noise', type=float, default=0.02,
                        help='Std dev of Gaussian noise on key positions for min jerk (default: 0.02)')
    parser.add_argument('--min-jerk-augmentations', type=int, default=2,
                        help='Number of min jerk samples per word (default: 2)')
    parser.add_argument('--gpu', type=str, default='T4',
                        choices=['T4', 'L4', 'A10G', 'L40S', 'A100'],
                        help='GPU type (default: T4, most cost-effective for contrastive)')
    args = parser.parse_args()

    async with app.run():
        aug_str = f" with min jerk augmentation ({args.min_jerk_augmentations}x)" if args.augment_min_jerk else ""
        print(f"Training for {args.epochs} epochs on {args.gpu} (resume={not args.no_resume}){aug_str}...")
        return_code = await run_train_sandbox(
            args.epochs,
            resume=not args.no_resume,
            augment_min_jerk=args.augment_min_jerk,
            min_jerk_noise=args.min_jerk_noise,
            min_jerk_augmentations=args.min_jerk_augmentations,
            gpu=args.gpu
        )

    print(f"\nCompleted with return code: {return_code}")


if __name__ == '__main__':
    asyncio.run(main())
