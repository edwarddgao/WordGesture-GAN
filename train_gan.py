#!/usr/bin/env python3
"""
WordGesture-GAN Training on Modal with Checkpointing

Usage:
    python modal_train.py                    # Train 200 epochs (resumes if checkpoint exists)
    python modal_train.py --epochs 50        # Train 50 epochs
    python modal_train.py --no-resume        # Start fresh, ignore existing checkpoint
"""

import os
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('wordgesture-gan')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'wandb', 'pillow', 'matplotlib', 'fastdtw', 'joblib', 'wordfreq')
    .add_local_python_source('src')
)

# WandB API key injected via Modal Secret (set with: modal secret create wandb-secret WANDB_API_KEY=<your-key>)
wandb_secret = modal.Secret.from_name('wandb-secret')


# ============================================================================
# Training via Sandbox (real-time stdout streaming)
# ============================================================================

TRAIN_SCRIPT = '''
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from dataclasses import asdict
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.amp import GradScaler

from src.shared.config import ModelConfig, TrainingConfig, ModalConfig
from src.shared.keyboard import QWERTYKeyboard
from src.shared.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
from src.gan.trainer import WordGestureGANTrainer
from src.gan.losses import WassersteinLoss
from src.shared.utils import seed_everything, log, train_epoch_with_grad_clip
from src.gan.visualization import create_comparison_figure

# Parse args
num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
resume = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
checkpoint_every = 10
grad_clip_norm = 1.0

device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig(num_epochs=num_epochs, save_every=checkpoint_every)

seed_everything(config.random_seed)

# Enable cuDNN benchmarking for faster convolutions with fixed input sizes
torch.backends.cudnn.benchmark = True

log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'Training for {num_epochs} epochs (resume={resume})')

# Load data
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=training_config.train_ratio, seed=config.random_seed)
train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=training_config.batch_size, num_workers=training_config.num_workers)
log(f'Data: {len(train_ds)} train, {len(test_ds)} test')

# Create trainer
trainer = WordGestureGANTrainer(model_config, training_config, device=device)

# Mixed precision - disabled for now due to multi-optimizer complexity
# The GradScaler doesn't handle 4 separate optimizers well
scaler = None  # Set to GradScaler() to enable AMP

# LR schedulers
schedulers = {
    'G': CosineAnnealingLR(trainer.optimizer_G, T_max=num_epochs, eta_min=1e-5),
    'E': CosineAnnealingLR(trainer.optimizer_E, T_max=num_epochs, eta_min=1e-5),
    'D1': CosineAnnealingLR(trainer.optimizer_D1, T_max=num_epochs, eta_min=1e-5),
    'D2': CosineAnnealingLR(trainer.optimizer_D2, T_max=num_epochs, eta_min=1e-5),
}

# Resume from checkpoint
checkpoint_dir = Path(config.checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / 'latest.pt'

start_epoch = 0
wandb_run_id = None
if resume and checkpoint_path.exists():
    log(f'Loading checkpoint from {checkpoint_path}...')
    ckpt = torch.load(checkpoint_path, map_location=device)
    trainer.load_modal_checkpoint(ckpt)
    start_epoch = ckpt['epoch'] + 1
    wandb_run_id = ckpt.get('wandb_run_id')
    # Restore scaler state if available and scaler is enabled
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    log(f'Resumed from epoch {start_epoch}')
    for _ in range(start_epoch):
        for sched in schedulers.values():
            sched.step()

if start_epoch >= num_epochs:
    log(f'Already trained to epoch {start_epoch}, nothing to do.')
    sys.exit(0)

# Initialize W&B
disc_type = 'temporal' if model_config.use_temporal_disc else 'mlp'
proto_type = 'xy' if not model_config.prototype_has_time else 'xyt'
run_name = f'{disc_type}_{proto_type}_{training_config.lambda_rec}_{training_config.lambda_kld}'
wandb.init(
    project=config.wandb_project,
    name=run_name,
    config={
        'model': asdict(model_config),
        'training': asdict(training_config),
        'num_epochs': num_epochs,
    },
    resume='allow',
    id=wandb_run_id,
)
log(f'W&B run: {wandb.run.name} (id={wandb.run.id})')

# Training loop
import time as _time
log(f'Starting training from epoch {start_epoch}...')
for epoch in range(start_epoch, num_epochs):
    trainer.current_epoch = epoch
    _epoch_start = _time.time()
    epoch_losses = train_epoch_with_grad_clip(trainer, train_loader, grad_clip_norm, model_config, training_config, device, scaler=scaler)
    _epoch_time = _time.time() - _epoch_start

    for sched in schedulers.values():
        sched.step()
    current_lr = schedulers['G'].get_last_lr()[0]

    log(f'Epoch {epoch+1}/{num_epochs} [{_epoch_time:.1f}s] - D1:{epoch_losses["d1_loss"]:.3f} D2:{epoch_losses["d2_loss"]:.3f} C1:{epoch_losses["cycle1_total"]:.3f} C2:{epoch_losses["cycle2_total"]:.3f} LR:{current_lr:.6f}')

    # Log to W&B
    wandb.log({
        'epoch': epoch + 1,
        'loss/d1': epoch_losses['d1_loss'],
        'loss/d2': epoch_losses['d2_loss'],
        'loss/cycle1_total': epoch_losses['cycle1_total'],
        'loss/cycle2_total': epoch_losses['cycle2_total'],
        'learning_rate': current_lr,
    }, step=epoch + 1)

    # Log sample visualizations every 10 epochs
    if (epoch + 1) % 10 == 0:
        trainer.generator.eval()
        with torch.no_grad():
            n_viz = 6
            real_samples, fake_samples, words = [], [], []
            for i in range(min(n_viz, len(test_ds))):
                item = test_ds[i]
                proto = item['prototype'].unsqueeze(0).to(device)
                z = torch.randn(1, model_config.latent_dim, device=device)
                fake = trainer.generator(proto, z).cpu().numpy()[0]
                real_samples.append(item['gesture'].numpy())
                fake_samples.append(fake)
                words.append(item['word'])
            fig = create_comparison_figure(np.array(real_samples), np.array(fake_samples), words)
            wandb.log({'gestures/training_samples': wandb.Image(fig)}, step=epoch + 1)
            plt.close(fig)
        trainer.generator.train()

    # Save checkpoint
    if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
        ckpt = trainer.get_modal_checkpoint_dict()
        ckpt['wandb_run_id'] = wandb.run.id
        if scaler is not None:
            ckpt['scaler'] = scaler.state_dict()
        torch.save(ckpt, checkpoint_dir / 'latest.pt')
        torch.save(ckpt, checkpoint_dir / f'epoch_{epoch+1}.pt')
        log(f'  Checkpoint saved at epoch {epoch+1}')

wandb.finish()
log('Training complete!')
'''


async def run_train_sandbox(num_epochs: int = 200, resume: bool = True, gpu: str = 'L40S'):
    """Run training in a Sandbox with real-time stdout streaming."""
    sb = modal.Sandbox.create(
        "python", "-c", TRAIN_SCRIPT, str(num_epochs), str(int(resume)),
        app=app,
        image=image,
        gpu=gpu,
        volumes={'/data': volume},
        secrets=[wandb_secret],
        timeout=7200,
    )

    for line in sb.stdout:
        print(line, end='', flush=True)

    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


# ============================================================================
# CLI Entry Point
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='WordGesture-GAN Training on Modal')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--gpu', type=str, default='L40S', help='GPU type (L40S, A10G, T4, etc.)')
    args = parser.parse_args()

    async with app.run():
        print(f'Starting training for {args.epochs} epochs on {args.gpu}...')
        returncode = await run_train_sandbox(num_epochs=args.epochs, resume=not args.no_resume, gpu=args.gpu)
        print(f'\nSandbox exited with code: {returncode}')


if __name__ == '__main__':
    asyncio.run(main())
