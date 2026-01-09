#!/usr/bin/env python3
"""
WordGesture-GAN Training on Modal with Checkpointing

Usage:
    python modal_train.py                    # Train 200 epochs (resumes if checkpoint exists)
    python modal_train.py --epochs 50        # Train 50 epochs
    python modal_train.py --eval-only        # Just run evaluation on saved model
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
# Helper Functions (imported from src.utils)
# ============================================================================

from src.utils import log


# ============================================================================
# Evaluation via Sandbox (real-time stdout streaming)
# ============================================================================

EVAL_SCRIPT = '''
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip, create_train_test_split
from src.models import Generator
from src.evaluation import evaluate_all_metrics
from src.visualization import create_comparison_figure, create_overlay_figure
from src.utils import log

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
truncation = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
savgol_window = int(sys.argv[3]) if len(sys.argv) > 3 else 5
precision_k = int(sys.argv[4]) if len(sys.argv) > 4 else 3

device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig(n_samples=n_samples, truncation=truncation, savgol_window=savgol_window, precision_recall_k=precision_k)

log(f'GPU: {torch.cuda.get_device_name(0)}')

# Load checkpoint
checkpoint_path = Path(f'{config.checkpoint_dir}/latest.pt')
if not checkpoint_path.exists():
    log(f'ERROR: No checkpoint found at {checkpoint_path}')
    sys.exit(1)

# Load generator
log('[1/5] Loading generator...')
generator = Generator(model_config).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(ckpt['generator'])
generator.eval()
epoch = ckpt['epoch'] + 1
wandb_run_id = ckpt.get('wandb_run_id')
log(f'  Loaded checkpoint from epoch {epoch}')

# Initialize W&B (resume training run if exists, else create standalone)
if wandb_run_id:
    wandb.init(project=config.wandb_project, id=wandb_run_id, resume='must')
    log(f'  Resumed W&B run: {wandb.run.name}')
else:
    wandb.init(
        project=config.wandb_project,
        name=f'eval_standalone_epoch{epoch}',
        config={
            'model': asdict(model_config),
            'eval': asdict(eval_config),
            'checkpoint_epoch': epoch,
        },
    )
    log(f'  Created standalone W&B run: {wandb.run.name}')

# Load data
log('[2/5] Loading data...')
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)
log(f'  Train: {len(train_ds)}, Test: {len(test_ds)}')

# Generate samples
n = min(n_samples, len(test_ds))
log(f'[3/5] Generating {n} samples (truncation={truncation}, savgol_window={savgol_window}, precision_k={precision_k})...')

real_g, fake_g, words = [], [], []
with torch.no_grad():
    for i in range(n):
        item = test_ds[i]
        proto = item['prototype'].unsqueeze(0).to(device)
        z = torch.randn(1, model_config.latent_dim, device=device) * truncation
        fake = generator(proto, z).cpu().numpy()[0]
        real_g.append(item['gesture'].numpy())
        fake_g.append(fake)
        words.append(item['word'])
        if (i + 1) % 100 == 0:
            log(f'  Generated {i+1}/{n}...')
real_g, fake_g = np.array(real_g), np.array(fake_g)

# Get training data for FID autoencoder
log('[4/5] Preparing training data for FID...')
train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])

# Run all metrics
log('[5/5] Computing metrics...')
results = evaluate_all_metrics(real_g, fake_g, train_g, model_config, eval_config, device)

# Log metrics to W&B summary (final eval metrics)
wandb.summary['eval/l2_wasserstein'] = results['l2_wasserstein']
wandb.summary['eval/dtw_wasserstein'] = results['dtw_wasserstein']
wandb.summary['eval/fid'] = results['fid']
wandb.summary['eval/precision'] = results['precision']
wandb.summary['eval/recall'] = results['recall']
wandb.summary['eval/velocity_corr'] = results['velocity_corr']
wandb.summary['eval/acceleration_corr'] = results['acceleration_corr']
wandb.summary['eval/jerk_fake'] = results['jerk_fake']
wandb.summary['eval/jerk_real'] = results['jerk_real']
wandb.summary['eval/duration_rmse_ms'] = results['duration_rmse_ms']
wandb.summary['eval/ae_reconstruction_loss'] = results['ae_reconstruction_loss']
wandb.summary['eval/ae_test_loss'] = results['ae_test_loss']
wandb.summary['eval/epoch'] = epoch

# Log gesture visualizations
log('Logging gesture visualizations to W&B...')
comparison_fig = create_comparison_figure(real_g[:6], fake_g[:6], words[:6])
wandb.log({'gestures/comparison': wandb.Image(comparison_fig)})
plt.close(comparison_fig)

# Pick a word with multiple samples for overlay
overlay_fig = create_overlay_figure(real_g[:5], fake_g[:5], words[0] if words else 'sample')
wandb.log({'gestures/overlay': wandb.Image(overlay_fig)})
plt.close(overlay_fig)

wandb.finish()

# Print results table
log('')
log('=' * 75)
log(f'{"Metric":<30} {"Ours":>15} {"Paper":>15} {"Notes":>12}')
log('=' * 75)
log(f'{"L2 Wasserstein (x,y)":<30} {results["l2_wasserstein"]:>15.3f} {"4.409":>15} {"lower=better":>12}')
log(f'{"DTW Wasserstein (x,y)":<30} {results["dtw_wasserstein"]:>15.3f} {"2.146":>15} {"lower=better":>12}')
log(f'{"Jerk (fake)":<30} {results["jerk_fake"]:>15.5f} {"0.0058":>15} {"~real":>12}')
log(f'{"Jerk (real)":<30} {results["jerk_real"]:>15.5f} {"0.0066":>15} {"reference":>12}')
log(f'{"Velocity Correlation":<30} {results["velocity_corr"]:>15.3f} {"0.40":>15} {"higher=better":>12}')
log(f'{"Acceleration Correlation":<30} {results["acceleration_corr"]:>15.3f} {"0.26":>15} {"higher=better":>12}')
log(f'{"Accel Corr (magnitude)":<30} {results.get("acceleration_corr_magnitude", 0):>15.3f} {"--":>15} {"test metric":>12}')
log(f'{"Duration RMSE (ms)":<30} {results["duration_rmse_ms"]:>15.1f} {"1180.3":>15} {"lower=better":>12}')
log('-' * 75)
log(f'{"AE Reconstruction (L1)":<30} {results["ae_reconstruction_loss"]:>15.4f} {"0.041":>15} {"lower=better":>12}')
log(f'{"AE Test Loss (L1)":<30} {results["ae_test_loss"]:>15.4f} {"0.046":>15} {"lower=better":>12}')
log(f'{"FID":<30} {results["fid"]:>15.4f} {"0.270":>15} {"lower=better":>12}')
log('-' * 75)
log(f'{f"Precision (k={precision_k})":<30} {results["precision"]:>15.3f} {"0.973":>15} {"higher=better":>12}')
log(f'{f"Recall (k={precision_k})":<30} {results["recall"]:>15.3f} {"0.258":>15} {"higher=better":>12}')
log('=' * 75)
log('')
log('Done.')
'''


async def run_eval_sandbox(n_samples: int = 200, truncation: float = 1.0, savgol_window: int = 5, precision_k: int = 3):
    """Run evaluation in a Sandbox with real-time stdout streaming."""
    import modal

    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT, str(n_samples), str(truncation), str(savgol_window), str(precision_k),
        app=app,
        image=image,
        gpu='T4',
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
# Training via Sandbox (real-time stdout streaming)
# ============================================================================

TRAIN_SCRIPT = '''
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from dataclasses import asdict
import wandb

from src.config import ModelConfig, TrainingConfig, ModalConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
from src.trainer import WordGestureGANTrainer
from src.losses import WassersteinLoss
from src.utils import seed_everything, log, train_epoch_with_grad_clip

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
log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'Training for {num_epochs} epochs (resume={resume})')

# Load data
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=training_config.train_ratio, seed=config.random_seed)
train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=training_config.batch_size, num_workers=2)
log(f'Data: {len(train_ds)} train, {len(test_ds)} test')

# Create trainer
trainer = WordGestureGANTrainer(model_config, training_config, device=device)

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
    log(f'Resumed from epoch {start_epoch}')
    for _ in range(start_epoch):
        for sched in schedulers.values():
            sched.step()

if start_epoch >= num_epochs:
    log(f'Already trained to epoch {start_epoch}, nothing to do.')
    sys.exit(0)

# Initialize W&B
run_name = f'linear_{training_config.lambda_rec}_{training_config.lambda_kld}_{model_config.gen_hidden_dim}'
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
log(f'Starting training from epoch {start_epoch}...')
for epoch in range(start_epoch, num_epochs):
    trainer.current_epoch = epoch
    epoch_losses = train_epoch_with_grad_clip(trainer, train_loader, grad_clip_norm, model_config, training_config, device)

    for sched in schedulers.values():
        sched.step()
    current_lr = schedulers['G'].get_last_lr()[0]

    log(f'Epoch {epoch+1}/{num_epochs} - D1:{epoch_losses["d1_loss"]:.3f} D2:{epoch_losses["d2_loss"]:.3f} C1:{epoch_losses["cycle1_total"]:.3f} C2:{epoch_losses["cycle2_total"]:.3f} LR:{current_lr:.6f}')

    # Log to W&B
    wandb.log({
        'epoch': epoch + 1,
        'loss/d1': epoch_losses['d1_loss'],
        'loss/d2': epoch_losses['d2_loss'],
        'loss/cycle1_total': epoch_losses['cycle1_total'],
        'loss/cycle2_total': epoch_losses['cycle2_total'],
        'learning_rate': current_lr,
    }, step=epoch + 1)

    # Save checkpoint
    if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
        ckpt = trainer.get_modal_checkpoint_dict()
        ckpt['wandb_run_id'] = wandb.run.id
        torch.save(ckpt, checkpoint_dir / 'latest.pt')
        torch.save(ckpt, checkpoint_dir / f'epoch_{epoch+1}.pt')
        log(f'  Checkpoint saved at epoch {epoch+1}')

wandb.finish()
log('Training complete!')
'''


async def run_train_sandbox(num_epochs: int = 200, resume: bool = True):
    """Run training in a Sandbox with real-time stdout streaming."""
    import modal

    sb = modal.Sandbox.create(
        "python", "-c", TRAIN_SCRIPT, str(num_epochs), str(int(resume)),
        app=app,
        image=image,
        gpu='T4',
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
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--checkpoint-epoch', type=int, help='Specific checkpoint epoch to use')
    parser.add_argument('--truncation', type=float, default=None, help='Truncation for latent sampling')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples for evaluation')
    parser.add_argument('--savgol-window', type=int, default=21, help='Savitzky-Golay filter window size')
    parser.add_argument('--precision-k', type=int, default=3, help='k for precision/recall k-NN (paper uses 3)')
    args = parser.parse_args()

    async with app.run():
        if args.eval_only:
            truncation = args.truncation if args.truncation is not None else 1.0
            print(f'Running evaluation (truncation={truncation}, savgol_window={args.savgol_window}, precision_k={args.precision_k})...')
            returncode = await run_eval_sandbox(
                n_samples=args.n_samples,
                truncation=truncation,
                savgol_window=args.savgol_window,
                precision_k=args.precision_k
            )
            print(f'\nSandbox exited with code: {returncode}')
            return
        else:
            print(f'Starting training for {args.epochs} epochs...')
            returncode = await run_train_sandbox(num_epochs=args.epochs, resume=not args.no_resume)
            print(f'\nSandbox exited with code: {returncode}')


if __name__ == '__main__':
    asyncio.run(main())
