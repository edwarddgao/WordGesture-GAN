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
# Helper Functions
# ============================================================================

def _seed_everything(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _log(msg: str):
    """Print with immediate flush for Modal streaming."""
    print(msg, flush=True)


def _train_epoch_with_grad_clip(trainer, dataloader, max_norm, model_config, training_config, device):
    """Train one epoch with gradient clipping.

    This is a modified version of WordGestureGANTrainer.train_epoch that adds
    gradient clipping after backward() and before optimizer.step().
    """
    import torch
    from src.losses import WassersteinLoss

    trainer.generator.train()
    trainer.encoder.train()
    trainer.discriminator_1.train()
    trainer.discriminator_2.train()

    epoch_losses = {
        'd1_loss': 0.0,
        'd2_loss': 0.0,
        'cycle1_total': 0.0,
        'cycle2_total': 0.0,
    }
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        real_gesture = batch['gesture'].to(device)
        prototype = batch['prototype'].to(device)

        # ----- Discriminator Training -----
        for _ in range(training_config.n_critic):
            # Cycle 1: Generate with random z
            with torch.no_grad():
                z_rand = torch.randn(real_gesture.size(0), model_config.latent_dim, device=device)
                fake_gesture_1 = trainer.generator(prototype, z_rand)

            trainer.optimizer_D1.zero_grad()
            real_scores = trainer.discriminator_1(real_gesture)
            fake_scores = trainer.discriminator_1(fake_gesture_1.detach())
            d1_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)
            d1_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.discriminator_1.parameters(), max_norm)
            trainer.optimizer_D1.step()

            # Cycle 2: Generate with encoded z
            with torch.no_grad():
                z_enc, _, _ = trainer.encoder(real_gesture)
                fake_gesture_2 = trainer.generator(prototype, z_enc)

            trainer.optimizer_D2.zero_grad()
            real_scores = trainer.discriminator_2(real_gesture)
            fake_scores = trainer.discriminator_2(fake_gesture_2.detach())
            d2_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)
            d2_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.discriminator_2.parameters(), max_norm)
            trainer.optimizer_D2.step()

        # ----- Generator Training -----
        trainer.optimizer_G.zero_grad()
        trainer.optimizer_E.zero_grad()

        # Cycle 1: z -> X' -> z'
        _, g_loss_1, loss_dict_1 = trainer.train_generator_step_cycle1(prototype, real_gesture)

        # Cycle 2: X -> z -> X'
        _, g_loss_2, loss_dict_2 = trainer.train_generator_step_cycle2(prototype, real_gesture)

        # Combined generator loss
        total_g_loss = g_loss_1 + g_loss_2
        total_g_loss.backward()

        # Gradient clipping for generator and encoder
        torch.nn.utils.clip_grad_norm_(trainer.generator.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(trainer.encoder.parameters(), max_norm)

        trainer.optimizer_G.step()
        trainer.optimizer_E.step()

        # Accumulate losses
        epoch_losses['d1_loss'] += d1_loss.item()
        epoch_losses['d2_loss'] += d2_loss.item()
        epoch_losses['cycle1_total'] += loss_dict_1['cycle1_total']
        epoch_losses['cycle2_total'] += loss_dict_2['cycle2_total']
        num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


# ============================================================================
# Training
# ============================================================================

@app.function(gpu='T4', image=image, volumes={'/data': volume}, secrets=[wandb_secret], timeout=7200)
def train(num_epochs: int = 200, resume: bool = True, checkpoint_every: int = 10,
          use_lr_scheduler: bool = True, grad_clip_norm: float = 1.0):
    """Train WordGesture-GAN with checkpointing to Modal Volume.

    Args:
        num_epochs: Number of training epochs
        resume: Whether to resume from checkpoint
        checkpoint_every: Save checkpoint every N epochs
        use_lr_scheduler: Use cosine annealing LR scheduler (default: True)
        grad_clip_norm: Gradient clipping max norm (default: 1.0, set to 0 to disable)
    """
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import wandb
    from pathlib import Path
    from datetime import datetime

    from src.config import ModelConfig, TrainingConfig, ModalConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
    from src.trainer import WordGestureGANTrainer

    device = 'cuda'
    config = ModalConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig(num_epochs=num_epochs, save_every=checkpoint_every)

    _seed_everything(config.random_seed)
    _log(f'GPU: {torch.cuda.get_device_name(0)}')
    _log(f'Training for {num_epochs} epochs, checkpoints every {checkpoint_every}')
    _log(f'LR scheduler: {"cosine annealing" if use_lr_scheduler else "disabled"}')
    _log(f'Gradient clipping: {f"max_norm={grad_clip_norm}" if grad_clip_norm > 0 else "disabled"}')

    # Load data
    keyboard = QWERTYKeyboard()
    gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=training_config.train_ratio, seed=config.random_seed)
    train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=training_config.batch_size, num_workers=2)
    _log(f'Data: {len(train_ds)} train, {len(test_ds)} test')

    # Create trainer
    trainer = WordGestureGANTrainer(model_config, training_config, device=device)

    # Create LR schedulers (if enabled)
    schedulers = None
    if use_lr_scheduler:
        schedulers = {
            'G': CosineAnnealingLR(trainer.optimizer_G, T_max=num_epochs, eta_min=1e-5),
            'E': CosineAnnealingLR(trainer.optimizer_E, T_max=num_epochs, eta_min=1e-5),
            'D1': CosineAnnealingLR(trainer.optimizer_D1, T_max=num_epochs, eta_min=1e-5),
            'D2': CosineAnnealingLR(trainer.optimizer_D2, T_max=num_epochs, eta_min=1e-5),
        }
        _log(f'Created cosine annealing schedulers (T_max={num_epochs}, eta_min=1e-5)')

    # Resume from checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'latest.pt'

    start_epoch = 0
    if resume and checkpoint_path.exists():
        _log(f'Loading checkpoint from {checkpoint_path}...')
        ckpt = torch.load(checkpoint_path, map_location=device)
        trainer.load_modal_checkpoint(ckpt)
        start_epoch = ckpt['epoch'] + 1
        _log(f'Resumed from epoch {start_epoch}')
        # Fast-forward schedulers to current epoch
        if schedulers:
            for _ in range(start_epoch):
                for sched in schedulers.values():
                    sched.step()

    if start_epoch >= num_epochs:
        _log(f'Already trained to epoch {start_epoch}, nothing to do.')
        return {'status': 'already_trained', 'epoch': start_epoch}

    # Initialize wandb
    run = wandb.init(
        project=config.wandb_project,
        name=f'train-{datetime.now().strftime("%Y%m%d-%H%M")}',
        config={
            'epochs': num_epochs,
            'batch_size': training_config.batch_size,
            'lr': training_config.learning_rate,
            'latent_dim': model_config.latent_dim
        },
        resume='allow' if resume else False
    )

    # Training loop (manual to match original behavior with wandb logging)
    for epoch in range(start_epoch, num_epochs):
        trainer.current_epoch = epoch

        # Custom training epoch with gradient clipping
        if grad_clip_norm > 0:
            # Train with gradient clipping
            epoch_losses = _train_epoch_with_grad_clip(
                trainer, train_loader, grad_clip_norm, model_config, training_config, device
            )
        else:
            epoch_losses = trainer.train_epoch(train_loader)

        # Step LR schedulers
        if schedulers:
            for sched in schedulers.values():
                sched.step()
            current_lr = schedulers['G'].get_last_lr()[0]
        else:
            current_lr = training_config.learning_rate

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'd1': epoch_losses['d1_loss'],
            'd2': epoch_losses['d2_loss'],
            'g_loss': (epoch_losses['cycle1_total'] + epoch_losses['cycle2_total']) / 2,
            'rec': epoch_losses.get('cycle2_rec', 0),
            'lr': current_lr,
        })
        _log(f'Epoch {epoch+1}/{num_epochs} - D1:{epoch_losses["d1_loss"]:.3f} D2:{epoch_losses["d2_loss"]:.3f} LR:{current_lr:.6f}')

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            ckpt = trainer.get_modal_checkpoint_dict()
            torch.save(ckpt, checkpoint_dir / 'latest.pt')
            torch.save(ckpt, checkpoint_dir / f'epoch_{epoch+1}.pt')
            volume.commit()
            _log(f'Checkpoint saved at epoch {epoch+1}')

    wandb.finish()
    return {'status': 'complete', 'final_epoch': num_epochs}


# ============================================================================
# Evaluation
# ============================================================================

@app.function(gpu='T4', image=image, volumes={'/data': volume}, secrets=[wandb_secret], timeout=3600)
def evaluate(n_samples: int = 200, checkpoint_epoch: int = None, truncation: float = 1.0):
    """Evaluate trained model with all paper metrics."""
    import torch
    import numpy as np
    import wandb
    from pathlib import Path

    from src.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split
    from src.models import Generator
    from src.evaluation import evaluate_all_metrics

    device = 'cuda'
    config = ModalConfig()
    model_config = ModelConfig()
    eval_config = EvaluationConfig(n_samples=n_samples, truncation=truncation)
    training_config = TrainingConfig()

    # Load checkpoint
    if checkpoint_epoch is not None:
        checkpoint_path = Path(f'{config.checkpoint_dir}/epoch_{checkpoint_epoch}.pt')
    else:
        checkpoint_path = Path(f'{config.checkpoint_dir}/latest.pt')

    if not checkpoint_path.exists():
        return {'error': f'No checkpoint found at {checkpoint_path}'}

    _log(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load generator
    generator = Generator(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    epoch = ckpt['epoch'] + 1
    _log(f'Loaded checkpoint from epoch {epoch}')

    # Load data
    keyboard = QWERTYKeyboard()
    gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)

    # Generate samples
    n = min(n_samples, len(test_ds))
    _log(f'Generating {n} samples (truncation={truncation})...')

    real_g, fake_g = [], []
    with torch.no_grad():
        for i in range(n):
            item = test_ds[i]
            proto = item['prototype'].unsqueeze(0).to(device)
            z = torch.randn(1, model_config.latent_dim, device=device) * truncation
            fake = generator(proto, z).cpu().numpy()[0]
            real_g.append(item['gesture'].numpy())
            fake_g.append(fake)
    real_g, fake_g = np.array(real_g), np.array(fake_g)

    # Get training data for FID autoencoder
    train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])

    # Run all metrics
    _log('Computing metrics...')
    results = evaluate_all_metrics(real_g, fake_g, train_g, model_config, eval_config, device)

    # Print results table (Paper values from Tables 1-6)
    _log('=' * 75)
    _log(f'{"Metric":<30} {"Ours":>15} {"Paper":>15} {"Notes":>12}')
    _log('=' * 75)
    _log(f'{"L2 Wasserstein (x,y)":<30} {results["l2_wasserstein"]:>15.3f} {"4.409":>15} {"lower=better":>12}')
    _log(f'{"DTW Wasserstein (x,y)":<30} {results["dtw_wasserstein"]:>15.3f} {"2.146":>15} {"lower=better":>12}')
    _log(f'{"Jerk":<30} {results["jerk_fake"]:>15.4f} {"0.0058":>15} {"~real":>12}')
    _log(f'{"Velocity Correlation":<30} {results["velocity_corr"]:>15.3f} {"0.40":>15} {"higher=better":>12}')
    _log(f'{"Acceleration Correlation":<30} {results["acceleration_corr"]:>15.3f} {"0.26":>15} {"higher=better":>12}')
    _log(f'{"Accel Corr (magnitude)":<30} {results.get("acceleration_corr_magnitude", 0):>15.3f} {"--":>15} {"test metric":>12}')
    _log(f'{"Duration RMSE (ms)":<30} {results["duration_rmse_ms"]:>15.1f} {"1180.3":>15} {"lower=better":>12}')
    _log('-' * 75)
    _log(f'{"AE Reconstruction (L1)":<30} {results["ae_reconstruction_loss"]:>15.4f} {"0.041":>15} {"lower=better":>12}')
    _log(f'{"AE Test Loss (L1)":<30} {results["ae_test_loss"]:>15.4f} {"0.046":>15} {"lower=better":>12}')
    _log(f'{"FID":<30} {results["fid"]:>15.4f} {"0.270":>15} {"lower=better":>12}')
    _log('-' * 75)
    _log(f'{"Precision (k=3)":<30} {results["precision"]:>15.3f} {"0.973":>15} {"higher=better":>12}')
    _log(f'{"Recall (k=3)":<30} {results["recall"]:>15.3f} {"0.258":>15} {"higher=better":>12}')
    _log('=' * 75)

    # Log to wandb
    wandb.init(project=config.wandb_project, name=f'eval-epoch{epoch}', reinit=True)
    wandb.log({f'eval/{k}': v for k, v in results.items()})
    wandb.finish()

    return results


# ============================================================================
# Evaluation via Sandbox (real-time stdout streaming)
# ============================================================================

EVAL_SCRIPT = '''
import sys
import torch
import numpy as np
from pathlib import Path

from src.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip, create_train_test_split
from src.models import Generator
from src.evaluation import evaluate_all_metrics

def log(msg):
    print(msg, flush=True)

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
truncation = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
savgol_window = int(sys.argv[3]) if len(sys.argv) > 3 else 5
precision_k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
use_minimum_jerk_proto = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig(n_samples=n_samples, truncation=truncation, savgol_window=savgol_window, precision_recall_k=precision_k)

log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'use_minimum_jerk_proto={use_minimum_jerk_proto}')

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
log(f'  Loaded checkpoint from epoch {epoch}')

# Load data
log('[2/5] Loading data...')
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config, use_minimum_jerk_proto=use_minimum_jerk_proto)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)
log(f'  Train: {len(train_ds)}, Test: {len(test_ds)}')

# Generate samples
n = min(n_samples, len(test_ds))
log(f'[3/5] Generating {n} samples (truncation={truncation}, savgol_window={savgol_window}, precision_k={precision_k})...')

real_g, fake_g = [], []
with torch.no_grad():
    for i in range(n):
        item = test_ds[i]
        proto = item['prototype'].unsqueeze(0).to(device)
        z = torch.randn(1, model_config.latent_dim, device=device) * truncation
        fake = generator(proto, z).cpu().numpy()[0]
        real_g.append(item['gesture'].numpy())
        fake_g.append(fake)
        if (i + 1) % 100 == 0:
            log(f'  Generated {i+1}/{n}...')
real_g, fake_g = np.array(real_g), np.array(fake_g)

# Get training data for FID autoencoder
log('[4/5] Preparing training data for FID...')
train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])

# Run all metrics
log('[5/5] Computing metrics...')
results = evaluate_all_metrics(real_g, fake_g, train_g, model_config, eval_config, device)

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


async def run_eval_sandbox(n_samples: int = 200, truncation: float = 1.0, savgol_window: int = 5, precision_k: int = 3, use_minimum_jerk_proto: bool = False):
    """Run evaluation in a Sandbox with real-time stdout streaming."""
    import modal

    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT, str(n_samples), str(truncation), str(savgol_window), str(precision_k), str(int(use_minimum_jerk_proto)),
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


# ============================================================================
# Training via Sandbox (real-time stdout streaming)
# ============================================================================

TRAIN_SCRIPT = '''
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from src.config import ModelConfig, TrainingConfig, ModalConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
from src.trainer import WordGestureGANTrainer
from src.losses import WassersteinLoss

def log(msg):
    print(msg, flush=True)

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_epoch_with_grad_clip(trainer, dataloader, max_norm, model_config, training_config, device):
    """Train one epoch with gradient clipping."""
    trainer.generator.train()
    trainer.encoder.train()
    trainer.discriminator_1.train()
    trainer.discriminator_2.train()

    epoch_losses = {'d1_loss': 0.0, 'd2_loss': 0.0, 'cycle1_total': 0.0, 'cycle2_total': 0.0}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        real_gesture = batch['gesture'].to(device)
        prototype = batch['prototype'].to(device)

        # Discriminator Training
        for _ in range(training_config.n_critic):
            with torch.no_grad():
                z_rand = torch.randn(real_gesture.size(0), model_config.latent_dim, device=device)
                fake_gesture_1 = trainer.generator(prototype, z_rand)

            trainer.optimizer_D1.zero_grad()
            real_scores = trainer.discriminator_1(real_gesture)
            fake_scores = trainer.discriminator_1(fake_gesture_1.detach())
            d1_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)
            d1_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.discriminator_1.parameters(), max_norm)
            trainer.optimizer_D1.step()

            with torch.no_grad():
                z_enc, _, _ = trainer.encoder(real_gesture)
                fake_gesture_2 = trainer.generator(prototype, z_enc)

            trainer.optimizer_D2.zero_grad()
            real_scores = trainer.discriminator_2(real_gesture)
            fake_scores = trainer.discriminator_2(fake_gesture_2.detach())
            d2_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)
            d2_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.discriminator_2.parameters(), max_norm)
            trainer.optimizer_D2.step()

        # Generator Training
        trainer.optimizer_G.zero_grad()
        trainer.optimizer_E.zero_grad()
        _, g_loss_1, loss_dict_1 = trainer.train_generator_step_cycle1(prototype, real_gesture)
        _, g_loss_2, loss_dict_2 = trainer.train_generator_step_cycle2(prototype, real_gesture)
        total_g_loss = g_loss_1 + g_loss_2
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.generator.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(trainer.encoder.parameters(), max_norm)
        trainer.optimizer_G.step()
        trainer.optimizer_E.step()

        epoch_losses['d1_loss'] += d1_loss.item()
        epoch_losses['d2_loss'] += d2_loss.item()
        epoch_losses['cycle1_total'] += loss_dict_1['cycle1_total']
        epoch_losses['cycle2_total'] += loss_dict_2['cycle2_total']
        num_batches += 1

    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    return epoch_losses

# Parse args
num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
resume = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
use_minimum_jerk_proto = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
checkpoint_every = 10
grad_clip_norm = 1.0

device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig(num_epochs=num_epochs, save_every=checkpoint_every)

seed_everything(config.random_seed)
log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'Training for {num_epochs} epochs (resume={resume}, minimum_jerk_proto={use_minimum_jerk_proto})')

# Load data
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config, use_minimum_jerk_proto=use_minimum_jerk_proto)
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
if resume and checkpoint_path.exists():
    log(f'Loading checkpoint from {checkpoint_path}...')
    ckpt = torch.load(checkpoint_path, map_location=device)
    trainer.load_modal_checkpoint(ckpt)
    start_epoch = ckpt['epoch'] + 1
    log(f'Resumed from epoch {start_epoch}')
    for _ in range(start_epoch):
        for sched in schedulers.values():
            sched.step()

if start_epoch >= num_epochs:
    log(f'Already trained to epoch {start_epoch}, nothing to do.')
    sys.exit(0)

# Training loop
log(f'Starting training from epoch {start_epoch}...')
for epoch in range(start_epoch, num_epochs):
    trainer.current_epoch = epoch
    epoch_losses = train_epoch_with_grad_clip(trainer, train_loader, grad_clip_norm, model_config, training_config, device)

    for sched in schedulers.values():
        sched.step()
    current_lr = schedulers['G'].get_last_lr()[0]

    log(f'Epoch {epoch+1}/{num_epochs} - D1:{epoch_losses["d1_loss"]:.3f} D2:{epoch_losses["d2_loss"]:.3f} C1:{epoch_losses["cycle1_total"]:.3f} C2:{epoch_losses["cycle2_total"]:.3f} LR:{current_lr:.6f}')

    # Save checkpoint
    if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
        ckpt = trainer.get_modal_checkpoint_dict()
        torch.save(ckpt, checkpoint_dir / 'latest.pt')
        torch.save(ckpt, checkpoint_dir / f'epoch_{epoch+1}.pt')
        log(f'  Checkpoint saved at epoch {epoch+1}')

log('Training complete!')
'''


async def run_train_sandbox(num_epochs: int = 200, resume: bool = True, use_minimum_jerk_proto: bool = False):
    """Run training in a Sandbox with real-time stdout streaming."""
    import modal

    sb = modal.Sandbox.create(
        "python", "-c", TRAIN_SCRIPT, str(num_epochs), str(int(resume)), str(int(use_minimum_jerk_proto)),
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
# Minimum Jerk Baseline Evaluation (via Sandbox for real-time stdout)
# ============================================================================

MINIMUM_JERK_EVAL_SCRIPT = '''
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from src.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
from src.keyboard import QWERTYKeyboard
from src.data import load_dataset_from_zip, create_train_test_split
from src.minimum_jerk import MinimumJerkModel, MinimumJerkConfig
from src.evaluation import evaluate_all_metrics

def log(msg):
    print(msg, flush=True)

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
eval_config = EvaluationConfig(n_samples=n_samples)
training_config = TrainingConfig()

log(f'GPU: {torch.cuda.get_device_name(0)}')

# 1. Load data
log('[1/5] Loading data...')
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)
log(f'  Train: {len(train_ds)}, Test: {len(test_ds)}')

# 2. Train MinimumJerkModel
log('[2/5] Training Minimum Jerk model...')
mj_config = MinimumJerkConfig(seq_length=model_config.seq_length, random_seed=config.random_seed)
mj_model = MinimumJerkModel(keyboard, mj_config)

train_gestures = [train_ds[i]['gesture'].numpy() for i in range(len(train_ds))]
train_words = [train_ds[i]['word'] for i in range(len(train_ds))]
mj_model.train(train_gestures, train_words, verbose=False)
log(f'  Offset mean: ({mj_model.offset_mean[0]:.4f}, {mj_model.offset_mean[1]:.4f})')
log(f'  Offset std:  ({mj_model.offset_std[0]:.4f}, {mj_model.offset_std[1]:.4f})')

# 3. Generate fake gestures
n = min(n_samples, len(test_ds))
log(f'[3/5] Generating {n} minimum jerk gestures...')

real_g, fake_g, test_words_list = [], [], []
for i in range(n):
    item = test_ds[i]
    word = item['word']
    real_g.append(item['gesture'].numpy())
    fake_g.append(mj_model.generate(word, add_noise=True))
    test_words_list.append(word)

real_g = np.array(real_g)
fake_g = np.array(fake_g)
log('  Gestures generated')

# 4. Compute FID/Precision/Recall
log('[4/5] Computing metrics...')
train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])
base_metrics = evaluate_all_metrics(real_g, fake_g, train_g, model_config, eval_config, device)
log(f'  FID={base_metrics["fid"]:.4f}, P={base_metrics["precision"]:.3f}, R={base_metrics["recall"]:.3f}')

# 5. Compute L2/DTW Wasserstein
log('  Computing L2/DTW Wasserstein with Hungarian matching...')
real_xy = real_g[:, :, :2]
fake_xy = fake_g[:, :, :2]

real_flat = real_xy.reshape(n, -1)
fake_flat = fake_xy.reshape(n, -1)
l2_dist_matrix = cdist(real_flat, fake_flat, metric='euclidean')
row_ind, col_ind = linear_sum_assignment(l2_dist_matrix)
l2_wasserstein = l2_dist_matrix[row_ind, col_ind].mean()
log(f'  L2 Wasserstein={l2_wasserstein:.3f}')

# DTW Wasserstein
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as euc_dist

log('  Computing DTW distance matrix...')
dtw_dist = np.zeros((n, n))
for i in range(n):
    if i % 100 == 0:
        log(f'    DTW row {i}/{n}...')
    for j in range(n):
        distance, _ = fastdtw(real_xy[i], fake_xy[j], dist=euc_dist)
        dtw_dist[i, j] = distance

row_ind2, col_ind2 = linear_sum_assignment(dtw_dist)
dtw_wasserstein = dtw_dist[row_ind2, col_ind2].mean() / np.sqrt(model_config.seq_length)
log(f'  DTW Wasserstein={dtw_wasserstein:.3f}')

# Velocity/Acceleration correlation
log('  Computing velocity/acceleration correlations...')
vcorrs, acorrs = [], []
for i in range(n):
    j = col_ind[i]
    vr = np.diff(real_xy[i], axis=0).flatten()
    vf = np.diff(fake_xy[j], axis=0).flatten()
    if len(vr) == len(vf):
        corr = np.corrcoef(vr, vf)[0, 1]
        if not np.isnan(corr):
            vcorrs.append(corr)

    xr, yr = real_xy[i, :, 0], real_xy[i, :, 1]
    xf, yf = fake_xy[j, :, 0], fake_xy[j, :, 1]
    if len(xr) >= 5:
        ax_r = savgol_filter(xr, 5, 3, deriv=2)
        ay_r = savgol_filter(yr, 5, 3, deriv=2)
        ax_f = savgol_filter(xf, 5, 3, deriv=2)
        ay_f = savgol_filter(yf, 5, 3, deriv=2)
        ar = np.concatenate([ax_r, ay_r])
        af = np.concatenate([ax_f, ay_f])
        corr = np.corrcoef(ar, af)[0, 1]
        if not np.isnan(corr):
            acorrs.append(corr)

vel_corr = np.mean(vcorrs) if vcorrs else 0.0
acc_corr = np.mean(acorrs) if acorrs else 0.0

# Results table
log('')
log('[5/5] Done.')
log('=' * 75)
log('Minimum Jerk Baseline Results (Paper Tables 1-6)')
log('=' * 75)
log(f'{"Metric":<30} {"Ours":>15} {"Paper":>15} {"Notes":>12}')
log('=' * 75)
log(f'{"L2 Wasserstein (x,y)":<30} {l2_wasserstein:>15.3f} {"5.004":>15} {"lower=better":>12}')
log(f'{"DTW Wasserstein (x,y)":<30} {dtw_wasserstein:>15.3f} {"2.752":>15} {"lower=better":>12}')
log(f'{"FID":<30} {base_metrics["fid"]:>15.4f} {"0.354":>15} {"lower=better":>12}')
log(f'{"Precision":<30} {base_metrics["precision"]:>15.3f} {"0.785":>15} {"higher=better":>12}')
log(f'{"Recall":<30} {base_metrics["recall"]:>15.3f} {"0.575":>15} {"higher=better":>12}')
log('-' * 75)
log(f'{"Jerk (fake)":<30} {base_metrics["jerk_fake"]:>15.6f} {"0.0034":>15} {"~real":>12}')
log(f'{"Jerk (real)":<30} {base_metrics["jerk_real"]:>15.6f} {"0.0066":>15} {"reference":>12}')
log(f'{"Velocity Correlation":<30} {vel_corr:>15.2f} {"0.40":>15} {"higher=better":>12}')
log(f'{"Acceleration Correlation":<30} {acc_corr:>15.2f} {"0.21":>15} {"higher=better":>12}')
log('=' * 75)
'''


async def run_minimum_jerk_sandbox(n_samples: int = 200):
    """Run Minimum Jerk evaluation in a Sandbox with real-time stdout streaming."""
    import modal

    # Create sandbox with GPU and volume
    sb = modal.Sandbox.create(
        "python", "-c", MINIMUM_JERK_EVAL_SCRIPT, str(n_samples),
        app=app,
        image=image,
        gpu='T4',
        volumes={'/data': volume},
        timeout=7200,
    )

    # Stream stdout in real-time
    for line in sb.stdout:
        print(line, end='', flush=True)

    # Check for errors
    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


# ============================================================================
# SHARK2 Evaluation
# ============================================================================

@app.function(gpu='T4', image=image, volumes={'/data': volume}, secrets=[wandb_secret], timeout=7200)
def evaluate_shark2_wer(n_train_user: int = 200, n_simulated: int = 0, n_test: int = 30000,
                        checkpoint_epoch: int = None, truncation: float = 1.0):
    """Evaluate SHARK2 decoder Word Error Rate (Section 5.10, Table 7 from paper)."""
    import torch
    import numpy as np
    import random
    import wandb
    from pathlib import Path

    from src.config import ModelConfig, TrainingConfig, ModalConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip
    from src.models import Generator
    from src.shark2 import SHARK2Decoder, SHARK2Config, evaluate_decoder, load_word_frequencies

    device = 'cuda'
    config = ModalConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()

    _log(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load data
    _log('[1/6] Loading gesture dataset...')
    keyboard = QWERTYKeyboard()
    gestures_by_word, _ = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)

    lexicon = list(gestures_by_word.keys())
    _log(f'  Loaded {len(lexicon)} words')

    # Initialize decoder with language model (COCA frequencies via wordfreq)
    _log('[2/6] Loading word frequencies (COCA corpus)...')
    word_freqs = load_word_frequencies(lexicon)
    _log(f'  Loaded frequencies for {len(word_freqs)} words')

    _log('[3/6] Initializing SHARK2 decoder...')
    decoder = SHARK2Decoder(lexicon, keyboard, model_config.seq_length, word_frequencies=word_freqs)
    _log(f'  Decoder ready with {decoder.n_words} words')

    # Prepare train/test split
    _log('[4/7] Preparing train/test split...')
    all_pairs = []
    for word, gesture_list in gestures_by_word.items():
        for gesture in gesture_list:
            all_pairs.append((gesture, word))

    random.seed(config.random_seed)
    random.shuffle(all_pairs)

    n_test = min(n_test, len(all_pairs) - n_train_user - 100)
    test_pairs = all_pairs[:n_test]
    remaining = all_pairs[n_test:]
    train_user_pairs = remaining[:n_train_user]

    _log(f'  Train user gestures: {len(train_user_pairs)}')
    _log(f'  Test gestures: {len(test_pairs)}')

    # Generate simulated gestures if needed
    train_simulated_pairs = []
    if n_simulated > 0:
        _log(f'[5/7] Generating {n_simulated} simulated gestures...')

        if checkpoint_epoch is not None:
            checkpoint_path = Path(f'{config.checkpoint_dir}/epoch_{checkpoint_epoch}.pt')
        else:
            checkpoint_path = Path(f'{config.checkpoint_dir}/latest.pt')

        if not checkpoint_path.exists():
            return {'error': f'No checkpoint found at {checkpoint_path}'}

        generator = Generator(model_config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        generator.eval()
        _log(f'  Loaded generator from epoch {ckpt["epoch"] + 1}')

        batch_size = 64
        with torch.no_grad():
            for batch_start in range(0, n_simulated, batch_size):
                batch_end = min(batch_start + batch_size, n_simulated)
                batch_words = [random.choice(lexicon) for _ in range(batch_end - batch_start)]

                protos = torch.stack([torch.FloatTensor(
                    keyboard.get_word_prototype(w, model_config.seq_length)
                ) for w in batch_words]).to(device)
                z = torch.randn(len(batch_words), model_config.latent_dim, device=device) * truncation
                fakes = generator(protos, z).cpu().numpy()

                for j, word in enumerate(batch_words):
                    train_simulated_pairs.append((fakes[j], word))

        _log(f'  Generated {len(train_simulated_pairs)} simulated gestures')
    else:
        _log('[5/7] Skipping simulated gesture generation')

    # Combine training data
    train_pairs = train_user_pairs + train_simulated_pairs
    _log(f'  Total training gestures: {len(train_pairs)}')

    # Optimize SHARK2 parameters
    _log('[6/7] Optimizing SHARK2 parameters...')
    train_gestures = [g for g, _ in train_pairs]
    train_labels = [w for _, w in train_pairs]
    best_params = decoder.optimize_parameters(train_gestures, train_labels, max_samples=200, verbose=True)
    sigma_loc, sigma_shape, sigma_lm = best_params
    _log(f'  Best: sigma_loc={sigma_loc}, sigma_shape={sigma_shape}, sigma_lm={sigma_lm}')

    # Evaluate
    _log(f'[7/7] Evaluating on {len(test_pairs)} test gestures...')
    test_gestures = [g for g, _ in test_pairs]
    test_labels = [w for _, w in test_pairs]
    eval_results = evaluate_decoder(decoder, test_gestures, test_labels, batch_size=500, verbose=True)
    test_wer = eval_results['wer']

    # Print results
    _log('=' * 65)
    _log(f'SHARK2 Word Error Rate: {test_wer * 100:.1f}%')
    _log(f'  Training: {n_train_user} user + {n_simulated} simulated gestures')
    _log(f'  Test: {len(test_pairs)} gestures')
    _log('')
    _log('Paper Table 7 Reference:')
    _log(f'  200 User-drawn: 32.8% WER')
    _log(f'  200 User + 10000 Simulated: 28.6% WER')
    _log(f'  10000 Simulated only: 28.6% WER')
    _log(f'  10000 User-drawn: 27.8% WER')
    _log('=' * 65)

    # Log to wandb
    wandb.init(project=config.wandb_project, name=f'shark2-{n_train_user}u-{n_simulated}s', reinit=True)
    wandb.log({
        'shark2/wer': test_wer,
        'shark2/n_train_user': n_train_user,
        'shark2/n_simulated': n_simulated,
        'shark2/n_test': len(test_pairs),
        'shark2/sigma_loc': sigma_loc,
        'shark2/sigma_shape': sigma_shape,
        'shark2/sigma_lm': sigma_lm,
    })
    wandb.finish()

    return {
        'wer': test_wer,
        'n_train_user': n_train_user,
        'n_simulated': n_simulated,
        'n_test': len(test_pairs),
        'sigma_loc': sigma_loc,
        'sigma_shape': sigma_shape,
        'sigma_lm': sigma_lm,
    }


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
    parser.add_argument('--shark2', action='store_true', help='Run SHARK2 WER evaluation')
    parser.add_argument('--shark2-train-user', type=int, default=200, help='User gestures for SHARK2 training')
    parser.add_argument('--shark2-simulated', type=int, default=0, help='Simulated gestures for SHARK2')
    parser.add_argument('--minimum-jerk', action='store_true', help='Evaluate Minimum Jerk baseline')
    parser.add_argument('--truncation', type=float, default=None, help='Truncation for latent sampling')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples for evaluation')
    parser.add_argument('--savgol-window', type=int, default=21, help='Savitzky-Golay filter window size')
    parser.add_argument('--precision-k', type=int, default=3, help='k for precision/recall k-NN (paper uses 3)')
    parser.add_argument('--minimum-jerk-proto', action='store_true', help='Use minimum jerk prototypes (paper Section 6.3)')
    # Training hyperparameters (Experiment 1)
    parser.add_argument('--no-lr-scheduler', action='store_true', help='Disable LR scheduler')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm (0 to disable)')
    args = parser.parse_args()

    async with app.run():
        if args.minimum_jerk:
            print('Running Minimum Jerk baseline evaluation (streaming stdout)...')
            returncode = await run_minimum_jerk_sandbox()
            print(f'\nSandbox exited with code: {returncode}')
            return
        elif args.shark2:
            print('Running SHARK2 WER evaluation...')
            result = await evaluate_shark2_wer.remote.aio(
                n_train_user=args.shark2_train_user,
                n_simulated=args.shark2_simulated,
                checkpoint_epoch=args.checkpoint_epoch
            )
        elif args.eval_only:
            truncation = args.truncation if args.truncation is not None else 1.0
            print(f'Running evaluation (truncation={truncation}, savgol_window={args.savgol_window}, precision_k={args.precision_k}, minimum_jerk_proto={args.minimum_jerk_proto})...')
            returncode = await run_eval_sandbox(
                n_samples=args.n_samples,
                truncation=truncation,
                savgol_window=args.savgol_window,
                precision_k=args.precision_k,
                use_minimum_jerk_proto=args.minimum_jerk_proto
            )
            print(f'\nSandbox exited with code: {returncode}')
            return
        else:
            print(f'Starting training for {args.epochs} epochs (streaming stdout via sandbox, minimum_jerk_proto={args.minimum_jerk_proto})...')
            returncode = await run_train_sandbox(num_epochs=args.epochs, resume=not args.no_resume, use_minimum_jerk_proto=args.minimum_jerk_proto)
            print(f'\nSandbox exited with code: {returncode}')
            return
        print(f'Result: {result}')


if __name__ == '__main__':
    asyncio.run(main())
