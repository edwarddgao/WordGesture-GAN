"""
Shared utilities for training and evaluation.

These functions are used by both the main modal_train.py code and
the embedded sandbox scripts.
"""

import torch
from torch.amp import autocast, GradScaler


def seed_everything(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def log(msg: str):
    """Print with immediate flush for Modal streaming."""
    print(msg, flush=True)


def train_epoch_with_grad_clip(trainer, dataloader, max_norm, model_config, training_config, device, scaler=None):
    """Train one epoch with gradient clipping and optional mixed precision.

    This is a modified version of WordGestureGANTrainer.train_epoch that adds
    gradient clipping after backward() and before optimizer.step().

    Args:
        trainer: WordGestureGANTrainer instance
        dataloader: Training data loader
        max_norm: Maximum gradient norm for clipping
        model_config: ModelConfig instance
        training_config: TrainingConfig instance
        device: Device to use ('cuda' or 'cpu')
        scaler: Optional GradScaler for mixed precision training

    Returns:
        Dictionary of average losses for the epoch
    """
    from src.gan.losses import WassersteinLoss

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
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        # Non-blocking transfers for better CPU/GPU overlap
        real_gesture = batch['gesture'].to(device, non_blocking=True)
        prototype = batch['prototype'].to(device, non_blocking=True)

        # ----- Discriminator Training -----
        for _ in range(training_config.n_critic):
            # Cycle 1: Generate with random z
            with torch.no_grad():
                z_rand = torch.randn(real_gesture.size(0), model_config.latent_dim, device=device)
                with autocast(device_type='cuda', enabled=use_amp):
                    fake_gesture_1 = trainer.generator(prototype, z_rand)

            trainer.optimizer_D1.zero_grad()
            with autocast(device_type='cuda', enabled=use_amp):
                real_scores = trainer.discriminator_1(real_gesture)
                fake_scores = trainer.discriminator_1(fake_gesture_1.detach())
                d1_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)

            if use_amp:
                scaler.scale(d1_loss).backward()
                scaler.step(trainer.optimizer_D1)
                scaler.update()
            else:
                d1_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.discriminator_1.parameters(), max_norm)
                trainer.optimizer_D1.step()

            # Cycle 2: Generate with encoded z
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=use_amp):
                    z_enc, _, _ = trainer.encoder(real_gesture)
                    fake_gesture_2 = trainer.generator(prototype, z_enc)

            trainer.optimizer_D2.zero_grad()
            with autocast(device_type='cuda', enabled=use_amp):
                real_scores = trainer.discriminator_2(real_gesture)
                fake_scores = trainer.discriminator_2(fake_gesture_2.detach())
                d2_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)

            if use_amp:
                scaler.scale(d2_loss).backward()
                scaler.step(trainer.optimizer_D2)
                scaler.update()
            else:
                d2_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.discriminator_2.parameters(), max_norm)
                trainer.optimizer_D2.step()

        # ----- Generator Training -----
        trainer.optimizer_G.zero_grad()
        trainer.optimizer_E.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            # Cycle 1: z -> X' -> z'
            _, g_loss_1, loss_dict_1 = trainer.train_generator_step_cycle1(prototype, real_gesture)

            # Cycle 2: X -> z -> X'
            _, g_loss_2, loss_dict_2 = trainer.train_generator_step_cycle2(prototype, real_gesture)

            # Combined generator loss
            total_g_loss = g_loss_1 + g_loss_2

        if use_amp:
            scaler.scale(total_g_loss).backward()
            scaler.step(trainer.optimizer_G)
            scaler.step(trainer.optimizer_E)
            scaler.update()
        else:
            total_g_loss.backward()
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
