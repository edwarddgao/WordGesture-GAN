"""
Shared utilities for training and evaluation.

These functions are used by both the main modal_train.py code and
the embedded sandbox scripts.
"""

import torch


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


def train_epoch_with_grad_clip(trainer, dataloader, max_norm, model_config, training_config, device):
    """Train one epoch with gradient clipping.

    This is a modified version of WordGestureGANTrainer.train_epoch that adds
    gradient clipping after backward() and before optimizer.step().

    Args:
        trainer: WordGestureGANTrainer instance
        dataloader: Training data loader
        max_norm: Maximum gradient norm for clipping
        model_config: ModelConfig instance
        training_config: TrainingConfig instance
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary of average losses for the epoch
    """
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
