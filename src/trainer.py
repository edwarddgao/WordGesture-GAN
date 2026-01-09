"""
Training loop for WordGesture-GAN with two-cycle training.

Based on the BicycleGAN training procedure:
- Cycle 1: z -> X' -> z' (latent code recovery)
- Cycle 2: X -> z -> X' (gesture reconstruction)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable

from .models import Generator, Discriminator, VariationalEncoder
from .losses import (
    WassersteinLoss,
    FeatureMatchingLoss,
    ReconstructionLoss,
    LatentEncodingLoss,
    KLDivergenceLoss
)
from .config import ModelConfig, TrainingConfig, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG


class WordGestureGANTrainer:
    """
    Trainer for WordGesture-GAN with two-cycle training.

    Two cycles following BicycleGAN:
    - Cycle 1 (z -> X' -> z'): Sample z, generate X', recover z', compare z and z'
    - Cycle 2 (X -> z -> X'): Encode X to z, generate X', compare X and X'
    """

    def __init__(
        self,
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(device)

        # Initialize model components
        self.generator = Generator(model_config).to(self.device)
        self.encoder = VariationalEncoder(model_config).to(self.device)

        # Two discriminators for two cycles
        self.discriminator_1 = Discriminator(model_config).to(self.device)
        self.discriminator_2 = Discriminator(model_config).to(self.device)

        # Loss functions
        self.feature_matching_loss = FeatureMatchingLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.latent_encoding_loss = LatentEncodingLoss()
        self.kl_divergence_loss = KLDivergenceLoss()

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=training_config.learning_rate,
            betas=(0.5, 0.999)
        )
        self.optimizer_E = optim.Adam(
            self.encoder.parameters(),
            lr=training_config.learning_rate,
            betas=(0.5, 0.999)
        )
        self.optimizer_D1 = optim.Adam(
            self.discriminator_1.parameters(),
            lr=training_config.learning_rate,
            betas=(0.5, 0.999)
        )
        self.optimizer_D2 = optim.Adam(
            self.discriminator_2.parameters(),
            lr=training_config.learning_rate,
            betas=(0.5, 0.999)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = []

    def train_discriminator_step(
        self,
        real_gesture: torch.Tensor,
        fake_gesture: torch.Tensor,
        discriminator: Discriminator,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Single discriminator training step.

        Args:
            real_gesture: Real user-drawn gesture
            fake_gesture: Generated gesture (detached)
            discriminator: Discriminator to train
            optimizer: Discriminator optimizer

        Returns:
            Discriminator loss value
        """
        optimizer.zero_grad()

        # Discriminate real and fake
        real_scores = discriminator(real_gesture)
        fake_scores = discriminator(fake_gesture.detach())

        # Wasserstein loss
        d_loss = WassersteinLoss.discriminator_loss(real_scores, fake_scores)

        d_loss.backward()
        optimizer.step()

        return d_loss.item()

    def train_generator_step_cycle1(
        self,
        prototype: torch.Tensor,
        real_gesture: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generator training step for Cycle 1: z -> X' -> z'

        Sample random z, generate gesture X', recover z' from X',
        compare z and z'.

        Args:
            prototype: Word prototype
            real_gesture: Real gesture (for discriminator features)

        Returns:
            Tuple of (generated gesture, loss dict)
        """
        batch_size = prototype.size(0)

        # Sample random latent code
        z = torch.randn(batch_size, self.model_config.latent_dim, device=self.device)

        # Generate gesture
        fake_gesture = self.generator(prototype, z)

        # Discriminator scores and features
        fake_scores = self.discriminator_1(fake_gesture)
        fake_features = self.discriminator_1.get_all_features(fake_gesture)
        real_features = self.discriminator_1.get_all_features(real_gesture)

        # Recover latent code from generated gesture (freeze encoder for this)
        with torch.no_grad():
            self.encoder.eval()
            z_recovered, _, _ = self.encoder(fake_gesture)
            self.encoder.train()

        # Compute losses
        loss_wgan = WassersteinLoss.generator_loss(fake_scores)
        loss_feat = self.feature_matching_loss(real_features, fake_features)
        loss_lat = self.latent_encoding_loss(z, z_recovered)

        # Combined loss for cycle 1
        total_loss = (
            loss_wgan +
            self.training_config.lambda_feat * loss_feat +
            self.training_config.lambda_lat * loss_lat
        )

        loss_dict = {
            'cycle1_wgan': loss_wgan.item(),
            'cycle1_feat': loss_feat.item(),
            'cycle1_lat': loss_lat.item(),
            'cycle1_total': total_loss.item()
        }

        return fake_gesture, total_loss, loss_dict

    def train_generator_step_cycle2(
        self,
        prototype: torch.Tensor,
        real_gesture: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generator training step for Cycle 2: X -> z -> X'

        Encode real gesture X to latent code z, generate X' from z,
        compare X and X'.

        Args:
            prototype: Word prototype
            real_gesture: Real gesture to reconstruct

        Returns:
            Tuple of (generated gesture, loss dict)
        """
        # Encode real gesture to latent space
        z_enc, mu, log_var = self.encoder(real_gesture)

        # Generate gesture from encoded latent code
        fake_gesture = self.generator(prototype, z_enc)

        # Discriminator scores and features
        fake_scores = self.discriminator_2(fake_gesture)
        fake_features = self.discriminator_2.get_all_features(fake_gesture)
        real_features = self.discriminator_2.get_all_features(real_gesture)

        # Compute losses
        loss_wgan = WassersteinLoss.generator_loss(fake_scores)
        loss_feat = self.feature_matching_loss(real_features, fake_features)
        loss_rec = self.reconstruction_loss(real_gesture, fake_gesture)
        loss_kld = self.kl_divergence_loss(mu, log_var)

        # Combined loss for cycle 2
        total_loss = (
            loss_wgan +
            self.training_config.lambda_feat * loss_feat +
            self.training_config.lambda_rec * loss_rec +
            self.training_config.lambda_kld * loss_kld
        )

        loss_dict = {
            'cycle2_wgan': loss_wgan.item(),
            'cycle2_feat': loss_feat.item(),
            'cycle2_rec': loss_rec.item(),
            'cycle2_kld': loss_kld.item(),
            'cycle2_total': total_loss.item()
        }

        return fake_gesture, total_loss, loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Following WGAN training procedure: update discriminator n_critic times
        per generator update.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of average losses for the epoch
        """
        self.generator.train()
        self.encoder.train()
        self.discriminator_1.train()
        self.discriminator_2.train()

        epoch_losses = {
            'd1_loss': 0.0,
            'd2_loss': 0.0,
            'cycle1_total': 0.0,
            'cycle2_total': 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            real_gesture = batch['gesture'].to(self.device)
            prototype = batch['prototype'].to(self.device)

            # ----- Discriminator Training -----
            for _ in range(self.training_config.n_critic):
                # Cycle 1: Generate with random z
                with torch.no_grad():
                    z_rand = torch.randn(
                        real_gesture.size(0),
                        self.model_config.latent_dim,
                        device=self.device
                    )
                    fake_gesture_1 = self.generator(prototype, z_rand)

                d1_loss = self.train_discriminator_step(
                    real_gesture, fake_gesture_1,
                    self.discriminator_1, self.optimizer_D1
                )

                # Cycle 2: Generate with encoded z
                with torch.no_grad():
                    z_enc, _, _ = self.encoder(real_gesture)
                    fake_gesture_2 = self.generator(prototype, z_enc)

                d2_loss = self.train_discriminator_step(
                    real_gesture, fake_gesture_2,
                    self.discriminator_2, self.optimizer_D2
                )

            # ----- Generator Training -----
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()

            # Cycle 1: z -> X' -> z'
            _, g_loss_1, loss_dict_1 = self.train_generator_step_cycle1(
                prototype, real_gesture
            )

            # Cycle 2: X -> z -> X'
            _, g_loss_2, loss_dict_2 = self.train_generator_step_cycle2(
                prototype, real_gesture
            )

            # Combined generator loss
            total_g_loss = g_loss_1 + g_loss_2
            total_g_loss.backward()

            self.optimizer_G.step()
            self.optimizer_E.step()

            # Accumulate losses
            epoch_losses['d1_loss'] += d1_loss
            epoch_losses['d2_loss'] += d2_loss
            epoch_losses['cycle1_total'] += loss_dict_1['cycle1_total']
            epoch_losses['cycle2_total'] += loss_dict_2['cycle2_total']
            num_batches += 1

            # Logging
            if batch_idx % self.training_config.log_every == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"D1={d1_loss:.4f}, D2={d2_loss:.4f}, "
                      f"G1={loss_dict_1['cycle1_total']:.4f}, "
                      f"G2={loss_dict_2['cycle2_total']:.4f}")

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: Optional[int] = None,
        checkpoint_dir: str = 'checkpoints',
        resume_from: Optional[str] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs (default from config)
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
            callbacks: Optional dict of callbacks:
                - 'on_epoch_end': fn(epoch, losses) called after each epoch
                - 'on_checkpoint': fn() called after checkpoint save
        """
        if num_epochs is None:
            num_epochs = self.training_config.num_epochs

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"Starting training for {num_epochs} epochs")
        print(f"Training on device: {self.device}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_losses = self.train_epoch(train_loader)

            # Log epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.4f}")

            self.training_history.append({
                'epoch': epoch + 1,
                **epoch_losses
            })

            # Callback: on_epoch_end
            if callbacks and 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](epoch, epoch_losses)

            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                )
                # Callback: on_checkpoint
                if callbacks and 'on_checkpoint' in callbacks:
                    callbacks['on_checkpoint']()

        # Save final checkpoint
        self.save_checkpoint(os.path.join(checkpoint_dir, 'checkpoint_final.pt'))
        if callbacks and 'on_checkpoint' in callbacks:
            callbacks['on_checkpoint']()
        print("Training complete!")

    def get_modal_checkpoint_dict(self) -> dict:
        """
        Get checkpoint dict in modal_train.py-compatible format.

        This format uses shorter keys and is compatible with the
        checkpoints saved by modal_train.py.

        Returns:
            Checkpoint dictionary
        """
        return {
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator_1': self.discriminator_1.state_dict(),
            'discriminator_2': self.discriminator_2.state_dict(),
            'encoder': self.encoder.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D1': self.optimizer_D1.state_dict(),
            'optimizer_D2': self.optimizer_D2.state_dict(),
            'optimizer_E': self.optimizer_E.state_dict(),
        }

    def load_modal_checkpoint(self, checkpoint: dict):
        """
        Load checkpoint in modal_train.py format.

        Args:
            checkpoint: Checkpoint dict with modal_train.py format keys
        """
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator_1.load_state_dict(checkpoint['discriminator_1'])
        self.discriminator_2.load_state_dict(checkpoint['discriminator_2'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
        self.optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
        self.optimizer_E.load_state_dict(checkpoint['optimizer_E'])
        print(f"Loaded modal checkpoint from epoch {checkpoint['epoch'] + 1}")

    def save_checkpoint(self, path: str):
        """Save training checkpoint.

        Uses the same format as get_modal_checkpoint_dict() for consistency.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator': self.generator.state_dict(),
            'encoder': self.encoder.state_dict(),
            'discriminator_1': self.discriminator_1.state_dict(),
            'discriminator_2': self.discriminator_2.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_E': self.optimizer_E.state_dict(),
            'optimizer_D1': self.optimizer_D1.state_dict(),
            'optimizer_D2': self.optimizer_D2.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Supports both old format (with _state_dict suffix) and new format for backwards compatibility.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)

        # Support both old and new key formats
        def get_state(key):
            return checkpoint.get(key) or checkpoint.get(f'{key}_state_dict')

        self.generator.load_state_dict(get_state('generator'))
        self.encoder.load_state_dict(get_state('encoder'))
        self.discriminator_1.load_state_dict(get_state('discriminator_1'))
        self.discriminator_2.load_state_dict(get_state('discriminator_2'))
        self.optimizer_G.load_state_dict(get_state('optimizer_G'))
        self.optimizer_E.load_state_dict(get_state('optimizer_E'))
        self.optimizer_D1.load_state_dict(get_state('optimizer_D1'))
        self.optimizer_D2.load_state_dict(get_state('optimizer_D2'))
        self.training_history = checkpoint.get('training_history', [])

        print(f"Checkpoint loaded from {path}, resuming from epoch {self.current_epoch + 1}")

    def generate_gestures(
        self,
        prototypes: torch.Tensor,
        num_samples: int = 1,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate gestures from prototypes.

        Args:
            prototypes: Word prototypes of shape (batch, seq_length, 3)
            num_samples: Number of samples to generate per prototype
            z: Optional latent codes

        Returns:
            Generated gestures
        """
        self.generator.eval()

        with torch.no_grad():
            batch_size = prototypes.size(0)

            if z is None:
                z = torch.randn(
                    batch_size * num_samples,
                    self.model_config.latent_dim,
                    device=self.device
                )

            # Repeat prototypes for multiple samples
            if num_samples > 1:
                prototypes = prototypes.repeat(num_samples, 1, 1)

            generated = self.generator(prototypes.to(self.device), z)

        return generated
