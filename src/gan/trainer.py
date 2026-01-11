"""
Training loop for WordGesture-GAN with two-cycle training.

Based on the BicycleGAN training procedure:
- Cycle 1: z -> X' -> z' (latent code recovery)
- Cycle 2: X -> z -> X' (gesture reconstruction)
"""

import torch
import torch.optim as optim
from typing import Dict, Tuple

from .models import Generator, Discriminator, TemporalDiscriminator, VariationalEncoder
from .losses import (
    WassersteinLoss,
    FeatureMatchingLoss,
    ReconstructionLoss,
    LatentEncodingLoss,
    KLDivergenceLoss
)
from src.shared.config import ModelConfig, TrainingConfig, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG


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
        # Use TemporalDiscriminator (Conv1D) for better temporal pattern detection
        DiscClass = TemporalDiscriminator if model_config.use_temporal_disc else Discriminator
        self.discriminator_1 = DiscClass(model_config).to(self.device)
        self.discriminator_2 = DiscClass(model_config).to(self.device)

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

    def train_generator_step_cycle1(
        self,
        prototype: torch.Tensor,
        real_gesture: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Generator training step for Cycle 1: z -> X' -> z'

        Sample random z, generate gesture X', recover z' from X',
        compare z and z'.

        Args:
            prototype: Word prototype
            real_gesture: Real gesture (for discriminator features)

        Returns:
            Tuple of (generated gesture, total loss, loss dict)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Generator training step for Cycle 2: X -> z -> X'

        Encode real gesture X to latent code z, generate X' from z,
        compare X and X'.

        Args:
            prototype: Word prototype
            real_gesture: Real gesture to reconstruct

        Returns:
            Tuple of (generated gesture, total loss, loss dict)
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

    def get_modal_checkpoint_dict(self) -> dict:
        """
        Get checkpoint dict in modal_train.py-compatible format.

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
