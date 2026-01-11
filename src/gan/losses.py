"""
Loss functions for WordGesture-GAN training.

Implements:
- Wasserstein GAN loss (discriminator and generator)
- Feature matching loss
- Reconstruction loss (L1)
- Latent encoding loss
- KL divergence loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class WassersteinLoss:
    """
    Wasserstein GAN loss functions.

    Discriminator loss: E[D(G(z,y))] - E[D(x)]
    Generator loss: -E[D(G(z,y))]
    """

    @staticmethod
    def discriminator_loss(
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss.

        Minimizing this makes D output high for real, low for fake.

        Args:
            real_scores: D(x) for real gestures
            fake_scores: D(G(z,y)) for generated gestures

        Returns:
            Discriminator loss value
        """
        return fake_scores.mean() - real_scores.mean()

    @staticmethod
    def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.

        Minimizing this makes G produce gestures that D rates highly.

        Args:
            fake_scores: D(G(z,y)) for generated gestures

        Returns:
            Generator loss value
        """
        return -fake_scores.mean()


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss from Pix2PixHD.

    Computes L1 distance between features of real and fake gestures
    across all hidden layers of the discriminator.

    L_feat = sum_{i=1}^{T} (1/N_i) * ||D^{(i)}(G(z,y)) - D^{(i)}(x)||_1
    """

    def forward(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: List of features from D for real gestures
            fake_features: List of features from D for fake gestures

        Returns:
            Feature matching loss value
        """
        loss = 0.0

        for real_feat, fake_feat in zip(real_features, fake_features):
            # Normalize by number of elements in layer
            n_elements = real_feat.numel() / real_feat.size(0)  # Per sample
            loss += F.l1_loss(fake_feat, real_feat.detach()) / n_elements

        return loss / len(real_features)


class ReconstructionLoss(nn.Module):
    """
    L1 reconstruction loss between generated and real gestures.

    L_rec = E[|x - G(z,y)|]

    Computed over (x, y, t) coordinates.
    """

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            real: Real gestures of shape (batch, seq_length, 3)
            fake: Generated gestures of shape (batch, seq_length, 3)

        Returns:
            Reconstruction loss value
        """
        return F.l1_loss(fake, real)


class LatentEncodingLoss(nn.Module):
    """
    Latent encoding loss for diversity (from BicycleGAN).

    Ensures that the latent code can be recovered from the generated gesture.

    L_lat = E[||E(G(z,y)) - z||_1]
    """

    def forward(
        self,
        z_original: torch.Tensor,
        z_recovered: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute latent encoding loss.

        Args:
            z_original: Original sampled latent code
            z_recovered: Latent code recovered from generated gesture

        Returns:
            Latent encoding loss value
        """
        return F.l1_loss(z_recovered, z_original)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for VAE.

    Keeps the encoder output distribution close to N(0, 1).

    L_KLD = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            mu: Mean of encoded distribution
            log_var: Log variance of encoded distribution

        Returns:
            KL divergence loss value
        """
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kld.mean()
