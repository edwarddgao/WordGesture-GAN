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
from typing import List, Tuple, Dict

from .config import TrainingConfig, DEFAULT_TRAINING_CONFIG


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


class AccelerationLoss(nn.Module):
    """
    Acceleration correlation loss for gesture dynamics.

    Maximizes Pearson correlation between real and fake acceleration profiles,
    which directly optimizes what the evaluation metric measures.

    Uses Savitzky-Golay filter (window=5, poly_order=3, deriv=2) matching the
    evaluation metric computation.

    Key insight: MSE and correlation measure different things!
    - MSE penalizes scale/offset differences
    - Correlation only cares about pattern similarity
    Training with MSE can actually hurt correlation by distorting natural patterns.

    L_acc = 1 - mean(correlation(savgol_acc(fake), savgol_acc(real)))
    """

    def __init__(self, window_size: int = 5, poly_order: int = 3):
        super().__init__()
        # Precompute Savitzky-Golay coefficients for second derivative
        from scipy.signal import savgol_coeffs
        coeffs = savgol_coeffs(window_size, poly_order, deriv=2)
        # Register as buffer (moves with model to GPU, but not a trainable parameter)
        # Flip for convolution (conv1d does correlation by default)
        self.register_buffer('savgol_filter', torch.tensor(coeffs, dtype=torch.float32).flip(0))
        self.padding = window_size // 2

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute acceleration correlation loss.

        Args:
            real: Real gestures of shape (batch, seq_length, 3)
            fake: Generated gestures of shape (batch, seq_length, 3)

        Returns:
            1 - mean_correlation (minimize to maximize correlation)
        """
        # Extract x,y coordinates: (batch, seq_len, 2)
        real_xy = real[:, :, :2]
        fake_xy = fake[:, :, :2]

        batch, seq_len, _ = real_xy.shape

        # Reshape for conv1d: (batch*2, 1, seq_len)
        real_flat = real_xy.permute(0, 2, 1).reshape(batch * 2, 1, seq_len)
        fake_flat = fake_xy.permute(0, 2, 1).reshape(batch * 2, 1, seq_len)

        # Apply Savitzky-Golay filter via conv1d
        weight = self.savgol_filter.to(real.device).view(1, 1, -1)
        acc_real = F.conv1d(real_flat, weight, padding=self.padding)  # (batch*2, 1, seq_len)
        acc_fake = F.conv1d(fake_flat, weight, padding=self.padding)

        # Reshape back: (batch, 2*seq_len) - concatenate x and y accelerations like eval does
        acc_real = acc_real.view(batch, 2 * seq_len)
        acc_fake = acc_fake.view(batch, 2 * seq_len)

        # Compute per-sample Pearson correlation (matching evaluation metric)
        real_centered = acc_real - acc_real.mean(dim=1, keepdim=True)
        fake_centered = acc_fake - acc_fake.mean(dim=1, keepdim=True)

        numerator = (real_centered * fake_centered).sum(dim=1)
        denominator = torch.sqrt(
            (real_centered**2).sum(dim=1) * (fake_centered**2).sum(dim=1)
        ) + 1e-8

        correlations = numerator / denominator

        # Return 1 - mean_correlation (minimize to maximize correlation)
        return 1.0 - correlations.mean()


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


class CombinedGeneratorLoss(nn.Module):
    """
    Combined generator loss as defined in the paper.

    L_gen(y) = -L_disc(y) + λ_feat * L_feat(y) + λ_rec * L_rec(y)
               + λ_lat * L_lat(y) + λ_KLD * L_KLD

    Where λ values are hyperparameters.
    """

    def __init__(self, config: TrainingConfig = DEFAULT_TRAINING_CONFIG):
        super().__init__()
        self.config = config

        self.feature_matching_loss = FeatureMatchingLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.latent_encoding_loss = LatentEncodingLoss()
        self.kl_divergence_loss = KLDivergenceLoss()

    def forward(
        self,
        real_gesture: torch.Tensor,
        fake_gesture: torch.Tensor,
        fake_scores: torch.Tensor,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
        z_original: torch.Tensor,
        z_recovered: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined generator loss.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Wasserstein generator loss (negative of discriminator's fake score)
        loss_wgan = WassersteinLoss.generator_loss(fake_scores)

        # Feature matching loss
        loss_feat = self.feature_matching_loss(real_features, fake_features)

        # Reconstruction loss
        loss_rec = self.reconstruction_loss(real_gesture, fake_gesture)

        # Latent encoding loss
        loss_lat = self.latent_encoding_loss(z_original, z_recovered)

        # KL divergence loss
        loss_kld = self.kl_divergence_loss(mu, log_var)

        # Combined loss with weights
        total_loss = (
            loss_wgan +
            self.config.lambda_feat * loss_feat +
            self.config.lambda_rec * loss_rec +
            self.config.lambda_lat * loss_lat +
            self.config.lambda_kld * loss_kld
        )

        loss_dict = {
            'loss_wgan': loss_wgan.item(),
            'loss_feat': loss_feat.item(),
            'loss_rec': loss_rec.item(),
            'loss_lat': loss_lat.item(),
            'loss_kld': loss_kld.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP (optional, not used in paper).

    This is an alternative to spectral normalization for enforcing Lipschitz constraint.

    Args:
        discriminator: Discriminator network
        real_data: Real gesture data
        fake_data: Generated gesture data
        device: Computation device

    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)

    # Random interpolation parameter
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)

    # Interpolated samples
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.requires_grad_(True)

    # Discriminator output on interpolated samples
    d_interpolated = discriminator(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty
