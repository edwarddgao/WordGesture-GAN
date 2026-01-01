"""
WordGesture-GAN model architectures.

Components:
- VariationalEncoder: Encodes user-drawn gestures into Gaussian latent space
- Generator: BiLSTM that generates gestures from prototype + latent code
- Discriminator: MLP with spectral normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Tuple, Optional

from .config import ModelConfig, DEFAULT_MODEL_CONFIG


class VariationalEncoder(nn.Module):
    """
    Variational Encoder for encoding user-drawn gestures into Gaussian latent space.

    Architecture (from paper Figure 3):
    - Linear 384x192, Leaky ReLU
    - Linear 192x96, Leaky ReLU
    - Linear 96x48, Leaky ReLU
    - Linear 48x32, Leaky ReLU
    - Two output heads: mu and log_var
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.config = config

        # Input: flattened gesture (seq_length * 3)
        input_dim = config.seq_length * config.input_dim  # 128 * 3 = 384

        # MLP layers
        layers = []
        in_dim = input_dim

        for hidden_dim in config.enc_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads for mu and log_var
        self.fc_mu = nn.Linear(config.enc_hidden_dims[-1], config.latent_dim)
        self.fc_log_var = nn.Linear(config.enc_hidden_dims[-1], config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode gesture into latent distribution.

        Args:
            x: Gesture tensor of shape (batch, seq_length, 3)

        Returns:
            Tuple of (z, mu, log_var) where z is sampled latent code
        """
        batch_size = x.size(0)
        # Flatten: (batch, seq_length, 3) -> (batch, seq_length * 3)
        x_flat = x.view(batch_size, -1)

        # Encode
        h = self.encoder(x_flat)

        # Get distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        z = mu + std * epsilon, where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Generator(nn.Module):
    """
    Generator network using Bidirectional LSTM.

    Architecture (from paper Figure 3):
    - Input: Word prototype (128x3) concatenated with repeated latent code (128x32)
    - 4 BiLSTM layers with hidden dim 32
    - Linear output layer with Tanh activation

    The generator takes a word prototype and latent code to produce a gesture.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.config = config

        # Input: prototype (3) + latent code (latent_dim)
        input_dim = config.input_dim + config.latent_dim  # 3 + 32 = 35

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.gen_hidden_dim,
            num_layers=config.gen_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output layer: BiLSTM hidden_dim * 2 (bidirectional) -> 3
        self.output_layer = nn.Linear(config.gen_hidden_dim * 2, config.input_dim)

    def forward(
        self,
        prototype: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate gesture from prototype and latent code.

        Uses residual connection: output = prototype + delta
        This helps with temporal alignment since prototype has correct timing.

        Args:
            prototype: Word prototype of shape (batch, seq_length, 3)
            z: Latent code of shape (batch, latent_dim)

        Returns:
            Generated gesture of shape (batch, seq_length, 3)
        """
        batch_size, seq_length = prototype.shape[:2]

        # Repeat latent code along sequence length: (batch, latent_dim) -> (batch, seq_length, latent_dim)
        z_repeated = z.unsqueeze(1).repeat(1, seq_length, 1)

        # Concatenate prototype and latent code: (batch, seq_length, 3 + latent_dim)
        x = torch.cat([prototype, z_repeated], dim=-1)

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(x)

        # Output layer predicts residual (delta from prototype)
        delta = torch.tanh(self.output_layer(lstm_out)) * 0.5  # Scale delta

        # Residual connection: prototype + delta, clamp to valid range
        output = torch.clamp(prototype + delta, -1.0, 1.0)

        return output


class Discriminator(nn.Module):
    """
    Discriminator network using MLP with spectral normalization.

    Architecture (from paper Figure 4):
    - Linear 384x192, Leaky ReLU
    - Linear 192x96, Leaky ReLU
    - Linear 96x48, Leaky ReLU
    - Linear 48x24, Leaky ReLU
    - Linear 24x1 (output)

    All layers use spectral normalization for WGAN stability.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.config = config

        # Input: flattened gesture (seq_length * 3)
        input_dim = config.seq_length * config.input_dim  # 128 * 3 = 384

        # Build MLP with spectral normalization
        self.layers = nn.ModuleList()
        in_dim = input_dim

        for hidden_dim in config.disc_hidden_dims:
            self.layers.append(spectral_norm(nn.Linear(in_dim, hidden_dim)))
            in_dim = hidden_dim

        # Output layer
        self.output_layer = spectral_norm(nn.Linear(in_dim, 1))

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate gesture.

        Args:
            x: Gesture tensor of shape (batch, seq_length, 3)

        Returns:
            Discriminator output of shape (batch, 1)
        """
        batch_size = x.size(0)
        # Flatten: (batch, seq_length, 3) -> (batch, seq_length * 3)
        x_flat = x.view(batch_size, -1)

        # Forward through layers
        for layer in self.layers:
            x_flat = self.activation(layer(x_flat))

        # Output (no activation - WGAN uses unbounded output)
        output = self.output_layer(x_flat)

        return output

    def get_all_features(self, x: torch.Tensor) -> list:
        """
        Get features from all hidden layers (for feature matching loss).

        Args:
            x: Gesture tensor of shape (batch, seq_length, 3)

        Returns:
            List of feature tensors from each hidden layer
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        features = []
        for layer in self.layers:
            x_flat = self.activation(layer(x_flat))
            features.append(x_flat)

        return features


class WordGestureGAN(nn.Module):
    """
    Complete WordGesture-GAN model.

    Components:
    - Generator: Produces gestures from word prototypes
    - Discriminator: Distinguishes real from generated gestures
    - VariationalEncoder: Encodes gestures into latent space

    The model uses two-cycle training similar to BicycleGAN.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.config = config

        self.generator = Generator(config)
        self.encoder = VariationalEncoder(config)

        # Two discriminators for two-cycle training
        self.discriminator_1 = Discriminator(config)
        self.discriminator_2 = Discriminator(config)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode gesture to latent space."""
        return self.encoder(x)

    def generate(
        self,
        prototype: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate gesture from prototype.

        Args:
            prototype: Word prototype
            z: Optional latent code. If None, sample from N(0,1)

        Returns:
            Generated gesture
        """
        if z is None:
            batch_size = prototype.size(0)
            z = torch.randn(batch_size, self.config.latent_dim, device=prototype.device)

        return self.generator(prototype, z)

    def discriminate(
        self,
        x: torch.Tensor,
        use_disc_1: bool = True
    ) -> torch.Tensor:
        """Discriminate gesture using specified discriminator."""
        if use_disc_1:
            return self.discriminator_1(x)
        return self.discriminator_2(x)

    def forward(
        self,
        prototype: torch.Tensor,
        real_gesture: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass for training.

        Returns dictionary with generated gesture and intermediate values.
        """
        batch_size = prototype.size(0)

        # Sample or use provided latent code
        if z is None:
            z = torch.randn(batch_size, self.config.latent_dim, device=prototype.device)

        # Generate gesture
        generated = self.generator(prototype, z)

        result = {
            'generated': generated,
            'z': z
        }

        # If real gesture provided, encode it
        if real_gesture is not None:
            z_enc, mu, log_var = self.encoder(real_gesture)
            result.update({
                'z_enc': z_enc,
                'mu': mu,
                'log_var': log_var
            })

        return result


class AutoEncoder(nn.Module):
    """
    Auto-encoder for computing FID score.

    Separate from the GAN model, used only for evaluation.
    """

    def __init__(
        self,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.config = config

        input_dim = config.seq_length * config.input_dim  # 384

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 96),
            nn.LeakyReLU(0.2),
            nn.Linear(96, hidden_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.LeakyReLU(0.2),
            nn.Linear(96, 192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, input_dim),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gesture to latent space."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to gesture."""
        x_flat = self.decoder(z)
        return x_flat.view(-1, self.config.seq_length, self.config.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct gesture."""
        z = self.encode(x)
        return self.decode(z)
