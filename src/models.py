"""
WordGesture-GAN model architectures.

Components:
- VariationalEncoder: Encodes user-drawn gestures into Gaussian latent space
- Generator: BiLSTM that generates gestures from prototype + latent code
- Discriminator: MLP with spectral normalization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple

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

        Direct output architecture (matching paper Section 3.1.2, Figure 3).
        This allows the model to learn full gesture dynamics including
        realistic acceleration patterns.

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

        # Direct output (paper architecture - no residual connection)
        # This allows learning full acceleration dynamics
        output = torch.tanh(self.output_layer(lstm_out))

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


class AutoEncoder(nn.Module):
    """
    Auto-encoder for computing FID score.

    Architecture from paper Section 4.3:
    "Our encoder consists of 4 layers of 192-96-48-32 neurons,
    followed by a mean pooling and a linear layer."

    Processes each timestep through MLP, then applies mean pooling
    across the sequence dimension.
    """

    def __init__(
        self,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        hidden_dim: int = 32  # Paper uses 32-dim embedding
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # Per-timestep encoder: (x,y,t) -> 192 -> 96 -> 48 -> 32
        self.timestep_encoder = nn.Sequential(
            nn.Linear(config.input_dim, 192),  # 3 -> 192
            nn.LeakyReLU(0.2),
            nn.Linear(192, 96),
            nn.LeakyReLU(0.2),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.2),
            nn.Linear(48, hidden_dim),  # 48 -> 32
        )

        # After mean pooling, optional linear layer (paper mentions "a linear layer")
        self.post_pool = nn.Linear(hidden_dim, hidden_dim)

        # Decoder: reverse the process
        self.pre_expand = nn.Linear(hidden_dim, hidden_dim)

        self.timestep_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.LeakyReLU(0.2),
            nn.Linear(48, 96),
            nn.LeakyReLU(0.2),
            nn.Linear(96, 192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, config.input_dim),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode gesture to latent space.

        Args:
            x: Gesture tensor of shape (batch, seq_length, 3)

        Returns:
            Latent features of shape (batch, hidden_dim)
        """
        batch_size, seq_length, _ = x.shape

        # Process each timestep: (batch, seq, 3) -> (batch, seq, hidden_dim)
        timestep_features = self.timestep_encoder(x)

        # Mean pooling across sequence: (batch, seq, hidden_dim) -> (batch, hidden_dim)
        pooled = timestep_features.mean(dim=1)

        # Final linear layer
        return self.post_pool(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to gesture.

        Args:
            z: Latent tensor of shape (batch, hidden_dim)

        Returns:
            Reconstructed gesture of shape (batch, seq_length, 3)
        """
        batch_size = z.size(0)

        # Expand to sequence
        z_expanded = self.pre_expand(z)

        # Broadcast to all timesteps: (batch, hidden_dim) -> (batch, seq, hidden_dim)
        z_seq = z_expanded.unsqueeze(1).expand(-1, self.config.seq_length, -1)

        # Decode each timestep
        return self.timestep_decoder(z_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct gesture."""
        z = self.encode(x)
        return self.decode(z)
