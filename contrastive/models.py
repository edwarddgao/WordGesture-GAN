"""Two-tower contrastive encoder models."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GestureEncoder(nn.Module):
    """1D CNN encoder for gesture trajectories.

    Input: (B, seq_len, in_channels) - default 31 with key proximity + velocity
    Output: (B, embedding_dim) - L2 normalized embeddings
    """

    def __init__(self, in_channels: int = 31, embedding_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # (B, in_channels, 128) -> (B, 32, 64)
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # (B, 32, 64) -> (B, 64, 32)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # (B, 64, 32) -> (B, 128, 16)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # (B, 128, 16) -> (B, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 3) -> (B, 3, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)  # (B, 128)
        x = self.fc(x)  # (B, embedding_dim)
        return F.normalize(x, dim=-1)


class PrototypeEncoder(nn.Module):
    """1D CNN encoder for word prototypes.

    Input: (B, seq_len, 2) - x, y channels only (no timing)
    Output: (B, embedding_dim) - L2 normalized embeddings
    """

    def __init__(self, in_channels: int = 2, embedding_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 2, 128) -> (B, 32, 64)
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # (B, 32, 64) -> (B, 64, 32)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # (B, 64, 32) -> (B, 128, 16)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # (B, 128, 16) -> (B, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 2) -> (B, 2, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)  # (B, 128)
        x = self.fc(x)  # (B, embedding_dim)
        return F.normalize(x, dim=-1)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning (discarded at inference)."""

    def __init__(self, embedding_dim: int = 64, projection_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    """Complete two-tower contrastive model.

    Gesture tower: encodes (B, seq_len, gesture_in_channels) -> (B, embedding_dim)
    Prototype tower: encodes (B, seq_len, 2) -> (B, embedding_dim)

    Training uses projection heads; inference uses raw embeddings.
    """

    def __init__(
        self,
        gesture_in_channels: int = 31,
        embedding_dim: int = 64,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.gesture_in_channels = gesture_in_channels
        self.gesture_encoder = GestureEncoder(in_channels=gesture_in_channels, embedding_dim=embedding_dim)
        self.prototype_encoder = PrototypeEncoder(in_channels=2, embedding_dim=embedding_dim)
        self.gesture_proj = ProjectionHead(embedding_dim, projection_dim)
        self.prototype_proj = ProjectionHead(embedding_dim, projection_dim)

        # Learnable temperature (use float32 for MPS compatibility)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=1.0)

    def encode_gesture(self, gesture: torch.Tensor) -> torch.Tensor:
        """Get gesture embedding (for inference, includes projection)."""
        enc = self.gesture_encoder(gesture)
        return F.normalize(self.gesture_proj(enc), dim=-1)

    def encode_prototype(self, prototype: torch.Tensor) -> torch.Tensor:
        """Get prototype embedding (for inference, includes projection)."""
        enc = self.prototype_encoder(prototype)
        return F.normalize(self.prototype_proj(enc), dim=-1)

    def forward(
        self,
        gesture: torch.Tensor,
        prototype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            gesture: (B, seq_len, 3) gesture trajectories
            prototype: (B, seq_len, 2) word prototypes (x, y only)

        Returns:
            gesture_proj: Projected gesture embeddings
            prototype_proj: Projected prototype embeddings
            temperature: Current temperature value
        """
        g_emb = self.gesture_encoder(gesture)
        p_emb = self.prototype_encoder(prototype)
        g_proj = self.gesture_proj(g_emb)
        p_proj = self.prototype_proj(p_emb)
        return g_proj, p_proj, self.temperature
