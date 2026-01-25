"""WordGesture-GAN model definitions."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim: int = 384, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(192, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(48, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        hidden = self.net(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 16,
        num_layers: int = 5,
        dt_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.dt_scale = dt_scale
        self.lstm = nn.LSTM(
            input_size=3 + latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 3)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, prototype: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = prototype.shape
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([prototype, z_rep], dim=-1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        # x, y use tanh [-1, 1]; dt uses sigmoid * scale [0, dt_scale]
        xy = self.tanh(out[..., :2])
        dt = self.sigmoid(out[..., 2:3]) * self.dt_scale
        return torch.cat([xy, dt], dim=-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 384) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.utils.spectral_norm(nn.Linear(input_dim, 192)),
                nn.utils.spectral_norm(nn.Linear(192, 96)),
                nn.utils.spectral_norm(nn.Linear(96, 48)),
                nn.utils.spectral_norm(nn.Linear(48, 24)),
                nn.utils.spectral_norm(nn.Linear(24, 1)),
            ]
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.view(x.size(0), -1)
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
            feats.append(x)
        out = self.layers[-1](x)
        return out.squeeze(-1), feats


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
