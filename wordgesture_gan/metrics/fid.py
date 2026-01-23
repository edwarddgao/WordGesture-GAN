"""FID computation using a simple autoencoder."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wordgesture_gan.utils import get_device


class GestureAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 384, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(
    gestures: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> GestureAutoEncoder:
    if device is None:
        device = get_device()
    model = GestureAutoEncoder(input_dim=gestures.shape[1] * gestures.shape[2], latent_dim=latent_dim).to(device)
    dataset = TensorDataset(torch.from_numpy(gestures.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch.view(batch.size(0), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _compute_stats(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = embeddings.mean(axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(
    real: np.ndarray,
    fake: np.ndarray,
    model: GestureAutoEncoder,
    device: torch.device | None = None,
) -> float:
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        real_tensor = torch.from_numpy(real.astype(np.float32)).to(device)
        fake_tensor = torch.from_numpy(fake.astype(np.float32)).to(device)
        _, real_z = model(real_tensor)
        _, fake_z = model(fake_tensor)
    real_z = real_z.cpu().numpy()
    fake_z = fake_z.cpu().numpy()
    mu1, sigma1 = _compute_stats(real_z)
    mu2, sigma2 = _compute_stats(fake_z)
    return frechet_distance(mu1, sigma1, mu2, sigma2)
