"""Train WordGesture-GAN."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared import KeyboardLayout, build_word_prototype, get_device, load_dataset, load_yaml
from wg_gan.models import Discriminator, Generator, VariationalEncoder


class GestureDataset(Dataset):
    def __init__(self, npz_path: Path, n_points: int, layout: KeyboardLayout) -> None:
        gestures, words = load_dataset(npz_path)
        self.gestures = gestures
        self.words = words
        self.n_points = n_points
        self.layout = layout
        self._prototype_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.gestures)

    def __getitem__(self, idx: int):
        gesture = self.gestures[idx]
        word = self.words[idx]
        if word not in self._prototype_cache:
            self._prototype_cache[word] = build_word_prototype(word, self.n_points, self.layout)
        prototype = self._prototype_cache[word]
        return (
            torch.from_numpy(gesture),
            torch.from_numpy(prototype),
        )


@contextmanager
def freeze_module(module: nn.Module):
    reqs = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad = False
    try:
        yield
    finally:
        for p, req in zip(module.parameters(), reqs):
            p.requires_grad = req


def train(config_path: Path) -> None:
    """Train WordGesture-GAN using the given config file.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg = load_yaml(config_path)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(cfg["data"]["data_dir"])
    n_points = int(cfg["data"]["n_points"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    n_epochs = int(cfg["training"]["epochs"])
    n_critic = int(cfg["training"]["n_critic"])
    latent_dim = int(cfg["model"]["latent_dim"])
    hidden_size = int(cfg["model"]["hidden_size"])
    num_layers = int(cfg["model"]["num_layers"])
    lambdas = cfg["loss"]

    layout = KeyboardLayout()
    dataset = GestureDataset(data_dir / "train.npz", n_points, layout)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    generator = Generator(latent_dim=latent_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)
    encoder = VariationalEncoder(input_dim=n_points * 3, latent_dim=latent_dim).to(device)
    disc_z = Discriminator(input_dim=n_points * 3).to(device)
    disc_x = Discriminator(input_dim=n_points * 3).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_e = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_dz = torch.optim.Adam(disc_z.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_dx = torch.optim.Adam(disc_x.parameters(), lr=lr, betas=(0.5, 0.999))

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    for epoch in range(1, n_epochs + 1):
        generator.train()
        encoder.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}")
        epoch_losses = {"d": 0.0, "g": 0.0}
        n_batches = 0
        for real, prototype in pbar:
            real = real.to(device)
            prototype = prototype.to(device)
            batch = real.size(0)

            # Discriminator updates
            for _ in range(n_critic):
                z_random = torch.randn(batch, latent_dim, device=device)
                fake_z = generator(prototype, z_random).detach()
                with torch.no_grad():
                    z_real, _, _ = encoder(real)
                    fake_x = generator(prototype, z_real).detach()

                dz_real, _ = disc_z(real)
                dz_fake, _ = disc_z(fake_z)
                dx_real, _ = disc_x(real)
                dx_fake, _ = disc_x(fake_x)

                loss_dz = dz_fake.mean() - dz_real.mean()
                loss_dx = dx_fake.mean() - dx_real.mean()

                opt_dz.zero_grad()
                loss_dz.backward()
                opt_dz.step()

                opt_dx.zero_grad()
                loss_dx.backward()
                opt_dx.step()

            # Generator + Encoder update
            z_random = torch.randn(batch, latent_dim, device=device)
            fake_z = generator(prototype, z_random)
            with freeze_module(encoder):
                z_rec, _, _ = encoder(fake_z)
            lat_loss = torch.mean(torch.abs(z_rec - z_random))

            z_real, mu_real, logvar_real = encoder(real)
            fake_x = generator(prototype, z_real)

            dz_fake, dz_feat_fake = disc_z(fake_z)
            dz_real, dz_feat_real = disc_z(real)
            dx_fake, dx_feat_fake = disc_x(fake_x)
            dx_real, dx_feat_real = disc_x(real)

            adv_loss = -0.5 * (dz_fake.mean() + dx_fake.mean())

            feat_loss = 0.0
            for f_fake, f_real in zip(dz_feat_fake, dz_feat_real):
                feat_loss += torch.mean(torch.abs(f_fake - f_real))
            for f_fake, f_real in zip(dx_feat_fake, dx_feat_real):
                feat_loss += torch.mean(torch.abs(f_fake - f_real))
            feat_loss = feat_loss / max(len(dz_feat_fake) + len(dx_feat_fake), 1)

            rec_loss = torch.mean(torch.abs(fake_x - real))
            kld = -0.5 * torch.mean(1 + logvar_real - mu_real.pow(2) - logvar_real.exp())

            loss_g = (
                adv_loss
                + float(lambdas["lambda_feat"]) * feat_loss
                + float(lambdas["lambda_rec"]) * rec_loss
                + float(lambdas["lambda_lat"]) * lat_loss
                + float(lambdas["lambda_kld"]) * kld
            )

            opt_g.zero_grad()
            opt_e.zero_grad()
            loss_g.backward()
            opt_g.step()
            opt_e.step()

            epoch_losses["d"] += (loss_dz.item() + loss_dx.item()) / 2
            epoch_losses["g"] += loss_g.item()
            n_batches += 1
            pbar.set_postfix(D=epoch_losses["d"] / n_batches, G=epoch_losses["g"] / n_batches)

        print(f"Epoch {epoch}: D_loss={epoch_losses['d']/n_batches:.4f}, G_loss={epoch_losses['g']/n_batches:.4f}")

        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "encoder": encoder.state_dict(),
            "disc_z": disc_z.state_dict(),
            "disc_x": disc_x.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, checkpoint_dir / "wg_gan_latest.pt")
        if epoch % int(cfg["training"]["checkpoint_every"]) == 0:
            torch.save(checkpoint, checkpoint_dir / f"wg_gan_epoch_{epoch}.pt")


def main() -> None:
    """CLI entrypoint for training WordGesture-GAN."""
    parser = argparse.ArgumentParser(description="Train WordGesture-GAN.")
    parser.add_argument("--config", required=True, type=Path, help="Path to config YAML file.")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
