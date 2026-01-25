"""Train the two-tower contrastive model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared import KeyboardLayout, get_device

from .data import ContrastiveGestureDataset
from .losses import InfoNCELoss
from .models import TwoTowerModel


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train(config_path: Path) -> None:
    cfg = load_config(config_path)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    n_points = int(cfg["data"]["n_points"])
    batch_size = int(cfg["training"]["batch_size"])

    layout = KeyboardLayout()
    train_dataset = ContrastiveGestureDataset(
        data_dir / "train.npz",
        n_points=n_points,
        layout=layout,
        augment=cfg["augmentation"]["enabled"],
        noise_std=cfg["augmentation"]["noise_std"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Important for consistent batch size in contrastive learning
        num_workers=0,
    )

    print(f"Training samples: {len(train_dataset)}, vocab size: {train_dataset.vocab_size}")

    # Model
    model = TwoTowerModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        projection_dim=cfg["model"]["projection_dim"],
        temperature=cfg["contrastive"]["temperature"],
    ).to(device)

    loss_fn = InfoNCELoss()

    # Optimizer with separate LR for temperature
    optimizer = torch.optim.AdamW(
        [
            {"params": model.gesture_encoder.parameters()},
            {"params": model.prototype_encoder.parameters()},
            {"params": model.gesture_proj.parameters()},
            {"params": model.prototype_proj.parameters()},
            {"params": [model.log_temperature], "lr": cfg["training"]["lr_temp"]},
        ],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    # Checkpoints
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    n_epochs = int(cfg["training"]["epochs"])
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_metrics: Dict[str, float] = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for gesture, prototype, word_idx in pbar:
            gesture = gesture.to(device)
            prototype = prototype.to(device)

            g_proj, p_proj, temp = model(gesture, prototype)
            loss, metrics = loss_fn(g_proj, p_proj, temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            pbar.set_postfix(
                loss=epoch_metrics["loss_g2p"] / n_batches,
                acc=epoch_metrics["acc_g2p"] / n_batches,
                temp=metrics["temperature"],
            )

        scheduler.step()

        # Log epoch metrics
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(
            f"Epoch {epoch}: "
            f"loss_g2p={avg_metrics['loss_g2p']:.4f}, "
            f"loss_p2g={avg_metrics['loss_p2g']:.4f}, "
            f"acc_g2p={avg_metrics['acc_g2p']:.4f}, "
            f"acc_p2g={avg_metrics['acc_p2g']:.4f}, "
            f"temp={avg_metrics['temperature']:.4f}"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, checkpoint_dir / "contrastive_latest.pt")
        if epoch % int(cfg["training"]["checkpoint_every"]) == 0:
            torch.save(checkpoint, checkpoint_dir / f"contrastive_epoch_{epoch}.pt")

    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-tower contrastive model.")
    parser.add_argument("--config", type=Path, default=Path("contrastive/config.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
