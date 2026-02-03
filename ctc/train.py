"""Train the BLSTM CTC decoder model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared import KeyboardLayout, get_device, load_yaml

from .data import CTCGestureDataset, ctc_collate_fn


def get_ctc_device() -> torch.device:
    """Get device for CTC training.

    Note: CTC loss is not implemented on MPS, so we fall back to CPU
    if MPS is the default device. CUDA works fine.
    """
    device = get_device()
    if device.type == "mps":
        print("Warning: CTC loss not supported on MPS, using CPU instead.")
        return torch.device("cpu")
    # CUDA works fine for CTC
    return device
from features import GestureFeatureExtractor
from .models import BLSTMCTCModel


def compute_cer(predictions: list[str], targets: list[str]) -> float:
    """Compute Character Error Rate.

    Args:
        predictions: List of predicted strings.
        targets: List of target strings.

    Returns:
        Character Error Rate (0-1).
    """
    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        # Levenshtein distance
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == target[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        total_errors += dp[m][n]
        total_chars += max(n, 1)  # Avoid division by zero

    return total_errors / total_chars if total_chars > 0 else 0.0


def train(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    device = get_ctc_device()
    print(f"Using device: {device}")

    # Feature extractor
    layout = KeyboardLayout()
    feature_extractor = GestureFeatureExtractor(
        layout=layout,
        sigma=cfg["data"]["key_proximity_sigma"],
        use_key_proximity=cfg["features"]["use_key_proximity"],
        use_velocity=cfg["features"]["use_velocity"],
    )
    print(f"Feature extractor: {feature_extractor.n_features} features per timestep")

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    n_points = int(cfg["data"]["n_points"])
    batch_size = int(cfg["training"]["batch_size"])

    train_dataset = CTCGestureDataset(
        data_dir / "train.npz",
        n_points=n_points,
        feature_extractor=feature_extractor,
        augment=cfg["augmentation"]["enabled"],
        noise_std=cfg["augmentation"]["noise_std"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    # Optional: validation set
    test_path = data_dir / "test.npz"
    val_dataset = None
    val_loader = None
    if test_path.exists():
        val_dataset = CTCGestureDataset(
            test_path,
            n_points=n_points,
            feature_extractor=feature_extractor,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
            num_workers=0,
        )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    # Model
    model = BLSTMCTCModel(
        input_size=feature_extractor.n_features,
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    # Gradient clipping threshold
    grad_clip = cfg["training"].get("grad_clip", 5.0)

    # Checkpoints
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    n_epochs = int(cfg["training"]["epochs"])
    best_val_cer = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_metrics: Dict[str, float] = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for features, targets, input_lengths, target_lengths in pbar:
            features = features.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            # Forward pass
            logits = model(features)  # (B, T, C)

            # CTC loss expects (T, B, C) and log probabilities
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            # Compute loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_metrics["loss"] += loss.item()
            n_batches += 1

            pbar.set_postfix(loss=epoch_metrics["loss"] / n_batches)

        scheduler.step()

        # Compute training metrics
        avg_loss = epoch_metrics["loss"] / n_batches

        # Validation
        val_cer = None
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for features, targets, input_lengths, target_lengths in val_loader:
                    features = features.to(device)
                    logits = model(features)
                    decoded = model.decode_greedy(logits)
                    all_preds.extend(decoded)

                    # Convert targets to strings
                    for t, l in zip(targets, target_lengths):
                        target_str = "".join(
                            chr(ord("a") + c - 1) for c in t[:l].tolist()
                        )
                        all_targets.append(target_str)

            val_cer = compute_cer(all_preds, all_targets)

            # Show some examples
            print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, val_cer={val_cer:.4f}")
            print("  Examples (pred | target):")
            for i in range(min(5, len(all_preds))):
                print(f"    {all_preds[i]:20} | {all_targets[i]}")
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
        }

        # Always save latest
        torch.save(checkpoint, checkpoint_dir / "ctc_latest.pt")

        # Save periodic checkpoints
        if epoch % int(cfg["training"]["checkpoint_every"]) == 0:
            torch.save(checkpoint, checkpoint_dir / f"ctc_epoch_{epoch}.pt")

        # Save best model based on validation CER
        if val_cer is not None and val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(checkpoint, checkpoint_dir / "ctc_best.pt")
            print(f"  New best model (CER={val_cer:.4f})")

    print(f"\nTraining complete. Checkpoints saved to {checkpoint_dir}")
    if val_loader is not None:
        print(f"Best validation CER: {best_val_cer:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BLSTM CTC decoder model.")
    parser.add_argument(
        "--config", type=Path, default=Path("ctc/config.yaml"),
        help="Path to config YAML file."
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
