"""Modal app for training WordGesture-GAN on cloud GPUs."""

import modal

VOLUME_NAME = "wordgesture-gan-data"

# Container image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "numba",
    )
    .env({"NUMBA_CACHE_DIR": "/tmp/numba_cache"})
    .add_local_dir("wg_gan", remote_path="/app/wg_gan")
    .add_local_dir("shared", remote_path="/app/shared")
)

app = modal.App("wordgesture-gan", image=image)

# Persistent volume for data and checkpoints
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="L40S",
    volumes={"/data": vol},
    timeout=7200,
)
def train_on_modal(epochs: int | None = None, batch_size: int | None = None):
    """Train WordGesture-GAN on Modal.

    Args:
        epochs: Override epochs from config (optional)
        batch_size: Override batch size from config (optional)
    """
    import os
    import sys
    from pathlib import Path

    import torch
    import yaml

    os.chdir("/app")
    sys.path.insert(0, "/app")

    # Load config
    config_path = Path("/app/wg_gan/config.yaml")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Override config values if provided
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size

    # Update paths to use volume
    cfg["data"]["data_dir"] = "/data/processed"
    cfg["training"]["checkpoint_dir"] = "/data/checkpoints"

    # Write modified config
    temp_config = Path("/tmp/training_config.yaml")
    with temp_config.open("w") as f:
        yaml.safe_dump(cfg, f)

    print("=" * 60)
    print("WordGesture-GAN Training on Modal")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: {device}")
    print(f"Epochs: {cfg['training']['epochs']}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print("=" * 60)

    # Import and call the training function directly
    from wg_gan.train import train

    train(temp_config)

    # Commit volume to persist checkpoints
    vol.commit()
    print("\nTraining complete! Checkpoints saved to volume.")


@app.local_entrypoint()
def main(
    epochs: int = None,
    batch_size: int = None,
):
    """CLI entrypoint for training.

    Usage:
        modal run wg_gan/modal_app.py --epochs 50
        modal run wg_gan/modal_app.py --epochs 10 --batch-size 256
    """
    train_on_modal.remote(epochs=epochs, batch_size=batch_size)
