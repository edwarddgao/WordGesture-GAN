"""Modal app for training CTC decoder on cloud GPUs."""

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
    )
    .add_local_dir("ctc", remote_path="/app/ctc")
    .add_local_dir("shared", remote_path="/app/shared")
)

app = modal.App("ctc-decoder", image=image)

# Persistent volume for data and checkpoints (shared with wg_gan)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="L40S",
    volumes={"/data": vol},
    timeout=7200,
)
def train_on_modal(
    epochs: int | None = None,
    batch_size: int | None = None,
    hidden_size: int | None = None,
):
    """Train CTC decoder on Modal.

    Args:
        epochs: Override epochs from config (optional)
        batch_size: Override batch size from config (optional)
        hidden_size: Override LSTM hidden size from config (optional)
    """
    import os
    import sys
    from pathlib import Path

    import torch
    import yaml

    os.chdir("/app")
    sys.path.insert(0, "/app")

    # Load config
    config_path = Path("/app/ctc/config.yaml")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Override config values if provided
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    if hidden_size is not None:
        cfg["model"]["hidden_size"] = hidden_size

    # Update paths to use volume
    cfg["data"]["data_dir"] = "/data/processed"
    cfg["training"]["checkpoint_dir"] = "/data/checkpoints/ctc"

    # Ensure checkpoint directory exists
    Path("/data/checkpoints/ctc").mkdir(parents=True, exist_ok=True)

    # Write modified config
    temp_config = Path("/tmp/training_config.yaml")
    with temp_config.open("w") as f:
        yaml.safe_dump(cfg, f)

    print("=" * 60)
    print("CTC Decoder Training on Modal")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: {device}")
    print(f"Epochs: {cfg['training']['epochs']}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Hidden size: {cfg['model']['hidden_size']}")
    print(f"Num layers: {cfg['model']['num_layers']}")
    print("=" * 60)

    # Import and call the training function directly
    from ctc.train import train

    train(temp_config)

    # Commit volume to persist checkpoints
    vol.commit()
    print("\nTraining complete! Checkpoints saved to volume.")


@app.local_entrypoint()
def main(
    epochs: int = None,
    batch_size: int = None,
    hidden_size: int = None,
):
    """CLI entrypoint for training.

    Usage:
        modal run ctc/modal_app.py --epochs 100
        modal run ctc/modal_app.py --epochs 50 --batch-size 128
        modal run ctc/modal_app.py --epochs 100 --hidden-size 256
    """
    train_on_modal.remote(
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )
