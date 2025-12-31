"""Modal training functions for WordGesture-GAN.

This module defines Modal functions for training the WordGesture-GAN model.
Functions are defined here without any proxy-related imports so they can
be safely serialized and run on Modal's cloud infrastructure.
"""
import modal

# Create the training app
app = modal.App("wordgesture-gan-training")

# Define the training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    )
)

# Volume for storing checkpoints and results
volume = modal.Volume.from_name("wordgesture-gan-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    gpu="T4",
    timeout=3600,  # 1 hour
    image=training_image,
    volumes={VOLUME_PATH: volume},
)
def train_epoch(
    epoch: int,
    batch_size: int = 512,
    learning_rate: float = 0.0002,
    checkpoint_path: str = None,
):
    """Train one epoch of the WordGesture-GAN model.

    Args:
        epoch: Current epoch number
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
        checkpoint_path: Path to load checkpoint from (in volume)

    Returns:
        dict with training metrics and checkpoint info
    """
    import torch
    import numpy as np
    import os

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # For now, return a test result
    # Full training code would be loaded from the mounted volume
    result = {
        "epoch": epoch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "status": "test_complete",
    }

    # Test GPU computation
    if torch.cuda.is_available():
        x = torch.randn(batch_size, 128, 3, device=device)
        z = torch.mm(x.view(batch_size, -1), x.view(batch_size, -1).T)
        result["test_computation"] = float(z.sum())

    return result


@app.function(
    gpu="T4",
    timeout=7200,  # 2 hours
    image=training_image,
    volumes={VOLUME_PATH: volume},
)
def full_training_run(
    epochs: int = 200,
    batch_size: int = 512,
    learning_rate: float = 0.0002,
    n_critic: int = 5,
    latent_dim: int = 32,
    resume_from: str = None,
):
    """Run full WordGesture-GAN training.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
        n_critic: Discriminator updates per generator update
        latent_dim: Dimension of latent code
        resume_from: Checkpoint filename to resume from

    Returns:
        dict with final metrics and checkpoint info
    """
    import torch
    import os
    import json
    from datetime import datetime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full training run on: {device}")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # Training configuration
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_critic": n_critic,
        "latent_dim": latent_dim,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now().isoformat(),
    }

    # Save config
    config_path = os.path.join(VOLUME_PATH, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    volume.commit()

    # Placeholder for actual training loop
    # In production, this would:
    # 1. Load dataset from volume
    # 2. Initialize models (Generator, Discriminator, Encoder)
    # 3. Run training loop with two-cycle procedure
    # 4. Save checkpoints periodically
    # 5. Return final metrics

    result = {
        **config,
        "status": "training_placeholder",
        "message": "Full training code to be implemented with src/ modules",
    }

    return result


@app.function(
    timeout=300,
    image=training_image,
    volumes={VOLUME_PATH: volume},
)
def check_volume_contents():
    """Check what's in the training volume."""
    import os

    contents = []
    for root, dirs, files in os.walk(VOLUME_PATH):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            contents.append({"path": path, "size": size})

    return {
        "volume_path": VOLUME_PATH,
        "file_count": len(contents),
        "files": contents[:50],  # Limit to first 50
    }


@app.function(
    timeout=600,
    image=training_image,
    volumes={VOLUME_PATH: volume},
)
def upload_source_code(source_code: dict):
    """Upload source code to the volume for training.

    Args:
        source_code: dict mapping filename to file content

    Returns:
        dict with upload status
    """
    import os

    src_dir = os.path.join(VOLUME_PATH, "src")
    os.makedirs(src_dir, exist_ok=True)

    uploaded = []
    for filename, content in source_code.items():
        filepath = os.path.join(src_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        uploaded.append(filepath)

    volume.commit()

    return {
        "status": "success",
        "uploaded_files": uploaded,
        "src_dir": src_dir,
    }
