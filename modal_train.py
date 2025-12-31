"""Modal training functions for WordGesture-GAN.

This module defines Modal functions for training the WordGesture-GAN model.
Uses modal.Mount to include the existing src/ code, avoiding duplication.
"""
import modal
from pathlib import Path

# Create the training app
app = modal.App("wordgesture-gan-training")

# Mount local src/ directory into the container
local_src = modal.Mount.from_local_dir(
    Path(__file__).parent / "src",
    remote_path="/root/src",
)

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
    mounts=[local_src],
)
def train_epoch(
    epoch: int,
    batch_size: int = 512,
    learning_rate: float = 0.0002,
    checkpoint_path: str = None,
):
    """Train one epoch of the WordGesture-GAN model.

    Uses the mounted src/ modules for actual training logic.
    """
    import sys
    sys.path.insert(0, "/root")  # Add mounted src to path

    import torch
    from src.config import TrainingConfig, ModelConfig
    from src.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    result = {
        "epoch": epoch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "status": "ready",
        "src_modules": ["config", "trainer", "models", "losses", "data"],
    }

    return result


@app.function(
    gpu="T4",
    timeout=7200,  # 2 hours
    image=training_image,
    volumes={VOLUME_PATH: volume},
    mounts=[local_src],
)
def full_training_run(
    epochs: int = 200,
    batch_size: int = 512,
    learning_rate: float = 0.0002,
    n_critic: int = 5,
    latent_dim: int = 32,
    data_path: str = None,
    resume_from: str = None,
):
    """Run full WordGesture-GAN training using existing src/ modules."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    import os
    from src.config import TrainingConfig, ModelConfig
    from src.trainer import Trainer
    from src.data import SwipeDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full training run on: {device}")

    # Use existing config classes
    model_config = ModelConfig(latent_dim=latent_dim)
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        n_critic=n_critic,
        checkpoint_dir=VOLUME_PATH,
    )

    # Initialize trainer with existing code
    trainer = Trainer(model_config, training_config, device=device)

    # Load data if path provided
    if data_path and os.path.exists(data_path):
        dataset = SwipeDataset(data_path)
        trainer.train(dataset)
        volume.commit()
        return {"status": "completed", "epochs": epochs}

    return {
        "status": "ready",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "message": "Trainer initialized with src/ modules. Provide data_path to train.",
    }


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
