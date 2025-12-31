#!/usr/bin/env python3
"""
Modal training script for WordGesture-GAN.

Runs the training experiment on Modal with GPU acceleration.

Usage:
    modal run modal_train.py
    modal run modal_train.py --epochs 100
"""

import modal

# Create Modal app
app = modal.App("wordgesture-gan-training")

# Define the container image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    )
    .add_local_dir(
        "/home/user/WordGesture-GAN/src",
        remote_path="/app/src",
    )
    .add_local_file(
        "/home/user/WordGesture-GAN/train.py",
        remote_path="/app/train.py",
    )
    .add_local_dir(
        "/home/user/WordGesture-GAN/dataset",
        remote_path="/app/dataset",
    )
)

# Create a Modal volume for storing checkpoints
volume = modal.Volume.from_name("wordgesture-gan-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU for training
    timeout=3600 * 4,  # 4 hour timeout
    volumes={"/checkpoints": volume},
)
def train(
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 0.0002,
    save_every: int = 10,
):
    """Run WordGesture-GAN training on Modal."""
    import os
    import sys
    import subprocess

    # Change to app directory
    os.chdir("/app")
    sys.path.insert(0, "/app")

    # Print environment info
    import torch
    print("=" * 60)
    print("WordGesture-GAN Training on Modal")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # List contents
    print("\nApp directory contents:")
    for item in os.listdir("/app"):
        print(f"  {item}")

    # Run training
    cmd = [
        "python", "train.py",
        "--data_path", "dataset/swipelogs.zip",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--checkpoint_dir", "/checkpoints",
        "--save_every", str(save_every),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Execute training
    result = subprocess.run(cmd, capture_output=False)

    # Commit volume to persist checkpoints
    volume.commit()

    print("\n" + "=" * 60)
    print("Training complete! Checkpoints saved to volume.")
    print("=" * 60)

    return result.returncode


@app.local_entrypoint()
def main(epochs: int = 200, batch_size: int = 512, lr: float = 0.0002):
    """Local entrypoint for Modal training."""
    print("Starting WordGesture-GAN training on Modal...")
    print(f"Configuration: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    return_code = train.remote(epochs=epochs, batch_size=batch_size, lr=lr)

    if return_code == 0:
        print("\nTraining completed successfully!")
    else:
        print(f"\nTraining failed with return code: {return_code}")

    return return_code
