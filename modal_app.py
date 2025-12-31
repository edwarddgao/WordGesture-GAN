#!/usr/bin/env python3
"""
Modal app for running WordGesture-GAN experiments on cloud GPUs.

Usage:
    modal run modal_app.py::train
    modal run modal_app.py::train --epochs 100 --batch-size 256
"""

import modal

# Create the Modal app
app = modal.App("wordgesture-gan")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    )
    .add_local_dir("src", remote_path="/app/src")
    .add_local_file("train.py", remote_path="/app/train.py")
    .add_local_file("evaluate.py", remote_path="/app/evaluate.py")
)

# Create volumes for dataset and checkpoints
dataset_volume = modal.Volume.from_name("wordgesture-dataset", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("wordgesture-checkpoints", create_if_missing=True)

DATASET_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,  # 4 hours max
    volumes={
        DATASET_PATH: dataset_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
)
def train(
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 0.0002,
    latent_dim: int = 32,
    n_critic: int = 5,
    save_every: int = 10,
    seed: int = 42,
):
    """Train WordGesture-GAN on Modal with GPU."""
    import os
    import sys
    import subprocess

    # Add app directory to path
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Check if dataset exists
    dataset_file = f"{DATASET_PATH}/swipelogs.zip"
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset not found at {dataset_file}")
        print("Please upload the dataset first using: modal run modal_app.py::upload_dataset")
        return {"status": "error", "message": "Dataset not found"}

    # Run training
    cmd = [
        sys.executable,
        "train.py",
        "--data_path", dataset_file,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--latent_dim", str(latent_dim),
        "--n_critic", str(n_critic),
        "--checkpoint_dir", CHECKPOINTS_PATH,
        "--save_every", str(save_every),
        "--seed", str(seed),
        "--device", "cuda",
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)

    # Run training with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()

    # Commit volume changes
    checkpoints_volume.commit()

    return {
        "status": "success" if process.returncode == 0 else "error",
        "return_code": process.returncode,
        "epochs": epochs,
        "checkpoints_path": CHECKPOINTS_PATH,
    }


@app.function(
    image=image,
    volumes={DATASET_PATH: dataset_volume},
    timeout=600,
)
def upload_dataset(local_path: str = "dataset/swipelogs.zip"):
    """Upload dataset to Modal volume."""
    import shutil
    import os

    # This function is called locally to upload the dataset
    dest_path = f"{DATASET_PATH}/swipelogs.zip"

    if os.path.exists(dest_path):
        print(f"Dataset already exists at {dest_path}")
        return {"status": "exists", "path": dest_path}

    print(f"Dataset would be uploaded to {dest_path}")
    return {"status": "ready", "path": dest_path}


@app.local_entrypoint()
def main(
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 0.0002,
    upload_data: bool = False,
):
    """Main entrypoint for Modal training."""
    import os

    if upload_data:
        # First upload the dataset
        local_dataset = "dataset/swipelogs.zip"
        if os.path.exists(local_dataset):
            print(f"Uploading dataset from {local_dataset}...")
            # Use modal volume put command
            import subprocess
            subprocess.run([
                "modal", "volume", "put",
                "wordgesture-dataset",
                local_dataset,
                "/swipelogs.zip"
            ], check=True)
            print("Dataset uploaded successfully!")
        else:
            print(f"Warning: Local dataset not found at {local_dataset}")

    # Run training
    print(f"\nStarting training with epochs={epochs}, batch_size={batch_size}, lr={lr}")
    result = train.remote(epochs=epochs, batch_size=batch_size, lr=lr)
    print(f"\nTraining result: {result}")
