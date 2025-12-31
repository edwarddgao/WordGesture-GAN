"""Modal training functions for WordGesture-GAN."""
import modal
import os
from pathlib import Path

app = modal.App("wordgesture-gan-training")

# Mount local directories
local_src = modal.Mount.from_local_dir(Path(__file__).parent / "src", remote_path="/root/src")
local_dataset = modal.Mount.from_local_dir(Path(__file__).parent / "dataset", remote_path="/root/dataset")

# Training image
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.0.0", "numpy>=1.24.0", "scipy>=1.10.0", "pandas>=2.0.0", "matplotlib>=3.7.0", "tqdm>=4.65.0")
    .env({"PYTHONPATH": "/root"})
)

# Volume for checkpoints
volume = modal.Volume.from_name("wordgesture-gan-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(gpu="T4", timeout=300, image=training_image, mounts=[local_src])
def test_gpu():
    """Test GPU access and src/ imports."""
    import torch
    from src.config import ModelConfig
    from src.models import Generator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # Quick GPU test
    if device == "cuda":
        x = torch.randn(100, 100, device=device)
        _ = x @ x.T

    return {"device": device, "gpu": gpu_name, "status": "ok"}


@app.function(gpu="T4", timeout=7200, image=training_image, volumes={VOLUME_PATH: volume}, mounts=[local_src, local_dataset])
def train(epochs: int = 200, batch_size: int = 512, learning_rate: float = 0.0002):
    """Run training with mounted dataset."""
    import torch
    from src.config import TrainingConfig, ModelConfig
    from src.trainer import Trainer
    from src.data import SwipeDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset from mount
    data_path = "/root/dataset/swipelogs.zip"
    if not os.path.exists(data_path):
        return {"status": "error", "message": f"Dataset not found at {data_path}"}

    model_config = ModelConfig()
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        checkpoint_dir=VOLUME_PATH,
    )

    dataset = SwipeDataset(data_path)
    trainer = Trainer(model_config, training_config, device=device)
    trainer.train(dataset)

    volume.commit()
    return {"status": "completed", "epochs": epochs, "checkpoint_dir": VOLUME_PATH}


@app.function(timeout=300, image=training_image, volumes={VOLUME_PATH: volume})
def list_checkpoints():
    """List saved checkpoints."""
    contents = []
    for root, dirs, files in os.walk(VOLUME_PATH):
        for f in files:
            path = os.path.join(root, f)
            contents.append({"path": path, "size": os.path.getsize(path)})
    return {"files": contents}
