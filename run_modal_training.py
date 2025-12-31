#!/usr/bin/env python3
"""
Run WordGesture-GAN training on Modal with GPU.

Based on the paper: "WordGesture-GAN: Modeling Word-Gesture Movement with
Generative Adversarial Network" (CHI '23)

NOTE: The dataset upload through HTTP proxy is slow (~69MB).
For faster runs, consider uploading the dataset to a Modal Volume first.

Validated components:
- Modal connection and GPU access (Tesla T4)
- Full WordGesture-GAN architecture (Generator, Discriminator, Encoder)
- Two-cycle BicycleGAN training procedure

Usage:
    python run_modal_training.py
"""

import os
if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch  # Only patch locally, not in Modal container
import modal
import asyncio

# Create Modal app
app = modal.App("wordgesture-gan-training")

# Use persistent volume for dataset (faster than uploading each time)
data_volume = modal.Volume.from_name("wordgesture-data", create_if_missing=True)

# Create image with PyTorch and dependencies (src/data are in volume)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    )
)


@app.function(
    gpu="T4",
    image=image,
    volumes={"/data": data_volume},
    timeout=3600,  # 1 hour timeout
)
def train_wordgesture_gan(
    num_epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 0.0002,
    max_files: int = None,
):
    """
    Train WordGesture-GAN on Modal GPU.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizers
        max_files: Max number of log files to process (None for all)

    Returns:
        Dictionary with training results and final losses
    """
    import sys
    import torch
    import random
    import numpy as np
    from datetime import datetime

    # Add src to path (stored in volume)
    sys.path.insert(0, '/data')

    from src.config import ModelConfig, TrainingConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import (
        load_dataset_from_zip,
        create_train_test_split,
        create_data_loaders
    )
    from src.trainer import WordGestureGANTrainer

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create configs
    model_config = ModelConfig(
        seq_length=128,
        latent_dim=32
    )

    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        n_critic=5,
        lambda_feat=1.0,
        lambda_rec=5.0,
        lambda_lat=0.5,
        lambda_kld=0.05,
        save_every=10,
        log_every=50
    )

    print("=" * 60)
    print("WordGesture-GAN Training on Modal")
    print("=" * 60)
    print(f"Model Config: {model_config}")
    print(f"Training Config: {training_config}")
    print("=" * 60)

    # Initialize keyboard
    keyboard = QWERTYKeyboard()
    print("Keyboard initialized")

    # Load dataset
    data_path = '/data/swipelogs.zip'
    print(f"\nLoading dataset from {data_path}...")

    gestures_by_word, prototypes_by_word = load_dataset_from_zip(
        data_path,
        keyboard,
        model_config,
        training_config,
        max_files=max_files
    )

    print(f"Loaded {len(gestures_by_word)} unique words")
    total_gestures = sum(len(g) for g in gestures_by_word.values())
    print(f"Total gestures: {total_gestures}")

    # Create train/test split
    print("\nCreating train/test split...")
    train_dataset, test_dataset = create_train_test_split(
        gestures_by_word,
        prototypes_by_word,
        train_ratio=training_config.train_ratio,
        seed=seed
    )

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=training_config.batch_size,
        num_workers=2  # Reduce workers for Modal environment
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = WordGestureGANTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device
    )

    # Train
    print("\nStarting training...")
    start_time = datetime.now()

    trainer.train(
        train_loader,
        num_epochs=num_epochs,
        checkpoint_dir='/tmp/checkpoints'
    )

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_duration:.1f} seconds")

    # Get final losses
    final_losses = {}
    if trainer.training_history:
        final_losses = trainer.training_history[-1]

    # Generate some sample gestures for validation
    print("\nGenerating sample gestures...")
    test_words = ['hello', 'world', 'gesture', 'typing', 'keyboard']
    sample_prototypes = []
    for word in test_words:
        proto = keyboard.get_word_prototype(word, model_config.seq_length)
        sample_prototypes.append(proto)

    sample_prototypes = torch.FloatTensor(np.stack(sample_prototypes)).to(device)
    generated_gestures = trainer.generate_gestures(sample_prototypes, num_samples=1)

    print(f"Generated {len(generated_gestures)} sample gestures")

    results = {
        'training_duration_seconds': training_duration,
        'num_epochs': num_epochs,
        'final_losses': final_losses,
        'device': device,
        'total_training_samples': len(train_dataset),
        'total_test_samples': len(test_dataset),
        'sample_words': test_words,
        'training_history': trainer.training_history
    }

    return results


async def main():
    """Main entry point."""
    print("Starting WordGesture-GAN training on Modal...")
    print("=" * 60)

    async with app.run():
        # Run training with a shorter experiment first
        # Paper uses 200 epochs, but let's do 50 for initial testing
        result = await train_wordgesture_gan.remote.aio(
            num_epochs=50,
            batch_size=512,
            learning_rate=0.0002,
            max_files=None  # Use all files
        )

        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Training Duration: {result['training_duration_seconds']:.1f} seconds")
        print(f"Epochs: {result['num_epochs']}")
        print(f"Training Samples: {result['total_training_samples']}")
        print(f"Test Samples: {result['total_test_samples']}")
        print(f"Device: {result['device']}")

        if result['final_losses']:
            print("\nFinal Losses:")
            for key, value in result['final_losses'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

        print("\nTraining History Summary:")
        history = result['training_history']
        if history:
            print(f"  First epoch losses: {history[0]}")
            print(f"  Last epoch losses: {history[-1]}")

        return result


if __name__ == "__main__":
    asyncio.run(main())
