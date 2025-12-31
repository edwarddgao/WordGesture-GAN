#!/usr/bin/env python3
"""
Main training script for WordGesture-GAN.

Usage:
    python train.py --data_path dataset/swipelogs.zip --epochs 200
    python train.py --resume checkpoints/checkpoint_epoch_100.pt
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ModelConfig, TrainingConfig
from src.keyboard import QWERTYKeyboard
from src.data import (
    load_dataset_from_zip,
    create_train_test_split,
    create_data_loaders
)
from src.trainer import WordGestureGANTrainer
from src.visualization import plot_training_curves, plot_gesture_comparison


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train WordGesture-GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, default='dataset/swipelogs.zip',
                       help='Path to swipelogs.zip')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of log files to process (for debugging)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--n_critic', type=int, default=5,
                       help='Number of discriminator updates per generator update')

    # Loss weights
    parser.add_argument('--lambda_feat', type=float, default=1.0,
                       help='Feature matching loss weight')
    parser.add_argument('--lambda_rec', type=float, default=5.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--lambda_lat', type=float, default=0.5,
                       help='Latent encoding loss weight')
    parser.add_argument('--lambda_kld', type=float, default=0.05,
                       help='KL divergence loss weight')

    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent code dimension')
    parser.add_argument('--seq_length', type=int, default=128,
                       help='Gesture sequence length')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create configs
    model_config = ModelConfig(
        seq_length=args.seq_length,
        latent_dim=args.latent_dim
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        n_critic=args.n_critic,
        lambda_feat=args.lambda_feat,
        lambda_rec=args.lambda_rec,
        lambda_lat=args.lambda_lat,
        lambda_kld=args.lambda_kld,
        save_every=args.save_every
    )

    print("=" * 60)
    print("WordGesture-GAN Training")
    print("=" * 60)
    print(f"Model Config: {model_config}")
    print(f"Training Config: {training_config}")
    print("=" * 60)

    # Initialize keyboard
    keyboard = QWERTYKeyboard()
    print("Keyboard initialized")

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please ensure the dataset is available.")
        sys.exit(1)

    gestures_by_word, prototypes_by_word = load_dataset_from_zip(
        args.data_path,
        keyboard,
        model_config,
        training_config,
        max_files=args.max_files
    )

    # Create train/test split
    print("\nCreating train/test split...")
    train_dataset, test_dataset = create_train_test_split(
        gestures_by_word,
        prototypes_by_word,
        train_ratio=training_config.train_ratio,
        seed=args.seed
    )

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=training_config.batch_size,
        num_workers=args.num_workers
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

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Train
    print("\nStarting training...")
    start_time = datetime.now()

    trainer.train(
        train_loader,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )

    end_time = datetime.now()
    print(f"\nTraining completed in {end_time - start_time}")

    # Save training curves
    if trainer.training_history:
        print("\nSaving training curves...")
        fig = plot_training_curves(trainer.training_history)
        fig.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("\nDone!")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    main()
