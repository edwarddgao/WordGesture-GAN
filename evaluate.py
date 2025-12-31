#!/usr/bin/env python3
"""
Evaluation script for WordGesture-GAN.

Usage:
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt \
                       --data_path dataset/swipelogs.zip
"""

import argparse
import os
import sys
import torch
import numpy as np
from collections import defaultdict
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ModelConfig, TrainingConfig
from src.keyboard import QWERTYKeyboard
from src.data import (
    load_dataset_from_zip,
    create_train_test_split,
    GestureDataset
)
from src.models import Generator, VariationalEncoder
from src.evaluation import (
    compute_wasserstein_distance,
    compute_precision_recall,
    compute_velocity_correlation,
    compute_acceleration_correlation,
    compute_mean_jerk,
    FIDCalculator,
    evaluate_model
)
from src.visualization import (
    plot_gesture_comparison,
    plot_multiple_gestures,
    create_gesture_grid
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate WordGesture-GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--data_path', type=str, default='dataset/swipelogs.zip',
                       help='Path to swipelogs.zip')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples per word to generate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files to process (for debugging)')

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get('model_config', ModelConfig())

    generator = Generator(model_config).to(device)
    encoder = VariationalEncoder(model_config).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    generator.eval()
    encoder.eval()

    return generator, encoder, model_config


def generate_gestures_for_evaluation(
    generator: torch.nn.Module,
    prototypes_by_word: dict,
    gestures_by_word: dict,
    model_config: ModelConfig,
    device: torch.device,
    num_samples: int = None
) -> dict:
    """Generate gestures for all words in the test set."""
    generated_by_word = defaultdict(list)

    with torch.no_grad():
        for word, prototype in prototypes_by_word.items():
            if word not in gestures_by_word:
                continue

            # Number of samples to generate
            n = num_samples if num_samples else len(gestures_by_word[word])

            prototype_tensor = torch.FloatTensor(prototype).unsqueeze(0).to(device)

            for _ in range(n):
                # Sample random latent code
                z = torch.randn(1, model_config.latent_dim, device=device)

                # Generate gesture
                generated = generator(prototype_tensor, z)
                generated_np = generated.squeeze(0).cpu().numpy()

                generated_by_word[word].append(generated_np)

    return dict(generated_by_word)


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    generator, encoder, model_config = load_model(args.checkpoint, device)

    # Initialize keyboard and training config
    keyboard = QWERTYKeyboard()
    training_config = TrainingConfig()

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    gestures_by_word, prototypes_by_word = load_dataset_from_zip(
        args.data_path,
        keyboard,
        model_config,
        training_config,
        max_files=args.max_files
    )

    # Create train/test split (use same seed as training)
    _, test_dataset = create_train_test_split(
        gestures_by_word,
        prototypes_by_word,
        train_ratio=0.8,
        seed=42
    )

    # Get test words
    test_words = set(test_dataset.words)
    test_gestures_by_word = defaultdict(list)
    test_prototypes_by_word = {}

    for i in range(len(test_dataset)):
        word = test_dataset.words[i]
        gesture = test_dataset.gestures[i]
        prototype = test_dataset.prototypes[i]

        test_gestures_by_word[word].append(gesture)
        test_prototypes_by_word[word] = prototype

    print(f"Test set: {len(test_words)} words, {len(test_dataset)} samples")

    # Generate gestures
    print("Generating gestures...")
    generated_by_word = generate_gestures_for_evaluation(
        generator,
        test_prototypes_by_word,
        test_gestures_by_word,
        model_config,
        device,
        num_samples=args.num_samples
    )

    total_generated = sum(len(v) for v in generated_by_word.values())
    print(f"Generated {total_generated} gestures for {len(generated_by_word)} words")

    # Evaluate
    print("\nComputing evaluation metrics...")
    results = evaluate_model(
        dict(test_gestures_by_word),
        generated_by_word
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Wasserstein Distances ---")
    print(f"L2 (x,y) mean: {results.get('l2_wasserstein_xy_mean', 'N/A'):.4f} "
          f"(std: {results.get('l2_wasserstein_xy_std', 'N/A'):.4f})")
    print(f"L2 (x,y,t) mean: {results.get('l2_wasserstein_xyz_mean', 'N/A'):.4f} "
          f"(std: {results.get('l2_wasserstein_xyz_std', 'N/A'):.4f})")
    print(f"DTW (x,y,t) mean: {results.get('dtw_wasserstein_xyz_mean', 'N/A'):.4f} "
          f"(std: {results.get('dtw_wasserstein_xyz_std', 'N/A'):.4f})")

    print("\n--- Precision & Recall ---")
    print(f"Precision: {results.get('precision', 'N/A'):.4f}")
    print(f"Recall: {results.get('recall', 'N/A'):.4f}")

    print("\n--- Velocity & Acceleration Correlations ---")
    print(f"Velocity corr mean: {results.get('velocity_correlation_mean', 'N/A'):.4f} "
          f"(std: {results.get('velocity_correlation_std', 'N/A'):.4f})")
    print(f"Acceleration corr mean: {results.get('acceleration_correlation_mean', 'N/A'):.4f} "
          f"(std: {results.get('acceleration_correlation_std', 'N/A'):.4f})")

    print("\n--- Jerk Analysis ---")
    print(f"Real jerk mean: {results.get('real_jerk_mean', 'N/A'):.6f} "
          f"(std: {results.get('real_jerk_std', 'N/A'):.6f})")
    print(f"Generated jerk mean: {results.get('fake_jerk_mean', 'N/A'):.6f} "
          f"(std: {results.get('fake_jerk_std', 'N/A'):.6f})")

    print("=" * 60)

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Sample some words for visualization
    sample_words = list(test_gestures_by_word.keys())[:5]

    for word in sample_words:
        if word not in generated_by_word or len(generated_by_word[word]) == 0:
            continue

        real = test_gestures_by_word[word][0]
        fake = generated_by_word[word][0]
        prototype = test_prototypes_by_word[word]

        fig = plot_gesture_comparison(
            real, fake, prototype, word, keyboard,
            save_path=os.path.join(args.output_dir, f'comparison_{word}.png')
        )
        plt.close(fig)

    # Create gesture grid for generated samples
    fig = create_gesture_grid(
        generated_by_word,
        n_words=min(4, len(generated_by_word)),
        n_samples=3,
        keyboard=keyboard,
        save_path=os.path.join(args.output_dir, 'generated_samples_grid.png')
    )
    plt.close(fig)

    print(f"\nVisualizations saved to {args.output_dir}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
