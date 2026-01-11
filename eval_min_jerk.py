#!/usr/bin/env python3
"""
Minimum Jerk Model Evaluation (Quinn & Zhai 2018)

Evaluates the minimum jerk baseline model using the same metrics from Table 6
of the WordGesture-GAN paper.

This implementation follows the paper's approach:
1. Fits offset distributions from training data (key center offsets, midpoint angles)
2. Generates trajectories by sampling from learned distributions

Usage:
    python eval_min_jerk.py                    # Evaluate with defaults
    python eval_min_jerk.py --n-samples 200    # Custom sample count
"""

import os
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('min-jerk-eval')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'fastdtw', 'joblib', 'matplotlib')
    .add_local_python_source('src')
)


EVAL_SCRIPT = '''
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
from src.keyboard import QWERTYKeyboard, MinimumJerkModel
from src.data import load_dataset_from_zip, create_train_test_split
from src.evaluation import evaluate_all_metrics

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
savgol_window = int(sys.argv[2]) if len(sys.argv) > 2 else 21
precision_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig(
    n_samples=n_samples,
    savgol_window=savgol_window,
    precision_recall_k=precision_k
)

print(f'Minimum Jerk Model Evaluation (Quinn & Zhai 2018 - learned distributions)')
print(f'  n_samples={n_samples}')
print(f'  savgol_window={savgol_window}, precision_k={precision_k}')
print()

# Load data
print('[1/5] Loading data...')
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)
print(f'  Train: {len(train_ds)}, Test: {len(test_ds)}')

# Fit minimum jerk model to training data (Quinn & Zhai 2018 approach)
print('[2/5] Fitting MinimumJerkModel to training data...')
# Reconstruct gestures_by_word from train_ds for fitting
train_gestures_by_word = defaultdict(list)
for i in range(len(train_ds)):
    item = train_ds[i]
    train_gestures_by_word[item['word']].append(item['gesture'].numpy())

min_jerk_model = MinimumJerkModel(keyboard)
min_jerk_model.fit(dict(train_gestures_by_word), verbose=True)

# Generate minimum jerk trajectories for test words using fitted model
n = min(n_samples, len(test_ds))
print(f'[3/5] Generating {n} minimum jerk trajectories with learned distributions...')

real_g, min_jerk_g, words = [], [], []
for i in range(n):
    item = test_ds[i]
    word = item['word']
    real_gesture = item['gesture'].numpy()

    # Generate minimum jerk trajectory using fitted model
    min_jerk = min_jerk_model.generate_trajectory(
        word,
        num_points=model_config.seq_length,
        include_midpoints=True
    )

    real_g.append(real_gesture)
    min_jerk_g.append(min_jerk)
    words.append(word)

    if (i + 1) % 100 == 0:
        print(f'  Generated {i+1}/{n}...')

real_g = np.array(real_g)
min_jerk_g = np.array(min_jerk_g)

# Get training data for FID autoencoder
print('[4/5] Preparing training data for FID...')
train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])

# Run all metrics
print('[5/5] Computing metrics...')
results = evaluate_all_metrics(real_g, min_jerk_g, train_g, model_config, eval_config, device)

# Print results table comparing to paper Table 6
print()
print('=' * 80)
print(f'{"Metric":<35} {"Ours":>15} {"Paper":>15} {"Notes":>12}')
print('=' * 80)
print(f'{"L2 Wasserstein (x,y)":<35} {results["l2_wasserstein"]:>15.3f} {"5.004":>15} {"lower=better":>12}')
print(f'{"DTW Wasserstein (x,y)":<35} {results["dtw_wasserstein"]:>15.3f} {"2.752":>15} {"lower=better":>12}')
print(f'{"Jerk (min jerk)":<35} {results["jerk_fake"]:>15.5f} {"0.0034":>15} {"~real":>12}')
print(f'{"Jerk (real)":<35} {results["jerk_real"]:>15.5f} {"0.0066":>15} {"reference":>12}')
print(f'{"Velocity Corr":<35} {results["velocity_corr"]:>15.3f} {"0.40":>15} {"higher=better":>12}')
print(f'{"Acceleration Corr":<35} {results["acceleration_corr"]:>15.3f} {"0.21":>15} {"higher=better":>12}')
print(f'{"Speed Profile Corr":<35} {results["speed_profile_corr"]:>15.3f} {"--":>15} {"higher=better":>12}')
print(f'{"Time Delta Corr":<35} {results["time_delta_corr"]:>15.3f} {"--":>15} {"higher=better":>12}')
print('-' * 80)
print(f'{"AE Reconstruction (L1)":<35} {results["ae_reconstruction_loss"]:>15.4f} {"0.041":>15} {"lower=better":>12}')
print(f'{"AE Test Loss (L1)":<35} {results["ae_test_loss"]:>15.4f} {"0.046":>15} {"lower=better":>12}')
print(f'{"FID":<35} {results["fid"]:>15.4f} {"0.354":>15} {"lower=better":>12}')
print('-' * 80)
print(f'{f"Precision (k={precision_k})":<35} {results["precision"]:>15.3f} {"0.785":>15} {"higher=better":>12}')
print(f'{f"Recall (k={precision_k})":<35} {results["recall"]:>15.3f} {"0.575":>15} {"higher=better":>12}')
print('=' * 80)
print()
print('Done.')
'''


async def run_eval_sandbox(
    n_samples: int = 200,
    savgol_window: int = 21,
    precision_k: int = 3
):
    """Run evaluation in a CPU Sandbox with real-time stdout streaming."""
    import modal

    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT,
        str(n_samples), str(savgol_window), str(precision_k),
        app=app,
        image=image,
        gpu='T4',  # Use GPU for FID autoencoder training
        volumes={'/data': volume},
        timeout=3600,
    )

    for line in sb.stdout:
        print(line, end='', flush=True)

    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Minimum Jerk Model Evaluation (Quinn & Zhai 2018)')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples for evaluation')
    parser.add_argument('--savgol-window', type=int, default=21, help='Savitzky-Golay filter window size')
    parser.add_argument('--precision-k', type=int, default=3, help='k for precision/recall k-NN')
    args = parser.parse_args()

    async with app.run():
        print(f'Running minimum jerk evaluation (learned distributions)...')
        print(f'  n_samples={args.n_samples}')
        print(f'  savgol_window={args.savgol_window}, precision_k={args.precision_k}')
        print()
        returncode = await run_eval_sandbox(
            n_samples=args.n_samples,
            savgol_window=args.savgol_window,
            precision_k=args.precision_k
        )
        print(f'\nSandbox exited with code: {returncode}')


if __name__ == '__main__':
    asyncio.run(main())
