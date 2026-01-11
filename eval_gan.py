#!/usr/bin/env python3
"""
Unified Evaluation for WordGesture-GAN and Minimum Jerk Baseline

Usage:
    python eval.py --model gan              # Evaluate GAN (default)
    python eval.py --model min-jerk         # Evaluate minimum jerk baseline
    python eval.py --model both             # Evaluate both and compare
    python eval.py --model gan --wandb      # Log GAN results to W&B
"""

import os
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('wordgesture-eval')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'wandb', 'pillow', 'matplotlib', 'fastdtw', 'joblib', 'wordfreq')
    .add_local_python_source('src')
)

# WandB API key injected via Modal Secret
wandb_secret = modal.Secret.from_name('wandb-secret')


# ============================================================================
# Evaluation Script (embedded for Modal Sandbox)
# ============================================================================

EVAL_SCRIPT = '''
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.shared.config import ModelConfig, TrainingConfig, ModalConfig, EvaluationConfig
from src.shared.keyboard import QWERTYKeyboard, MinimumJerkModel
from src.shared.data import load_dataset_from_zip, create_train_test_split
from src.gan.models import Generator
from src.gan.evaluation import evaluate_all_metrics
from src.gan.visualization import create_comparison_figure, create_overlay_figure

def log(msg):
    print(msg, flush=True)

# Parse args
model_type = sys.argv[1] if len(sys.argv) > 1 else 'gan'  # gan, min-jerk, or both
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
truncation = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
savgol_window = int(sys.argv[4]) if len(sys.argv) > 4 else 21
precision_k = int(sys.argv[5]) if len(sys.argv) > 5 else 3
use_wandb = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
fast_mode = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False

device = 'cuda'
config = ModalConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig(
    n_samples=n_samples,
    truncation=truncation,
    savgol_window=savgol_window,
    precision_recall_k=precision_k
)

log(f'GPU: {torch.cuda.get_device_name(0)}')
log(f'Model: {model_type}, Samples: {n_samples}, Truncation: {truncation}')
log(f'Savgol window: {savgol_window}, Precision k: {precision_k}, Fast: {fast_mode}')
log('')

# Load data
log('[1/5] Loading data...')
keyboard = QWERTYKeyboard()
gestures, protos = load_dataset_from_zip(config.data_path, keyboard, model_config, training_config)
train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=config.random_seed)
log(f'  Train: {len(train_ds)}, Test: {len(test_ds)}')

# Get training data for FID autoencoder
log('[2/5] Preparing training data for FID...')
train_g = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])

# Prepare test samples
n = min(n_samples, len(test_ds))
real_g = []
words = []
for i in range(n):
    item = test_ds[i]
    real_g.append(item['gesture'].numpy())
    words.append(item['word'])
real_g = np.array(real_g)

# Results storage
gan_results = None
minjerk_results = None
gan_fake_g = None
minjerk_fake_g = None
cached_real = None  # For sharing computation between GAN and min jerk

# GAN evaluation
if model_type in ['gan', 'both']:
    log('[3/5] Evaluating GAN...')
    checkpoint_path = Path(f'{config.checkpoint_dir}/latest.pt')
    if not checkpoint_path.exists():
        log(f'  ERROR: No checkpoint found at {checkpoint_path}')
        if model_type == 'gan':
            sys.exit(1)
        log('  Skipping GAN evaluation.')
    else:
        generator = Generator(model_config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        generator.eval()
        epoch = ckpt['epoch'] + 1
        wandb_run_id = ckpt.get('wandb_run_id')
        log(f'  Loaded checkpoint from epoch {epoch}')

        # Generate samples (batched for speed)
        with torch.no_grad():
            protos = torch.stack([test_ds[i]['prototype'] for i in range(n)]).to(device)
            z = torch.randn(n, model_config.latent_dim, device=device) * truncation
            gan_fake_g = generator(protos, z).cpu().numpy()
        log(f'    Generated {n} samples (batched)')

        log('  Computing GAN metrics...')
        gan_results = evaluate_all_metrics(real_g, gan_fake_g, train_g, model_config, eval_config, device, skip_dtw=fast_mode)
        cached_real = gan_results.pop('_cached_real', None)  # Save for min jerk

        # W&B logging
        if use_wandb:
            import wandb
            if wandb_run_id:
                wandb.init(project=config.wandb_project, id=wandb_run_id, resume='must')
                log(f'  Resumed W&B run: {wandb.run.name}')
            else:
                wandb.init(
                    project=config.wandb_project,
                    name=f'eval_standalone_epoch{epoch}',
                    config={
                        'model': asdict(model_config),
                        'eval': asdict(eval_config),
                        'checkpoint_epoch': epoch,
                    },
                )
                log(f'  Created standalone W&B run: {wandb.run.name}')

            # Log metrics
            wandb.summary['eval/l2_wasserstein'] = gan_results['l2_wasserstein']
            wandb.summary['eval/dtw_wasserstein'] = gan_results['dtw_wasserstein']
            wandb.summary['eval/fid'] = gan_results['fid']
            wandb.summary['eval/precision'] = gan_results['precision']
            wandb.summary['eval/recall'] = gan_results['recall']
            wandb.summary['eval/velocity_corr'] = gan_results['velocity_corr']
            wandb.summary['eval/acceleration_corr'] = gan_results['acceleration_corr']
            wandb.summary['eval/speed_profile_corr'] = gan_results['speed_profile_corr']
            wandb.summary['eval/time_delta_corr'] = gan_results['time_delta_corr']
            wandb.summary['eval/jerk_fake'] = gan_results['jerk_fake']
            wandb.summary['eval/jerk_real'] = gan_results['jerk_real']
            wandb.summary['eval/ae_reconstruction_loss'] = gan_results['ae_reconstruction_loss']
            wandb.summary['eval/ae_test_loss'] = gan_results['ae_test_loss']
            wandb.summary['eval/epoch'] = epoch

            # Log visualizations
            comparison_fig = create_comparison_figure(real_g[:6], gan_fake_g[:6], words[:6])
            wandb.log({'gestures/comparison': wandb.Image(comparison_fig)})
            plt.close(comparison_fig)

            overlay_fig = create_overlay_figure(real_g[:5], gan_fake_g[:5], words[0] if words else 'sample')
            wandb.log({'gestures/overlay': wandb.Image(overlay_fig)})
            plt.close(overlay_fig)

            wandb.finish()

# Min jerk evaluation
if model_type in ['min-jerk', 'both']:
    log('[4/5] Evaluating Minimum Jerk...')

    # Fit model to training data
    train_gestures_by_word = defaultdict(list)
    for i in range(len(train_ds)):
        item = train_ds[i]
        train_gestures_by_word[item['word']].append(item['gesture'].numpy())

    min_jerk_model = MinimumJerkModel(keyboard)
    min_jerk_model.fit(dict(train_gestures_by_word), verbose=True)

    # Generate samples (parallelized)
    from joblib import Parallel, delayed
    def gen_minjerk(word):
        return min_jerk_model.generate_trajectory(
            word, num_points=model_config.seq_length, include_midpoints=True
        )
    test_words = [test_ds[i]['word'] for i in range(n)]
    minjerk_fake_g = np.array(Parallel(n_jobs=-1)(delayed(gen_minjerk)(w) for w in test_words))
    log(f'    Generated {n} samples (parallel)')

    log('  Computing Min Jerk metrics...')
    minjerk_results = evaluate_all_metrics(real_g, minjerk_fake_g, train_g, model_config, eval_config, device, skip_dtw=fast_mode, cached_real=cached_real)
    minjerk_results.pop('_cached_real', None)  # Don't need to save again

log('[5/5] Done computing metrics.')
log('')

# Print results
def print_results_table(results, model_name, paper_values):
    """Print results table for a single model."""
    log('=' * 75)
    log(f'{model_name} Results')
    log('=' * 75)
    log(f'{"Metric":<30} {"Ours":>15} {"Paper":>15} {"Notes":>12}')
    log('-' * 75)
    log(f'{"L2 Wasserstein (x,y)":<30} {results["l2_wasserstein"]:>15.3f} {paper_values["l2"]:>15} {"lower=better":>12}')
    dtw_val = "SKIPPED" if results["dtw_wasserstein"] < 0 else f'{results["dtw_wasserstein"]:.3f}'
    log(f'{"DTW Wasserstein (x,y)":<30} {dtw_val:>15} {paper_values["dtw"]:>15} {"lower=better":>12}')
    log(f'{"Jerk (generated)":<30} {results["jerk_fake"]:>15.5f} {paper_values["jerk_fake"]:>15} {"~real":>12}')
    log(f'{"Jerk (real)":<30} {results["jerk_real"]:>15.5f} {paper_values["jerk_real"]:>15} {"reference":>12}')
    log(f'{"Velocity Corr":<30} {results["velocity_corr"]:>15.3f} {paper_values["vel"]:>15} {"higher=better":>12}')
    log(f'{"Acceleration Corr":<30} {results["acceleration_corr"]:>15.3f} {paper_values["acc"]:>15} {"higher=better":>12}')
    log(f'{"Speed Profile Corr":<30} {results["speed_profile_corr"]:>15.3f} {"--":>15} {"higher=better":>12}')
    log(f'{"Time Delta Corr":<30} {results["time_delta_corr"]:>15.3f} {"--":>15} {"higher=better":>12}')
    log('-' * 75)
    log(f'{"AE Reconstruction (L1)":<30} {results["ae_reconstruction_loss"]:>15.4f} {"0.041":>15} {"lower=better":>12}')
    log(f'{"AE Test Loss (L1)":<30} {results["ae_test_loss"]:>15.4f} {"0.046":>15} {"lower=better":>12}')
    log(f'{"FID":<30} {results["fid"]:>15.4f} {paper_values["fid"]:>15} {"lower=better":>12}')
    log('-' * 75)
    log(f'{f"Precision (k={precision_k})":<30} {results["precision"]:>15.3f} {paper_values["precision"]:>15} {"higher=better":>12}')
    log(f'{f"Recall (k={precision_k})":<30} {results["recall"]:>15.3f} {paper_values["recall"]:>15} {"higher=better":>12}')
    log('=' * 75)

def print_comparison_table(gan_results, minjerk_results):
    """Print side-by-side comparison of GAN and Min Jerk."""
    log('=' * 90)
    log('Side-by-Side Comparison: GAN vs Minimum Jerk')
    log('=' * 90)
    log(f'{"Metric":<30} {"GAN":>15} {"Min Jerk":>15} {"Paper GAN":>12} {"Paper MJ":>12}')
    log('-' * 90)
    log(f'{"L2 Wasserstein (x,y)":<30} {gan_results["l2_wasserstein"]:>15.3f} {minjerk_results["l2_wasserstein"]:>15.3f} {"4.409":>12} {"5.004":>12}')
    gan_dtw = "SKIP" if gan_results["dtw_wasserstein"] < 0 else f'{gan_results["dtw_wasserstein"]:.3f}'
    mj_dtw = "SKIP" if minjerk_results["dtw_wasserstein"] < 0 else f'{minjerk_results["dtw_wasserstein"]:.3f}'
    log(f'{"DTW Wasserstein (x,y)":<30} {gan_dtw:>15} {mj_dtw:>15} {"2.146":>12} {"2.752":>12}')
    log(f'{"Jerk (generated)":<30} {gan_results["jerk_fake"]:>15.5f} {minjerk_results["jerk_fake"]:>15.5f} {"0.0058":>12} {"0.0034":>12}')
    log(f'{"Velocity Corr":<30} {gan_results["velocity_corr"]:>15.3f} {minjerk_results["velocity_corr"]:>15.3f} {"0.40":>12} {"0.40":>12}')
    log(f'{"Acceleration Corr":<30} {gan_results["acceleration_corr"]:>15.3f} {minjerk_results["acceleration_corr"]:>15.3f} {"0.26":>12} {"0.21":>12}')
    log(f'{"Speed Profile Corr":<30} {gan_results["speed_profile_corr"]:>15.3f} {minjerk_results["speed_profile_corr"]:>15.3f} {"--":>12} {"--":>12}')
    log(f'{"Time Delta Corr":<30} {gan_results["time_delta_corr"]:>15.3f} {minjerk_results["time_delta_corr"]:>15.3f} {"--":>12} {"--":>12}')
    log('-' * 90)
    log(f'{"FID":<30} {gan_results["fid"]:>15.4f} {minjerk_results["fid"]:>15.4f} {"0.270":>12} {"0.354":>12}')
    log(f'{f"Precision (k={precision_k})":<30} {gan_results["precision"]:>15.3f} {minjerk_results["precision"]:>15.3f} {"0.973":>12} {"0.785":>12}')
    log(f'{f"Recall (k={precision_k})":<30} {gan_results["recall"]:>15.3f} {minjerk_results["recall"]:>15.3f} {"0.258":>12} {"0.575":>12}')
    log('=' * 90)

# Paper values for each model
gan_paper = {
    'l2': '4.409', 'dtw': '2.146', 'jerk_fake': '0.0058', 'jerk_real': '0.0066',
    'vel': '0.40', 'acc': '0.26', 'fid': '0.270', 'precision': '0.973', 'recall': '0.258'
}
minjerk_paper = {
    'l2': '5.004', 'dtw': '2.752', 'jerk_fake': '0.0034', 'jerk_real': '0.0066',
    'vel': '0.40', 'acc': '0.21', 'fid': '0.354', 'precision': '0.785', 'recall': '0.575'
}

if model_type == 'both' and gan_results and minjerk_results:
    print_comparison_table(gan_results, minjerk_results)
elif gan_results:
    print_results_table(gan_results, 'GAN', gan_paper)
elif minjerk_results:
    print_results_table(minjerk_results, 'Minimum Jerk', minjerk_paper)

log('')
log('Done.')
'''


async def run_eval_sandbox(
    model_type: str = 'gan',
    n_samples: int = 200,
    truncation: float = 1.0,
    savgol_window: int = 21,
    precision_k: int = 3,
    use_wandb: bool = False,
    fast: bool = False
):
    """Run evaluation in a Sandbox with real-time stdout streaming."""
    secrets = [wandb_secret] if use_wandb else []

    sb = modal.Sandbox.create(
        "python", "-c", EVAL_SCRIPT,
        model_type, str(n_samples), str(truncation), str(savgol_window), str(precision_k), str(int(use_wandb)), str(int(fast)),
        app=app,
        image=image,
        gpu='T4',
        volumes={'/data': volume},
        secrets=secrets,
        timeout=7200,
    )

    for line in sb.stdout:
        print(line, end='', flush=True)

    for line in sb.stderr:
        print(f"STDERR: {line}", end='', flush=True)

    sb.wait()
    return sb.returncode


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified evaluation for WordGesture-GAN and Minimum Jerk')
    parser.add_argument('--model', type=str, default='gan', choices=['gan', 'min-jerk', 'both'],
                        help='Model to evaluate (default: gan)')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples for evaluation')
    parser.add_argument('--truncation', type=float, default=1.0, help='Truncation for latent sampling (GAN only)')
    parser.add_argument('--savgol-window', type=int, default=21, help='Savitzky-Golay filter window size')
    parser.add_argument('--precision-k', type=int, default=3, help='k for precision/recall k-NN')
    parser.add_argument('--wandb', action='store_true', help='Log results to W&B (GAN only)')
    parser.add_argument('--fast', action='store_true', help='Skip DTW (expensive O(n²) metric)')
    args = parser.parse_args()

    async with app.run():
        print(f'Running evaluation (model={args.model})...')
        print(f'  n_samples={args.n_samples}, truncation={args.truncation}')
        print(f'  savgol_window={args.savgol_window}, precision_k={args.precision_k}')
        if args.fast:
            print('  Fast mode: skipping DTW')
        if args.wandb:
            print('  W&B logging: enabled')
        print()
        returncode = await run_eval_sandbox(
            model_type=args.model,
            n_samples=args.n_samples,
            truncation=args.truncation,
            savgol_window=args.savgol_window,
            precision_k=args.precision_k,
            use_wandb=args.wandb,
            fast=args.fast
        )
        print(f'\nSandbox exited with code: {returncode}')


if __name__ == '__main__':
    asyncio.run(main())
