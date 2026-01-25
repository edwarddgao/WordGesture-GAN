"""Evaluate WordGesture-GAN and Minimum Jerk models."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from wordgesture_gan.data.preprocess import load_dataset
from wordgesture_gan.keyboard.qwerty import KeyboardLayout
from wordgesture_gan.metrics.distance import l2_distance, per_word_wasserstein
from wordgesture_gan.metrics.dynamics import dynamics_correlation, derivatives, jerk_stat
from wordgesture_gan.metrics.duration import clc_predict, fit_clc, gesture_duration, word_durations
from wordgesture_gan.metrics.fid import compute_fid, train_autoencoder
from wordgesture_gan.metrics.precision_recall import precision_recall
from wordgesture_gan.minjerk.fit import fit_distributions
from wordgesture_gan.minjerk.sample import sample_minimum_jerk
from wordgesture_gan.models.wg_gan import Generator
from wordgesture_gan.prototypes import build_word_prototype
from wordgesture_gan.utils import get_device


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _match_pairs(
    real: np.ndarray,
    real_words: List[str],
    fake: np.ndarray,
    fake_words: List[str],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    vocab = sorted(set(real_words))
    for word in vocab:
        real_list = [g for g, w in zip(real, real_words) if w == word]
        fake_list = [g for g, w in zip(fake, fake_words) if w == word]
        if not real_list or not fake_list:
            continue
        cost = np.zeros((len(real_list), len(fake_list)), dtype=np.float32)
        for i, r in enumerate(real_list):
            for j, f in enumerate(fake_list):
                cost[i, j] = l2_distance(r[:, :2], f[:, :2])
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            pairs.append((real_list[i], fake_list[j]))
    return pairs


def _velocity_accel_stats(gestures: np.ndarray) -> Dict[str, float]:
    velocities = []
    accelerations = []
    for gesture in gestures:
        vel, acc, _ = derivatives(gesture)
        velocities.append(np.linalg.norm(vel, axis=1))
        accelerations.append(np.linalg.norm(acc, axis=1))
    velocities = np.concatenate(velocities)
    accelerations = np.concatenate(accelerations)
    return {
        "velocity_mean": float(np.mean(velocities)),
        "velocity_std": float(np.std(velocities)),
        "accel_mean": float(np.mean(accelerations)),
        "accel_std": float(np.std(accelerations)),
    }


def _dt_duration_stats(gestures: np.ndarray) -> Dict[str, float]:
    # Exclude the first dt (always 0 in our preprocessing) for dt mean/std.
    if gestures.ndim != 3 or gestures.shape[-1] < 3:
        return {"dt_mean": float("nan"), "dt_std": float("nan"), "duration_mean": float("nan"), "duration_std": float("nan")}
    dt = gestures[:, 1:, 2].reshape(-1) if gestures.shape[1] > 1 else gestures[:, :, 2].reshape(-1)
    durations = np.sum(gestures[:, :, 2], axis=1)
    return {
        "dt_mean": float(np.mean(dt)) if dt.size else float("nan"),
        "dt_std": float(np.std(dt)) if dt.size else float("nan"),
        "duration_mean": float(np.mean(durations)) if durations.size else float("nan"),
        "duration_std": float(np.std(durations)) if durations.size else float("nan"),
    }


def _duration_rmse(real: np.ndarray, real_words: List[str], pred: np.ndarray, pred_words: List[str]) -> float:
    real_map = {w: np.mean(v) for w, v in word_durations(real, real_words).items()}
    pred_map = {w: np.mean(v) for w, v in word_durations(pred, pred_words).items()}
    shared = sorted(set(real_map.keys()) & set(pred_map.keys()))
    if not shared:
        return float("nan")
    diffs = [(pred_map[w] - real_map[w]) ** 2 for w in shared]
    return float(np.sqrt(np.mean(diffs)))


def evaluate_model(
    label: str,
    real: np.ndarray,
    real_words: List[str],
    fake: np.ndarray,
    fake_words: List[str],
    autoencoder,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")
    print(f"  Real samples: {len(real)}, Fake samples: {len(fake)}")

    t0 = time.time()
    print("  [1/6] Computing Wasserstein L2 distance...")
    l2_mean, l2_std = per_word_wasserstein(real[:, :, :2], real_words, fake[:, :, :2], fake_words, metric="l2")
    print(f"         Done in {time.time() - t0:.1f}s -> L2: {l2_mean:.4f} ± {l2_std:.4f}")

    t0 = time.time()
    print("  [2/6] Computing Wasserstein DTW distance...")
    dtw_mean, dtw_std = per_word_wasserstein(real[:, :, :2], real_words, fake[:, :, :2], fake_words, metric="dtw", band_width=15)
    print(f"         Done in {time.time() - t0:.1f}s -> DTW: {dtw_mean:.4f} ± {dtw_std:.4f}")

    t0 = time.time()
    print("  [3/6] Computing FID...")
    fid = compute_fid(real, fake, autoencoder)
    print(f"         Done in {time.time() - t0:.1f}s -> FID: {fid:.4f}")

    t0 = time.time()
    print("  [4/6] Computing Precision/Recall...")
    precision, recall = precision_recall(real, fake, k=3)
    print(f"         Done in {time.time() - t0:.1f}s -> P: {precision:.4f}, R: {recall:.4f}")

    t0 = time.time()
    print("  [5/6] Computing dynamics correlations...")
    pairs = _match_pairs(real, real_words, fake, fake_words)
    v_corrs = []
    a_corrs = []
    for r, f in tqdm(pairs, desc="         Dynamics", leave=False):
        v_corr, a_corr = dynamics_correlation(r, f)
        v_corrs.append(v_corr)
        a_corrs.append(a_corr)
    print(f"         Done in {time.time() - t0:.1f}s -> vel_corr: {np.mean(v_corrs):.4f}")

    t0 = time.time()
    print("  [6/6] Computing jerk statistics...")
    real_jerk = [jerk_stat(g) for g in tqdm(real, desc="         Real jerk", leave=False)]
    fake_jerk = [jerk_stat(g) for g in tqdm(fake, desc="         Fake jerk", leave=False)]
    print(f"         Done in {time.time() - t0:.1f}s")

    print(f"  {label} evaluation complete.\n")

    return {
        "label": label,
        "wasserstein_l2_mean": l2_mean,
        "wasserstein_l2_std": l2_std,
        "wasserstein_dtw_mean": dtw_mean,
        "wasserstein_dtw_std": dtw_std,
        "fid": fid,
        "precision": precision,
        "recall": recall,
        "velocity_corr_mean": float(np.mean(v_corrs)) if v_corrs else float("nan"),
        "velocity_corr_std": float(np.std(v_corrs)) if v_corrs else float("nan"),
        "accel_corr_mean": float(np.mean(a_corrs)) if a_corrs else float("nan"),
        "accel_corr_std": float(np.std(a_corrs)) if a_corrs else float("nan"),
        "jerk_real_mean": float(np.mean(real_jerk)),
        "jerk_real_std": float(np.std(real_jerk)),
        "jerk_fake_mean": float(np.mean(fake_jerk)),
        "jerk_fake_std": float(np.std(fake_jerk)),
        "velocity_stats_fake": _velocity_accel_stats(fake),
        "velocity_stats_real": _velocity_accel_stats(real),
        "dt_stats_fake": _dt_duration_stats(fake),
        "dt_stats_real": _dt_duration_stats(real),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WordGesture-GAN + Minimum Jerk.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--minjerk_config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--dt_calibration",
        type=str,
        default="none",
        choices=["none", "train_mean"],
        help="Optionally rescale WG-GAN dt to match training-set mean dt (helps dynamics/jerk/FID when timing is off).",
    )
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 60)
    print("WordGesture-GAN & Minimum Jerk Evaluation")
    print("=" * 60)

    cfg = load_config(args.config)
    minjerk_cfg = load_config(args.minjerk_config)
    layout = KeyboardLayout()
    device = get_device()
    print(f"Using device: {device}")

    print("\n[Step 1/7] Loading datasets...")
    t0 = time.time()
    train_gestures, train_words = load_dataset(args.data_dir / "train.npz")
    test_gestures, test_words = load_dataset(args.data_dir / "test.npz")
    print(f"  Train: {len(train_gestures)} samples, Test: {len(test_gestures)} samples")
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[Step 2/7] Loading WordGesture-GAN checkpoint...")
    t0 = time.time()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = Generator(
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dt_scale=float(cfg["model"].get("dt_scale", 0.05)),
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    print(f"  Done in {time.time() - t0:.1f}s")

    word_counts = Counter(test_words)
    print(f"\n[Step 3/7] Generating WordGesture-GAN samples ({len(test_gestures)} total)...")
    t0 = time.time()
    wg_gan_samples = []
    wg_gan_words: List[str] = []
    for word, count in tqdm(word_counts.items(), desc="  WG-GAN generation"):
        prototype = build_word_prototype(word, int(cfg["data"]["n_points"]), layout)
        proto_batch = torch.from_numpy(np.repeat(prototype[None, ...], count, axis=0)).to(device)
        z = torch.randn(count, int(cfg["model"]["latent_dim"]), device=device)
        with torch.no_grad():
            fake = generator(proto_batch, z).cpu().numpy()
        wg_gan_samples.append(fake)
        wg_gan_words.extend([word] * count)
    wg_gan_samples = np.concatenate(wg_gan_samples, axis=0)
    print(f"  Generated {len(wg_gan_samples)} samples in {time.time() - t0:.1f}s")

    if args.dt_calibration == "train_mean":
        train_dt_mean = _dt_duration_stats(train_gestures)["dt_mean"]
        fake_dt_mean = _dt_duration_stats(wg_gan_samples)["dt_mean"]
        if np.isfinite(train_dt_mean) and np.isfinite(fake_dt_mean) and fake_dt_mean > 1e-8:
            scale = float(train_dt_mean / fake_dt_mean)
            wg_gan_samples[:, :, 2] *= scale
            print(f"\n[Info] Applied WG-GAN dt calibration: scale={scale:.4f} (train_dt_mean={train_dt_mean:.6f}, fake_dt_mean={fake_dt_mean:.6f})")
        else:
            print("\n[Info] Skipped WG-GAN dt calibration (non-finite dt stats).")

    print("\n[Step 4/7] Fitting Minimum Jerk distributions...")
    t0 = time.time()
    fit = fit_distributions(train_gestures, train_words, layout)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n[Step 5/7] Generating Minimum Jerk samples ({len(test_gestures)} total)...")
    t0 = time.time()
    minjerk_samples = []
    minjerk_words: List[str] = []
    total_samples = sum(word_counts.values())
    with tqdm(total=total_samples, desc="  MinJerk generation") as pbar:
        for word, count in word_counts.items():
            for _ in range(count):
                minjerk_samples.append(
                    sample_minimum_jerk(word, fit, int(cfg["data"]["n_points"]), layout)
                )
                minjerk_words.append(word)
                pbar.update(1)
    minjerk_samples = np.stack(minjerk_samples, axis=0)
    print(f"  Generated {len(minjerk_samples)} samples in {time.time() - t0:.1f}s")

    print("\n[Step 6/7] Training FID autoencoder...")
    t0 = time.time()
    autoencoder = train_autoencoder(
        train_gestures,
        latent_dim=minjerk_cfg["fid"]["latent_dim"],
        epochs=minjerk_cfg["fid"]["epochs"],
        batch_size=minjerk_cfg["fid"]["batch_size"],
        lr=minjerk_cfg["fid"]["lr"],
        device=device,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[Step 7/8] Creating real baseline (test split)...")
    t0 = time.time()
    # Split test set by word into two halves for real-to-real comparison
    test_by_word: Dict[str, List[np.ndarray]] = {}
    for gesture, word in zip(test_gestures, test_words):
        test_by_word.setdefault(word, []).append(gesture)
    real_half_a = []
    real_half_a_words: List[str] = []
    real_half_b = []
    real_half_b_words: List[str] = []
    for word, gestures_list in test_by_word.items():
        np.random.shuffle(gestures_list)
        if len(gestures_list) == 1:
            # Randomly assign single-sample words to either half
            if np.random.random() < 0.5:
                real_half_a.extend(gestures_list)
                real_half_a_words.append(word)
            else:
                real_half_b.extend(gestures_list)
                real_half_b_words.append(word)
        else:
            mid = len(gestures_list) // 2
            if mid == 0:
                mid = 1
            real_half_a.extend(gestures_list[:mid])
            real_half_a_words.extend([word] * len(gestures_list[:mid]))
            real_half_b.extend(gestures_list[mid:])
            real_half_b_words.extend([word] * len(gestures_list[mid:]))

    real_metrics = None
    can_compute_real_baseline = bool(real_half_a) and bool(real_half_b)
    if not can_compute_real_baseline:
        print("  [Warning] Cannot create real baseline: insufficient samples for split.")
        print("            Skipping real-to-real comparison.")
    else:
        real_half_a = np.stack(real_half_a, axis=0)
        real_half_b = np.stack(real_half_b, axis=0)
        print(f"  Split test into {len(real_half_a)} + {len(real_half_b)} samples in {time.time() - t0:.1f}s")

    print("\n[Step 8/8] Computing evaluation metrics...")
    if can_compute_real_baseline:
        real_metrics = evaluate_model("Real (test split)", real_half_a, real_half_a_words, real_half_b, real_half_b_words, autoencoder)
    wg_metrics = evaluate_model("WordGesture-GAN", test_gestures, test_words, wg_gan_samples, wg_gan_words, autoencoder)
    mj_metrics = evaluate_model("MinimumJerk", test_gestures, test_words, minjerk_samples, minjerk_words, autoencoder)

    # Duration metrics
    print("\nComputing duration metrics...")
    t0 = time.time()
    m, n = fit_clc(train_gestures, train_words, layout)
    clc_preds = [clc_predict(word, m, n, layout) for word in word_counts.keys()]
    real_means = {w: np.mean(v) for w, v in word_durations(test_gestures, test_words).items()}
    clc_rmse = float(
        np.sqrt(np.mean([(clc_preds[idx] - real_means[word]) ** 2 for idx, word in enumerate(word_counts.keys())]))
    )
    wg_rmse = _duration_rmse(test_gestures, test_words, wg_gan_samples, wg_gan_words)
    print(f"  Done in {time.time() - t0:.1f}s")

    results = {
        "real_baseline": real_metrics,
        "wordgesture_gan": wg_metrics,
        "minimum_jerk": mj_metrics,
        "duration": {
            "clc_rmse": clc_rmse,
            "wg_gan_rmse": wg_rmse,
            "clc_params": {"m": m, "n": n},
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nResults saved to: {out_path}")
    print("\n--- Summary ---")
    if real_metrics is not None:
        print(f"Real (baseline): L2={real_metrics['wasserstein_l2_mean']:.4f}, DTW={real_metrics['wasserstein_dtw_mean']:.4f}, FID={real_metrics['fid']:.4f}")
    else:
        print("Real (baseline): skipped (insufficient samples for split)")
    print(f"WordGesture-GAN: L2={wg_metrics['wasserstein_l2_mean']:.4f}, DTW={wg_metrics['wasserstein_dtw_mean']:.4f}, FID={wg_metrics['fid']:.4f}")
    print(f"Minimum Jerk:    L2={mj_metrics['wasserstein_l2_mean']:.4f}, DTW={mj_metrics['wasserstein_dtw_mean']:.4f}, FID={mj_metrics['fid']:.4f}")
    print(f"Duration RMSE:   CLC={clc_rmse:.4f}, WG-GAN={wg_rmse:.4f}")


if __name__ == "__main__":
    main()