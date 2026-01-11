"""
Evaluation metrics for WordGesture-GAN.

Implements metrics from the paper:
- L2 and DTW Wasserstein distance
- Frechet Inception Distance (FID)
- Precision and Recall
- Velocity and Acceleration correlations
- Jerk analysis

Also implements time-aware dynamics metrics that properly use the time channel:
- Time-aware velocity correlation (true d(position)/d(time))
- Time-aware acceleration correlation (true d²(position)/d(time)²)
- Speed profile correlation (velocity magnitude over time)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from .models import AutoEncoder
from .config import ModelConfig, EvaluationConfig, DEFAULT_MODEL_CONFIG, DEFAULT_EVALUATION_CONFIG


def scipy_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using eigendecomposition."""
    from scipy.linalg import sqrtm
    return sqrtm(matrix).real


# =============================================================================
# Time-Aware Dynamics Computation
# =============================================================================
#
# For arc-length uniformly sampled data with non-uniform time:
# - Standard diff/savgol compute d/di = d/ds (spatial derivatives)
# - We need d/dt (temporal derivatives) for true dynamics
#
# The key insight: velocity = d(position)/d(time), not d(position)/d(index)
# =============================================================================

def compute_time_aware_velocity(gestures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity as d(position)/d(time) for non-uniform time sampling.

    For arc-length uniform data, this recovers the true velocity by dividing
    spatial displacement by actual time elapsed.

    Args:
        gestures: (n, seq_len, 3) array with (x, y, t)

    Returns:
        velocity: (n, seq_len-1, 2) array with (vx, vy) at midpoints
        t_mid: (n, seq_len-1) array with midpoint times
    """
    # Position and time
    xy = gestures[:, :, :2]  # (n, seq_len, 2)
    t = gestures[:, :, 2]    # (n, seq_len)

    # Deltas
    dxy = np.diff(xy, axis=1)  # (n, seq_len-1, 2)
    dt = np.diff(t, axis=1)    # (n, seq_len-1)

    # Midpoint times (velocity is defined at midpoints between samples)
    t_mid = (t[:, :-1] + t[:, 1:]) / 2  # (n, seq_len-1)

    # Avoid division by zero - use small epsilon
    dt_safe = np.where(np.abs(dt) > 1e-10, dt, 1e-10 * np.sign(dt + 1e-20))

    # Velocity = displacement / time
    velocity = dxy / dt_safe[:, :, np.newaxis]  # (n, seq_len-1, 2)

    return velocity, t_mid


def compute_time_aware_acceleration(gestures: np.ndarray) -> np.ndarray:
    """
    Compute acceleration as d²(position)/d(time)² for non-uniform time sampling.

    Uses the formula:
        a[i] = (v[i] - v[i-1]) / (t_mid[i] - t_mid[i-1])

    where v[i] is velocity at midpoint i, and t_mid[i] is the time at that midpoint.

    Args:
        gestures: (n, seq_len, 3) array with (x, y, t)

    Returns:
        acceleration: (n, seq_len-2, 2) array with (ax, ay)
    """
    velocity, t_mid = compute_time_aware_velocity(gestures)

    # Velocity deltas
    dv = np.diff(velocity, axis=1)  # (n, n_vel-1, 2)

    # Time deltas between consecutive midpoints
    dt_mid = np.diff(t_mid, axis=1)  # (n, n_vel-1)

    # Avoid division by zero
    dt_mid_safe = np.where(np.abs(dt_mid) > 1e-10, dt_mid, 1e-10 * np.sign(dt_mid + 1e-20))

    # Acceleration = d(velocity) / d(time)
    acceleration = dv / dt_mid_safe[:, :, np.newaxis]  # (n, n_vel-1, 2)

    return acceleration


def compute_time_aware_jerk(gestures: np.ndarray) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) with respect to time.

    Jerk = d³(position)/d(time)³

    Args:
        gestures: (n, seq_len, 3) array with (x, y, t)

    Returns:
        jerk_magnitude: (n,) array with mean jerk magnitude per gesture
    """
    velocity, t_mid = compute_time_aware_velocity(gestures)
    acceleration = compute_time_aware_acceleration(gestures)

    # Time at acceleration points (midpoints of velocity midpoints)
    t_acc = (t_mid[:, :-1] + t_mid[:, 1:]) / 2  # (n, seq_len-2)

    # Acceleration deltas
    da = np.diff(acceleration, axis=1)  # (n, seq_len-3, 2)

    # Time deltas between acceleration points
    dt_acc = np.diff(t_acc, axis=1)  # (n, seq_len-3)

    # Avoid division by zero
    dt_acc_safe = np.where(np.abs(dt_acc) > 1e-10, dt_acc, 1e-10)

    # Jerk = d(acceleration) / d(time)
    jerk = da / dt_acc_safe[:, :, np.newaxis]  # (n, seq_len-3, 2)

    # Return mean jerk magnitude per gesture
    jerk_magnitude = np.mean(np.linalg.norm(jerk, axis=-1), axis=1)  # (n,)

    return jerk_magnitude


def time_aware_velocity_correlation(real_gestures: np.ndarray, fake_gestures: np.ndarray) -> float:
    """
    Compute correlation of time-aware velocity vectors.

    This measures how well the fake gestures match the true velocity patterns
    (both direction and magnitude) of real gestures.

    Args:
        real_gestures: (n, seq_len, 3) array
        fake_gestures: (n, seq_len, 3) array

    Returns:
        Mean correlation across all gesture pairs
    """
    v_real, _ = compute_time_aware_velocity(real_gestures)
    v_fake, _ = compute_time_aware_velocity(fake_gestures)

    correlations = []
    for i in range(len(v_real)):
        vr = v_real[i].flatten()
        vf = v_fake[i].flatten()

        # Need sufficient variance for meaningful correlation
        if len(vr) > 1 and np.std(vr) > 1e-10 and np.std(vf) > 1e-10:
            # Clip extreme values to reduce noise sensitivity
            vr_clipped = np.clip(vr, np.percentile(vr, 1), np.percentile(vr, 99))
            vf_clipped = np.clip(vf, np.percentile(vf, 1), np.percentile(vf, 99))

            corr = np.corrcoef(vr_clipped, vf_clipped)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def time_aware_acceleration_correlation(real_gestures: np.ndarray, fake_gestures: np.ndarray) -> float:
    """
    Compute correlation of time-aware acceleration vectors.

    Args:
        real_gestures: (n, seq_len, 3) array
        fake_gestures: (n, seq_len, 3) array

    Returns:
        Mean correlation across all gesture pairs
    """
    a_real = compute_time_aware_acceleration(real_gestures)
    a_fake = compute_time_aware_acceleration(fake_gestures)

    correlations = []
    for i in range(len(a_real)):
        ar = a_real[i].flatten()
        af = a_fake[i].flatten()

        if len(ar) > 1 and np.std(ar) > 1e-10 and np.std(af) > 1e-10:
            # Clip extreme values
            ar_clipped = np.clip(ar, np.percentile(ar, 1), np.percentile(ar, 99))
            af_clipped = np.clip(af, np.percentile(af, 1), np.percentile(af, 99))

            corr = np.corrcoef(ar_clipped, af_clipped)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def speed_profile_correlation(real_gestures: np.ndarray, fake_gestures: np.ndarray) -> float:
    """
    Compute correlation of speed (velocity magnitude) profiles.

    This directly measures: does the model speed up and slow down
    at the same positions as real gestures?

    Speed is the magnitude of velocity, so this ignores direction and
    focuses purely on "how fast" at each point.

    Args:
        real_gestures: (n, seq_len, 3) array
        fake_gestures: (n, seq_len, 3) array

    Returns:
        Mean correlation of speed profiles
    """
    v_real, _ = compute_time_aware_velocity(real_gestures)
    v_fake, _ = compute_time_aware_velocity(fake_gestures)

    # Speed = magnitude of velocity
    speed_real = np.linalg.norm(v_real, axis=-1)  # (n, seq_len-1)
    speed_fake = np.linalg.norm(v_fake, axis=-1)

    correlations = []
    for i in range(len(speed_real)):
        sr = speed_real[i]
        sf = speed_fake[i]

        if len(sr) > 1 and np.std(sr) > 1e-10 and np.std(sf) > 1e-10:
            # Clip extreme values (high speed from small dt)
            sr_clipped = np.clip(sr, 0, np.percentile(sr, 99))
            sf_clipped = np.clip(sf, 0, np.percentile(sf, 99))

            corr = np.corrcoef(sr_clipped, sf_clipped)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def time_delta_correlation(real_gestures: np.ndarray, fake_gestures: np.ndarray) -> float:
    """
    Compute correlation of time delta patterns.

    For arc-length uniform data, time deltas directly encode velocity:
    - Large dt = slow movement (more time to cover same spatial distance)
    - Small dt = fast movement

    This is the simplest measure of temporal dynamics agreement.

    Args:
        real_gestures: (n, seq_len, 3) array
        fake_gestures: (n, seq_len, 3) array

    Returns:
        Mean correlation of time delta patterns
    """
    dt_real = np.diff(real_gestures[:, :, 2], axis=1)  # (n, seq_len-1)
    dt_fake = np.diff(fake_gestures[:, :, 2], axis=1)

    correlations = []
    for i in range(len(dt_real)):
        dtr = dt_real[i]
        dtf = dt_fake[i]

        if len(dtr) > 1 and np.std(dtr) > 1e-10 and np.std(dtf) > 1e-10:
            corr = np.corrcoef(dtr, dtf)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def evaluate_all_metrics(
    real_gestures: np.ndarray,
    fake_gestures: np.ndarray,
    train_gestures: Optional[np.ndarray] = None,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    eval_config: EvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Run all evaluation metrics from the paper on flat gesture arrays.

    This is a convenience wrapper that works with numpy arrays directly,
    as used in modal_train.py.

    Args:
        real_gestures: Array of shape (n, seq_len, 3) with real gestures
        fake_gestures: Array of shape (n, seq_len, 3) with fake gestures
        train_gestures: Optional training gestures for FID autoencoder training.
                       If None, uses real_gestures for training.
        model_config: Model configuration
        eval_config: Evaluation configuration
        device: Device for FID computation ('cuda' or 'cpu')

    Returns:
        Dictionary with all metrics:
        - l2_wasserstein: L2 Wasserstein distance (x,y only)
        - dtw_wasserstein: DTW Wasserstein distance (x,y only)
        - jerk_real, jerk_fake: Mean jerk values
        - velocity_corr: Time-aware velocity correlation (d/dt)
        - acceleration_corr: Time-aware acceleration correlation (d²/dt²)
        - speed_profile_corr: Speed magnitude correlation
        - time_delta_corr: Time delta pattern correlation
        - fid: FID score
        - precision, recall: k-NN precision/recall
    """
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    n = len(real_gestures)
    results = {}

    # L2 Wasserstein (using cdist for speed)
    real_flat_xy = real_gestures[:, :, :2].reshape(n, -1)
    fake_flat_xy = fake_gestures[:, :, :2].reshape(n, -1)
    dist_matrix = cdist(real_flat_xy, fake_flat_xy, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    results['l2_wasserstein'] = dist_matrix[row_ind, col_ind].mean()

    # DTW Wasserstein (using fastdtw with parallel computation)
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    from joblib import Parallel, delayed

    def compute_dtw_row(i):
        row = np.zeros(n)
        for j in range(n):
            distance, _ = fastdtw(real_gestures[i, :, :2], fake_gestures[j, :, :2], dist=euclidean)
            row[j] = distance
        return row

    dtw_rows = Parallel(n_jobs=-1, verbose=0)(delayed(compute_dtw_row)(i) for i in range(n))
    dtw_dist = np.array(dtw_rows)
    row_ind2, col_ind2 = linear_sum_assignment(dtw_dist)
    dtw_raw = dtw_dist[row_ind2, col_ind2].mean()
    # Normalize by sqrt(seq_length) to match paper scale
    results['dtw_wasserstein'] = dtw_raw / np.sqrt(model_config.seq_length)

    # Jerk (using Savitzky-Golay filter)
    def compute_gesture_jerk(g):
        x, y = g[:, 0], g[:, 1]
        if len(x) < eval_config.savgol_window:
            return 0.0
        d3x = savgol_filter(x, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=3)
        d3y = savgol_filter(y, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=3)
        return np.mean(np.sqrt(d3x**2 + d3y**2))

    results['jerk_real'] = np.mean([compute_gesture_jerk(g) for g in real_gestures])
    results['jerk_fake'] = np.mean([compute_gesture_jerk(g) for g in fake_gestures])

    # =========================================================================
    # Time-Aware Dynamics Metrics
    # =========================================================================
    # These metrics compute true temporal derivatives (d/dt) to measure
    # actual gesture dynamics, not spatial derivatives (d/ds).

    # Time-aware velocity: v = d(position) / d(time)
    results['velocity_corr'] = time_aware_velocity_correlation(real_gestures, fake_gestures)

    # Time-aware acceleration: a = d²(position) / d(time)²
    results['acceleration_corr'] = time_aware_acceleration_correlation(real_gestures, fake_gestures)

    # Speed profile: correlation of |velocity| (how fast at each point)
    results['speed_profile_corr'] = speed_profile_correlation(real_gestures, fake_gestures)

    # Time delta correlation: direct comparison of timing patterns
    results['time_delta_corr'] = time_delta_correlation(real_gestures, fake_gestures)

    # FID Score (train autoencoder on training data)
    # Paper Section 5.5: "trained an auto-encoder on the training dataset"
    # Target: L1 reconstruction loss of ~0.041 (paper)
    train_data = train_gestures if train_gestures is not None else real_gestures
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    ae_dataset = TensorDataset(train_tensor)
    ae_loader = TorchDataLoader(ae_dataset, batch_size=512, shuffle=True)

    # Custom training loop for FID
    autoencoder = AutoEncoder(model_config, eval_config.fid_hidden_dim).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=eval_config.fid_autoencoder_lr)
    ae_criterion = torch.nn.L1Loss()

    autoencoder.train()
    final_loss = 0.0
    for epoch in range(eval_config.fid_autoencoder_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for (batch,) in ae_loader:
            batch = batch.to(device)
            ae_optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = ae_criterion(reconstructed, batch)
            loss.backward()
            ae_optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        final_loss = epoch_loss / num_batches

    # Store reconstruction loss (paper target: 0.041)
    results['ae_reconstruction_loss'] = final_loss

    # Compute test reconstruction loss
    autoencoder.eval()
    with torch.no_grad():
        real_tensor = torch.tensor(real_gestures, dtype=torch.float32).to(device)
        fake_tensor = torch.tensor(fake_gestures, dtype=torch.float32).to(device)
        real_reconstructed = autoencoder(real_tensor)
        results['ae_test_loss'] = ae_criterion(real_reconstructed, real_tensor).item()

        # Extract features
        real_features = autoencoder.encode(real_tensor).cpu().numpy()
        fake_features = autoencoder.encode(fake_tensor).cpu().numpy()

    # Compute FID
    mu_real, mu_fake = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False) + np.eye(eval_config.fid_hidden_dim) * 1e-6
    cov_fake = np.cov(fake_features, rowvar=False) + np.eye(eval_config.fid_hidden_dim) * 1e-6
    diff = mu_real - mu_fake
    covmean = scipy_matrix_sqrt(cov_real @ cov_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    results['fid'] = float(np.sum(diff**2) + np.trace(cov_real + cov_fake - 2 * covmean))

    # Precision/Recall (k-NN based)
    k = eval_config.precision_recall_k
    real_dists = cdist(real_flat_xy, real_flat_xy, metric='euclidean')
    fake_dists = cdist(fake_flat_xy, fake_flat_xy, metric='euclidean')
    real_fake_dists = cdist(real_flat_xy, fake_flat_xy, metric='euclidean')

    real_radii = np.sort(real_dists, axis=1)[:, k]
    fake_radii = np.sort(fake_dists, axis=1)[:, k]

    # Precision: fraction of fake samples within real manifold
    prec = np.mean([np.any(real_fake_dists[:, j] <= real_radii) for j in range(n)])
    # Recall: fraction of real samples within fake manifold
    rec = np.mean([np.any(real_fake_dists[i, :] <= fake_radii) for i in range(n)])
    results['precision'] = prec
    results['recall'] = rec

    return results
