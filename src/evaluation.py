"""
Evaluation metrics for WordGesture-GAN.

Implements metrics from the paper:
- L2 and DTW Wasserstein distance
- Frechet Inception Distance (FID)
- Precision and Recall
- Velocity and Acceleration correlations
- Jerk analysis
"""

import numpy as np
import torch
from typing import Dict, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from .models import AutoEncoder
from .config import ModelConfig, EvaluationConfig, DEFAULT_MODEL_CONFIG, DEFAULT_EVALUATION_CONFIG


def compute_dtw_distance(gesture1: np.ndarray, gesture2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two gestures.

    Uses the full (x, y, t) representation.

    Args:
        gesture1: Gesture array of shape (seq_length, 3)
        gesture2: Gesture array of shape (seq_length, 3)

    Returns:
        DTW distance
    """
    n, m = len(gesture1), len(gesture2)

    # Compute pairwise distances
    cost_matrix = cdist(gesture1, gesture2, metric='euclidean')

    # Dynamic programming for DTW
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return dtw_matrix[n, m]


def scipy_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using eigendecomposition."""
    from scipy.linalg import sqrtm
    return sqrtm(matrix).real


def compute_duration_rmse(
    real_gestures: np.ndarray,
    fake_gestures: np.ndarray
) -> tuple[float, float]:
    """
    Compute duration RMSE between real and fake gestures.

    Assumes gestures have time in the third column (index 2) normalized to [0, 1].

    Args:
        real_gestures: Array of shape (n, seq_len, 3) with real gestures
        fake_gestures: Array of shape (n, seq_len, 3) with fake gestures

    Returns:
        Tuple of (rmse_normalized, rmse_ms) where rmse_ms assumes avg duration ~2000ms
    """
    # Duration is the last timestamp (normalized to [0, 1])
    real_durations = real_gestures[:, -1, 2]  # Last point's time
    fake_durations = fake_gestures[:, -1, 2]

    # RMSE in normalized units
    rmse_normalized = np.sqrt(np.mean((real_durations - fake_durations)**2))

    # Approximate ms (assuming avg duration of 2000ms)
    rmse_ms = rmse_normalized * 2000

    return rmse_normalized, rmse_ms


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
        - velocity_corr: Velocity correlation
        - acceleration_corr: Acceleration correlation
        - duration_rmse, duration_rmse_ms: Duration RMSE
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

    # DTW Wasserstein (using fastdtw if available, else simple DTW)
    try:
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
    except ImportError:
        # Fallback to simple per-sample DTW
        dtw_dists = []
        for i in range(min(n, 50)):  # Limit for speed
            d = compute_dtw_distance(real_gestures[i, :, :2], fake_gestures[i, :, :2])
            dtw_dists.append(d)
        results['dtw_wasserstein'] = np.mean(dtw_dists) / np.sqrt(model_config.seq_length)

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

    # Velocity correlation (direct comparison, i to i)
    vcorrs = []
    for i in range(n):
        vr = np.diff(real_gestures[i, :, :2], axis=0).flatten()
        vf = np.diff(fake_gestures[i, :, :2], axis=0).flatten()
        if len(vr) == len(vf) and len(vr) > 0:
            cr = np.corrcoef(vr, vf)[0, 1]
            if not np.isnan(cr):
                vcorrs.append(cr)
    results['velocity_corr'] = np.mean(vcorrs) if vcorrs else 0.0

    # Acceleration correlation (using savgol for smoother derivatives)
    # Compute both component-based (concat x,y) and magnitude-based correlation
    acorrs_concat = []
    acorrs_magnitude = []
    for i in range(n):
        xr, yr = real_gestures[i, :, 0], real_gestures[i, :, 1]
        xf, yf = fake_gestures[i, :, 0], fake_gestures[i, :, 1]
        if len(xr) >= eval_config.savgol_window:
            ax_r = savgol_filter(xr, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=2)
            ay_r = savgol_filter(yr, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=2)
            ax_f = savgol_filter(xf, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=2)
            ay_f = savgol_filter(yf, eval_config.savgol_window, eval_config.savgol_poly_order, deriv=2)

            # Component-based: concatenate x and y accelerations (256 values)
            ar_concat = np.concatenate([ax_r, ay_r])
            af_concat = np.concatenate([ax_f, ay_f])
            cr_concat = np.corrcoef(ar_concat, af_concat)[0, 1]
            if not np.isnan(cr_concat):
                acorrs_concat.append(cr_concat)

            # Magnitude-based: sqrt(ax^2 + ay^2) (128 values)
            ar_mag = np.sqrt(ax_r**2 + ay_r**2)
            af_mag = np.sqrt(ax_f**2 + ay_f**2)
            cr_mag = np.corrcoef(ar_mag, af_mag)[0, 1]
            if not np.isnan(cr_mag):
                acorrs_magnitude.append(cr_mag)

    results['acceleration_corr'] = np.mean(acorrs_concat) if acorrs_concat else 0.0
    results['acceleration_corr_magnitude'] = np.mean(acorrs_magnitude) if acorrs_magnitude else 0.0

    # Duration RMSE
    rmse_norm, rmse_ms = compute_duration_rmse(real_gestures, fake_gestures)
    results['duration_rmse'] = rmse_norm
    results['duration_rmse_ms'] = rmse_ms

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
