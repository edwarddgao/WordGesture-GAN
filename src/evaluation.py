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
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from collections import defaultdict

from .models import AutoEncoder
from .config import ModelConfig, DEFAULT_MODEL_CONFIG


def compute_l2_distance(gesture1: np.ndarray, gesture2: np.ndarray) -> float:
    """
    Compute L2 distance between two gestures.

    Args:
        gesture1: Gesture array of shape (seq_length, 3)
        gesture2: Gesture array of shape (seq_length, 3)

    Returns:
        L2 distance
    """
    return np.sqrt(np.sum((gesture1 - gesture2) ** 2))


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


def compute_wasserstein_distance_per_word(
    real_gestures: List[np.ndarray],
    fake_gestures: List[np.ndarray],
    distance_func: callable = compute_l2_distance
) -> float:
    """
    Compute Wasserstein distance for gestures of a single word.

    Uses minimum weight bipartite matching to find optimal pairing.

    Args:
        real_gestures: List of real gesture arrays
        fake_gestures: List of fake gesture arrays
        distance_func: Distance function to use

    Returns:
        Wasserstein distance for the word
    """
    n_real = len(real_gestures)
    n_fake = len(fake_gestures)

    if n_real == 0 or n_fake == 0:
        return 0.0

    # Build cost matrix
    n = max(n_real, n_fake)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < n_real and j < n_fake:
                cost_matrix[i, j] = distance_func(
                    real_gestures[i], fake_gestures[j]
                )
            else:
                cost_matrix[i, j] = 0  # Padding

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute mean distance from matched pairs
    total_distance = 0.0
    num_matches = min(n_real, n_fake)

    for i, j in zip(row_ind[:num_matches], col_ind[:num_matches]):
        if i < n_real and j < n_fake:
            total_distance += cost_matrix[i, j]

    return total_distance / num_matches if num_matches > 0 else 0.0


def compute_wasserstein_distance(
    real_gestures_by_word: Dict[str, List[np.ndarray]],
    fake_gestures_by_word: Dict[str, List[np.ndarray]],
    use_spatial_only: bool = False,
    use_dtw: bool = False
) -> Tuple[float, float]:
    """
    Compute overall Wasserstein distance.

    Args:
        real_gestures_by_word: Dict mapping word -> list of real gestures
        fake_gestures_by_word: Dict mapping word -> list of fake gestures
        use_spatial_only: If True, only use (x, y) coordinates
        use_dtw: If True, use DTW distance instead of L2

    Returns:
        Tuple of (mean_distance, std_distance)
    """
    distance_func = compute_dtw_distance if use_dtw else compute_l2_distance
    distances = []

    for word in real_gestures_by_word:
        if word not in fake_gestures_by_word:
            continue

        real = real_gestures_by_word[word]
        fake = fake_gestures_by_word[word]

        if use_spatial_only:
            real = [g[:, :2] for g in real]
            fake = [g[:, :2] for g in fake]

        dist = compute_wasserstein_distance_per_word(real, fake, distance_func)
        distances.append(dist)

    return np.mean(distances), np.std(distances)


class FIDCalculator:
    """
    Frechet Inception Distance calculator for gesture evaluation.

    Uses a trained autoencoder to extract features, then computes
    Frechet distance between real and generated gesture distributions.
    """

    def __init__(
        self,
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
        hidden_dim: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.autoencoder = AutoEncoder(model_config, hidden_dim).to(self.device)
        self.is_trained = False

    def train_autoencoder(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001
    ) -> Tuple[float, float]:
        """
        Train the autoencoder on real gestures.

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Tuple of (final_train_loss, final_test_loss)
        """
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=learning_rate
        )
        criterion = nn.L1Loss()

        self.autoencoder.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                gesture = batch['gesture'].to(self.device)

                optimizer.zero_grad()
                reconstructed = self.autoencoder(gesture)
                loss = criterion(reconstructed, gesture)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Autoencoder Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True
        return total_loss / num_batches

    def extract_features(self, gestures: torch.Tensor) -> np.ndarray:
        """
        Extract latent features from gestures.

        Args:
            gestures: Gesture tensor of shape (batch, seq_length, 3)

        Returns:
            Feature array of shape (batch, hidden_dim)
        """
        self.autoencoder.eval()
        with torch.no_grad():
            features = self.autoencoder.encode(gestures.to(self.device))
        return features.cpu().numpy()

    def compute_fid(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray
    ) -> float:
        """
        Compute Frechet Inception Distance.

        FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r * Sigma_f))

        Args:
            real_features: Features from real gestures
            fake_features: Features from generated gestures

        Returns:
            FID score
        """
        # Compute statistics
        mu_r = np.mean(real_features, axis=0)
        mu_f = np.mean(fake_features, axis=0)

        sigma_r = np.cov(real_features, rowvar=False)
        sigma_f = np.cov(fake_features, rowvar=False)

        # Ensure positive definite
        sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * 1e-6
        sigma_f = sigma_f + np.eye(sigma_f.shape[0]) * 1e-6

        # FID formula
        diff = mu_r - mu_f
        mean_term = np.dot(diff, diff)

        # Matrix square root using eigendecomposition
        try:
            covmean = scipy_matrix_sqrt(sigma_r @ sigma_f)
            trace_term = np.trace(sigma_r + sigma_f - 2 * covmean)
        except Exception:
            # Fallback: simplified approximation
            trace_term = np.trace(sigma_r) + np.trace(sigma_f)

        fid = mean_term + trace_term
        return float(fid)


def scipy_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using eigendecomposition."""
    from scipy.linalg import sqrtm
    return sqrtm(matrix).real


def compute_precision_recall(
    real_gestures: List[np.ndarray],
    fake_gestures: List[np.ndarray],
    k: int = 3
) -> Tuple[float, float]:
    """
    Compute precision and recall using k-NN manifold estimation.

    From "Improved Precision and Recall Metric for Assessing Generative Models"

    Args:
        real_gestures: List of real gesture arrays
        fake_gestures: List of generated gesture arrays
        k: Number of nearest neighbors for manifold estimation

    Returns:
        Tuple of (precision, recall)
    """
    if len(real_gestures) < k + 1 or len(fake_gestures) < k + 1:
        return 0.0, 0.0

    # Flatten gestures for distance computation
    real_flat = np.array([g.flatten() for g in real_gestures])
    fake_flat = np.array([g.flatten() for g in fake_gestures])

    # Compute pairwise distances
    real_to_real = cdist(real_flat, real_flat, metric='euclidean')
    fake_to_fake = cdist(fake_flat, fake_flat, metric='euclidean')
    fake_to_real = cdist(fake_flat, real_flat, metric='euclidean')
    real_to_fake = cdist(real_flat, fake_flat, metric='euclidean')

    # Get k-th nearest neighbor distances (excluding self)
    np.fill_diagonal(real_to_real, np.inf)
    np.fill_diagonal(fake_to_fake, np.inf)

    real_knn_dist = np.sort(real_to_real, axis=1)[:, k - 1]
    fake_knn_dist = np.sort(fake_to_fake, axis=1)[:, k - 1]

    # Precision: fraction of fake samples falling within real manifold
    precision_count = 0
    for i, fake_sample in enumerate(fake_flat):
        min_dist_to_real = np.min(fake_to_real[i])
        # Check if fake sample falls within any real sample's ball
        if np.any(min_dist_to_real <= real_knn_dist):
            precision_count += 1
    precision = precision_count / len(fake_gestures)

    # Recall: fraction of real samples falling within fake manifold
    recall_count = 0
    for i, real_sample in enumerate(real_flat):
        min_dist_to_fake = np.min(real_to_fake[i])
        # Check if real sample falls within any fake sample's ball
        if np.any(min_dist_to_fake <= fake_knn_dist):
            recall_count += 1
    recall = recall_count / len(real_gestures)

    return precision, recall


def compute_velocity_acceleration(
    gesture: np.ndarray,
    window_size: int = 5,
    poly_order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and acceleration using Savitzky-Golay filter.

    Args:
        gesture: Gesture array of shape (seq_length, 3)
        window_size: Window size for filter
        poly_order: Polynomial order for filter

    Returns:
        Tuple of (velocity, acceleration) arrays
    """
    # Extract spatial coordinates
    x = gesture[:, 0]
    y = gesture[:, 1]
    t = gesture[:, 2]

    # Ensure enough points for filter
    if len(x) < window_size:
        return np.zeros(len(x)), np.zeros(len(x))

    # Compute derivatives using Savitzky-Golay filter
    dx = savgol_filter(x, window_size, poly_order, deriv=1)
    dy = savgol_filter(y, window_size, poly_order, deriv=1)
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6  # Avoid division by zero

    # Velocity magnitude
    vx = dx / dt
    vy = dy / dt
    velocity = np.sqrt(vx**2 + vy**2)

    # Acceleration
    dvx = savgol_filter(vx, window_size, poly_order, deriv=1)
    dvy = savgol_filter(vy, window_size, poly_order, deriv=1)
    ax = dvx / dt
    ay = dvy / dt
    acceleration = np.sqrt(ax**2 + ay**2)

    return velocity, acceleration


def compute_jerk(
    gesture: np.ndarray,
    window_size: int = 5,
    poly_order: int = 3
) -> np.ndarray:
    """
    Compute jerk (third derivative of position).

    Args:
        gesture: Gesture array of shape (seq_length, 3)
        window_size: Window size for filter
        poly_order: Polynomial order for filter

    Returns:
        Jerk array
    """
    x = gesture[:, 0]
    y = gesture[:, 1]
    t = gesture[:, 2]

    if len(x) < window_size:
        return np.zeros(len(x))

    # Third derivative using Savitzky-Golay filter
    d3x = savgol_filter(x, window_size, poly_order, deriv=3)
    d3y = savgol_filter(y, window_size, poly_order, deriv=3)

    dt = np.gradient(t)
    dt[dt == 0] = 1e-6

    jerk_x = d3x / (dt ** 3)
    jerk_y = d3y / (dt ** 3)
    jerk = np.sqrt(jerk_x**2 + jerk_y**2)

    return jerk


def compute_velocity_correlation(
    real_gestures: List[np.ndarray],
    fake_gestures: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Compute mean velocity correlation between real and fake gestures.

    Returns:
        Tuple of (mean_correlation, std_correlation)
    """
    correlations = []

    for real, fake in zip(real_gestures, fake_gestures):
        vel_real, _ = compute_velocity_acceleration(real)
        vel_fake, _ = compute_velocity_acceleration(fake)

        if len(vel_real) > 1 and len(vel_fake) > 1:
            # Resample to same length if needed
            if len(vel_real) != len(vel_fake):
                min_len = min(len(vel_real), len(vel_fake))
                vel_real = vel_real[:min_len]
                vel_fake = vel_fake[:min_len]

            try:
                corr, _ = pearsonr(vel_real, vel_fake)
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception:
                pass

    if len(correlations) == 0:
        return 0.0, 0.0

    return np.mean(correlations), np.std(correlations)


def compute_acceleration_correlation(
    real_gestures: List[np.ndarray],
    fake_gestures: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Compute mean acceleration correlation between real and fake gestures.

    Returns:
        Tuple of (mean_correlation, std_correlation)
    """
    correlations = []

    for real, fake in zip(real_gestures, fake_gestures):
        _, acc_real = compute_velocity_acceleration(real)
        _, acc_fake = compute_velocity_acceleration(fake)

        if len(acc_real) > 1 and len(acc_fake) > 1:
            if len(acc_real) != len(acc_fake):
                min_len = min(len(acc_real), len(acc_fake))
                acc_real = acc_real[:min_len]
                acc_fake = acc_fake[:min_len]

            try:
                corr, _ = pearsonr(acc_real, acc_fake)
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception:
                pass

    if len(correlations) == 0:
        return 0.0, 0.0

    return np.mean(correlations), np.std(correlations)


def compute_mean_jerk(gestures: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute mean and std of jerk across gestures.

    Returns:
        Tuple of (mean_jerk, std_jerk)
    """
    all_jerks = []

    for gesture in gestures:
        jerk = compute_jerk(gesture)
        all_jerks.extend(jerk.tolist())

    return np.mean(all_jerks), np.std(all_jerks)


def evaluate_model(
    real_gestures_by_word: Dict[str, List[np.ndarray]],
    fake_gestures_by_word: Dict[str, List[np.ndarray]],
    fid_calculator: Optional[FIDCalculator] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        real_gestures_by_word: Dict mapping word -> list of real gestures
        fake_gestures_by_word: Dict mapping word -> list of fake gestures
        fid_calculator: Optional trained FID calculator

    Returns:
        Dictionary of evaluation metrics
    """
    results = {}

    # Flatten gestures for some metrics
    all_real = []
    all_fake = []
    for word in real_gestures_by_word:
        if word in fake_gestures_by_word:
            all_real.extend(real_gestures_by_word[word])
            all_fake.extend(fake_gestures_by_word[word])

    # L2 Wasserstein distance
    l2_mean, l2_std = compute_wasserstein_distance(
        real_gestures_by_word, fake_gestures_by_word,
        use_spatial_only=False, use_dtw=False
    )
    results['l2_wasserstein_xyz_mean'] = l2_mean
    results['l2_wasserstein_xyz_std'] = l2_std

    l2_xy_mean, l2_xy_std = compute_wasserstein_distance(
        real_gestures_by_word, fake_gestures_by_word,
        use_spatial_only=True, use_dtw=False
    )
    results['l2_wasserstein_xy_mean'] = l2_xy_mean
    results['l2_wasserstein_xy_std'] = l2_xy_std

    # DTW Wasserstein distance
    dtw_mean, dtw_std = compute_wasserstein_distance(
        real_gestures_by_word, fake_gestures_by_word,
        use_spatial_only=False, use_dtw=True
    )
    results['dtw_wasserstein_xyz_mean'] = dtw_mean
    results['dtw_wasserstein_xyz_std'] = dtw_std

    # Precision and Recall
    if len(all_real) > 10 and len(all_fake) > 10:
        precision, recall = compute_precision_recall(all_real, all_fake, k=3)
        results['precision'] = precision
        results['recall'] = recall

    # Velocity and Acceleration correlations
    vel_corr_mean, vel_corr_std = compute_velocity_correlation(all_real, all_fake)
    results['velocity_correlation_mean'] = vel_corr_mean
    results['velocity_correlation_std'] = vel_corr_std

    acc_corr_mean, acc_corr_std = compute_acceleration_correlation(all_real, all_fake)
    results['acceleration_correlation_mean'] = acc_corr_mean
    results['acceleration_correlation_std'] = acc_corr_std

    # Jerk analysis
    real_jerk_mean, real_jerk_std = compute_mean_jerk(all_real)
    fake_jerk_mean, fake_jerk_std = compute_mean_jerk(all_fake)
    results['real_jerk_mean'] = real_jerk_mean
    results['real_jerk_std'] = real_jerk_std
    results['fake_jerk_mean'] = fake_jerk_mean
    results['fake_jerk_std'] = fake_jerk_std

    return results
