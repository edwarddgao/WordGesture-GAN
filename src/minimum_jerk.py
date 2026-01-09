"""
Minimum Jerk Model for word-gesture generation.

Implements the minimum jerk model from:
"Modeling Gesture-Typing Movements" (Quinn & Zhai, Human-Computer Interaction, 2018)

The minimum jerk model generates smooth word-gestures by:
1. Extracting via-points (key centers + midpoints)
2. Applying learned offset distributions
3. Constructing minimum-jerk trajectories between via-points

The model produces (x, y) trajectories with relative timing only.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .keyboard import QWERTYKeyboard


@dataclass
class MinimumJerkConfig:
    """Configuration for Minimum Jerk model."""
    seq_length: int = 128  # Number of points per gesture
    use_midpoints: bool = False  # Include midpoints as via-points (False works better)
    noise_scale: float = 0.15  # Scale factor for offset noise (tuned for precision/recall)
    random_seed: Optional[int] = None


def minimum_jerk_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    n_points: int
) -> np.ndarray:
    """
    Generate minimum jerk trajectory between two points.

    Uses the quintic polynomial that minimizes jerk (3rd derivative):
        x(t) = x0 + (x1 - x0) * (10*t^3 - 15*t^4 + 6*t^5)

    This satisfies boundary conditions:
        x(0) = x0, x(1) = x1
        x'(0) = x'(1) = 0 (zero velocity at endpoints)
        x''(0) = x''(1) = 0 (zero acceleration at endpoints)

    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        n_points: Number of points in trajectory

    Returns:
        Array of shape (n_points, 2) with (x, y) coordinates
    """
    if n_points < 2:
        return start.reshape(1, -1)

    t = np.linspace(0, 1, n_points)

    # Quintic polynomial for minimum jerk
    # s(t) = 10*t^3 - 15*t^4 + 6*t^5
    s = 10 * t**3 - 15 * t**4 + 6 * t**5

    # Interpolate between start and end
    trajectory = start + np.outer(s, (end - start))

    return trajectory


class MinimumJerkModel:
    """
    Minimum Jerk gesture generator based on Quinn & Zhai (2018).

    Generates smooth word-gestures by:
    1. Extracting via-points (key centers + optional midpoints)
    2. Applying learned offset distributions from training data
    3. Constructing minimum-jerk trajectories between via-points

    The model only produces (x, y) spatial coordinates with uniform
    relative timing - it cannot predict absolute timestamps.

    Example usage:
        keyboard = QWERTYKeyboard()
        model = MinimumJerkModel(keyboard)

        # Train on real gestures
        model.train(gestures, words)

        # Generate gestures
        fake_gestures = model.generate_batch(test_words)
    """

    def __init__(
        self,
        keyboard: QWERTYKeyboard,
        config: MinimumJerkConfig = MinimumJerkConfig()
    ):
        """
        Initialize Minimum Jerk model.

        Args:
            keyboard: QWERTYKeyboard instance for key positions
            config: Model configuration
        """
        self.keyboard = keyboard
        self.config = config
        self.seq_length = config.seq_length

        # Learned parameters (initialized in train())
        self.offset_mean = np.zeros(2)
        self.offset_std = np.array([0.05, 0.05])  # Default small offset
        self.is_trained = False

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def train(
        self,
        gestures: List[np.ndarray],
        words: List[str],
        verbose: bool = True
    ) -> None:
        """
        Learn offset distributions from training data.

        For each gesture-word pair:
        1. Find gesture points at key center positions
        2. Compute offsets (gesture_point - key_center)
        3. Aggregate to get mean/std of offsets

        Args:
            gestures: List of gesture arrays, each (seq_len, 2) or (seq_len, 3)
            words: List of corresponding words
            verbose: Print training statistics
        """
        all_offsets = []

        for gesture, word in zip(gestures, words):
            # Get key centers and indices for this word
            key_centers = self.keyboard.get_key_centers_for_word(word)
            key_indices = self.keyboard.get_key_indices(word, len(gesture))

            if len(key_centers) == 0 or len(key_indices) == 0:
                continue

            # Compute offset at each key position
            for idx, center in zip(key_indices, key_centers):
                if idx < len(gesture):
                    gesture_point = gesture[idx, :2]  # Only (x, y)
                    offset = gesture_point - center
                    all_offsets.append(offset)

        if len(all_offsets) == 0:
            if verbose:
                print("Warning: No valid offsets found in training data")
            return

        all_offsets = np.array(all_offsets)

        # Compute statistics
        self.offset_mean = np.mean(all_offsets, axis=0)
        self.offset_std = np.std(all_offsets, axis=0)

        # Ensure minimum std to prevent degenerate sampling
        self.offset_std = np.maximum(self.offset_std, 0.01)

        self.is_trained = True

        if verbose:
            print(f"Minimum Jerk Model trained on {len(gestures)} gestures")
            print(f"  Offset mean: ({self.offset_mean[0]:.4f}, {self.offset_mean[1]:.4f})")
            print(f"  Offset std:  ({self.offset_std[0]:.4f}, {self.offset_std[1]:.4f})")

    def _extract_via_points(
        self,
        word: str,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Extract via-points for a word.

        Via-points consist of:
        - Key centers (with optional learned offset noise)
        - Midpoints between consecutive keys (if config.use_midpoints)

        Args:
            word: Target word
            add_noise: Whether to add Gaussian offset noise

        Returns:
            Array of via-points, shape (n_via_points, 2)
        """
        key_centers = self.keyboard.get_key_centers_for_word(word)

        if len(key_centers) == 0:
            return np.zeros((1, 2))

        if len(key_centers) == 1:
            if add_noise:
                offset = np.random.normal(self.offset_mean, self.offset_std)
                return (key_centers + offset).reshape(1, 2)
            return key_centers.reshape(1, 2)

        via_points = []

        for i, center in enumerate(key_centers):
            # Add key center with offset
            if add_noise:
                # Apply learned offset with scaled noise
                # Full std causes too much spread (low precision, high recall)
                offset = np.random.normal(self.offset_mean, self.offset_std * self.config.noise_scale)
                via_points.append(center + offset)
            else:
                via_points.append(center)

            # Add midpoint (except after last key)
            if self.config.use_midpoints and i < len(key_centers) - 1:
                midpoint = (key_centers[i] + key_centers[i + 1]) / 2
                via_points.append(midpoint)

        return np.array(via_points)

    def generate(
        self,
        word: str,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Generate a minimum jerk gesture for a word.

        Following Quinn & Zhai (2018): construct minimum-jerk trajectory segments
        between adjacent via-points and concatenate them. Each segment uses the
        quintic polynomial that minimizes jerk.

        Args:
            word: Target word
            add_noise: Whether to add learned offset noise

        Returns:
            Gesture array of shape (seq_length, 3) with (x, y, t)
            Time is uniformly distributed [0, 1]
        """
        via_points = self._extract_via_points(word, add_noise)

        if len(via_points) == 1:
            # Single point - replicate
            trajectory = np.tile(via_points[0], (self.seq_length, 1))
        elif len(via_points) == 2:
            # Two points - simple quintic polynomial
            trajectory = minimum_jerk_trajectory(
                via_points[0], via_points[1], self.seq_length
            )
        else:
            # Multiple via-points - concatenate minimum-jerk segments
            n_segments = len(via_points) - 1
            points_per_segment = max(2, self.seq_length // n_segments)

            segments = []
            for i in range(n_segments):
                # Last segment gets remaining points to total seq_length
                if i == n_segments - 1:
                    n_pts = self.seq_length - sum(len(s) - 1 for s in segments)
                else:
                    n_pts = points_per_segment

                # Generate minimum jerk segment
                segment = minimum_jerk_trajectory(
                    via_points[i], via_points[i + 1], n_pts
                )

                # Skip first point to avoid duplication at junction
                if i > 0:
                    segment = segment[1:]

                segments.append(segment)

            trajectory = np.vstack(segments)

        # Ensure exactly seq_length points
        if len(trajectory) > self.seq_length:
            indices = np.linspace(0, len(trajectory) - 1, self.seq_length, dtype=int)
            trajectory = trajectory[indices]
        elif len(trajectory) < self.seq_length:
            padding = np.tile(trajectory[-1], (self.seq_length - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])

        # Add uniform time dimension
        times = np.linspace(0, 1, self.seq_length).reshape(-1, 1)
        gesture = np.hstack([trajectory, times]).astype(np.float32)

        return gesture

    def generate_batch(
        self,
        words: List[str],
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Generate minimum jerk gestures for multiple words.

        Args:
            words: List of target words
            add_noise: Whether to add learned offset noise

        Returns:
            Array of shape (n_words, seq_length, 3)
        """
        gestures = []
        for word in words:
            gesture = self.generate(word, add_noise)
            gestures.append(gesture)
        return np.array(gestures)

    def generate_multiple(
        self,
        word: str,
        n_samples: int,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Generate multiple gesture samples for a single word.

        Args:
            word: Target word
            n_samples: Number of samples to generate
            add_noise: Whether to add learned offset noise

        Returns:
            Array of shape (n_samples, seq_length, 3)
        """
        gestures = []
        for _ in range(n_samples):
            gesture = self.generate(word, add_noise)
            gestures.append(gesture)
        return np.array(gestures)


def evaluate_minimum_jerk_model(
    model: MinimumJerkModel,
    test_gestures: List[np.ndarray],
    test_words: List[str],
    verbose: bool = True
) -> dict:
    """
    Evaluate Minimum Jerk model on test data.

    Computes metrics matching paper Tables 1-6:
    - L2 Wasserstein distance (x,y only)
    - DTW Wasserstein distance (x,y only)
    - Jerk
    - Velocity correlation
    - Acceleration correlation

    Note: FID and Precision/Recall require the full evaluation pipeline.

    Args:
        model: Trained MinimumJerkModel
        test_gestures: List of real test gestures
        test_words: List of corresponding words
        verbose: Print progress

    Returns:
        Dictionary of evaluation metrics
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import savgol_filter

    n = len(test_gestures)

    # Generate fake gestures
    if verbose:
        print(f"Generating {n} minimum jerk gestures...")
    fake_gestures = model.generate_batch(test_words)

    # Extract (x, y) only
    real_xy = np.array([g[:, :2] for g in test_gestures])
    fake_xy = fake_gestures[:, :, :2]

    results = {}

    # L2 Wasserstein distance
    if verbose:
        print("Computing L2 Wasserstein distance...")
    real_flat = real_xy.reshape(n, -1)
    fake_flat = fake_xy.reshape(n, -1)
    dist_matrix = cdist(real_flat, fake_flat, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    results['l2_wasserstein'] = dist_matrix[row_ind, col_ind].mean()

    # DTW Wasserstein distance
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        from joblib import Parallel, delayed

        if verbose:
            print("Computing DTW Wasserstein distance...")

        def compute_dtw_row(i):
            row = np.zeros(n)
            for j in range(n):
                distance, _ = fastdtw(real_xy[i], fake_xy[j], dist=euclidean)
                row[j] = distance
            return row

        dtw_rows = Parallel(n_jobs=-1, verbose=0)(
            delayed(compute_dtw_row)(i) for i in range(n)
        )
        dtw_dist = np.array(dtw_rows)
        row_ind2, col_ind2 = linear_sum_assignment(dtw_dist)
        dtw_raw = dtw_dist[row_ind2, col_ind2].mean()
        results['dtw_wasserstein'] = dtw_raw / np.sqrt(model.seq_length)
    except ImportError:
        results['dtw_wasserstein'] = None

    # Jerk
    if verbose:
        print("Computing jerk...")

    def compute_jerk(g):
        x, y = g[:, 0], g[:, 1]
        if len(x) < 5:
            return 0.0
        d3x = savgol_filter(x, 5, 3, deriv=3)
        d3y = savgol_filter(y, 5, 3, deriv=3)
        return np.mean(np.sqrt(d3x**2 + d3y**2))

    results['jerk_real'] = np.mean([compute_jerk(g) for g in real_xy])
    results['jerk_fake'] = np.mean([compute_jerk(g) for g in fake_xy])

    # Velocity correlation
    if verbose:
        print("Computing velocity correlation...")
    vcorrs = []
    for i in range(n):
        vr = np.diff(real_xy[i], axis=0).flatten()
        vf = np.diff(fake_xy[col_ind[i]], axis=0).flatten()
        if len(vr) == len(vf):
            corr = np.corrcoef(vr, vf)[0, 1]
            if not np.isnan(corr):
                vcorrs.append(corr)
    results['velocity_corr'] = np.mean(vcorrs) if vcorrs else 0.0

    # Acceleration correlation
    if verbose:
        print("Computing acceleration correlation...")
    acorrs = []
    for i in range(n):
        xr, yr = real_xy[i, :, 0], real_xy[i, :, 1]
        xf, yf = fake_xy[col_ind[i], :, 0], fake_xy[col_ind[i], :, 1]
        if len(xr) >= 5:
            ax_r = savgol_filter(xr, 5, 3, deriv=2)
            ay_r = savgol_filter(yr, 5, 3, deriv=2)
            ax_f = savgol_filter(xf, 5, 3, deriv=2)
            ay_f = savgol_filter(yf, 5, 3, deriv=2)
            ar = np.concatenate([ax_r, ay_r])
            af = np.concatenate([ax_f, ay_f])
            corr = np.corrcoef(ar, af)[0, 1]
            if not np.isnan(corr):
                acorrs.append(corr)
    results['acceleration_corr'] = np.mean(acorrs) if acorrs else 0.0

    if verbose:
        print("\n=== Minimum Jerk Model Results ===")
        print(f"L2 Wasserstein (x,y):     {results['l2_wasserstein']:.3f} (paper: 5.004)")
        if results['dtw_wasserstein']:
            print(f"DTW Wasserstein (x,y):    {results['dtw_wasserstein']:.3f} (paper: 2.752)")
        print(f"Jerk (fake):              {results['jerk_fake']:.6f} (paper: 0.0034)")
        print(f"Jerk (real):              {results['jerk_real']:.6f} (paper: 0.0066)")
        print(f"Velocity Correlation:     {results['velocity_corr']:.3f} (paper: 0.40)")
        print(f"Acceleration Correlation: {results['acceleration_corr']:.3f} (paper: 0.21)")

    return results
