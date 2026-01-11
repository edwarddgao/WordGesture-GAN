"""
Keyboard layout utilities for generating word prototypes.

Includes minimum jerk trajectory generation following Quinn & Zhai (2018):
"Modeling Gesture-Typing Movements", Human-Computer Interaction.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from .config import KeyboardConfig, DEFAULT_KEYBOARD_CONFIG


@dataclass
class MinimumJerkDistributions:
    """
    Learned distributions for minimum jerk model following Quinn & Zhai (2018).

    Contains:
    - Key center offset distribution: How far users deviate from key centers
    - Midpoint angle distribution: Angular deviation of midpoints between keys
    """
    # Key center offsets (for interior keys, not first/last)
    # Stored as (mean_x, std_x, mean_y, std_y)
    key_offset_mean_x: float = 0.0
    key_offset_std_x: float = 0.02
    key_offset_mean_y: float = 0.0
    key_offset_std_y: float = 0.02

    # Midpoint angle distribution
    # The angle is measured as perpendicular deviation from the key-to-key line
    # normalized by segment length (so it's dimensionless)
    midpoint_angle_mean: float = 0.0
    midpoint_angle_std: float = 0.1

    # Number of samples used to fit these distributions
    n_key_offset_samples: int = 0
    n_midpoint_samples: int = 0

    def is_fitted(self) -> bool:
        """Check if model has been fitted to data."""
        return self.n_key_offset_samples > 0 or self.n_midpoint_samples > 0


class MinimumJerkModel:
    """
    Minimum jerk model that learns offset distributions from training data.

    Following Quinn & Zhai (2018), this model learns:
    1. The aggregate distribution of offsets from key centers
    2. The mean and std of angles for midpoints between consecutive keys

    These learned distributions are then used when generating synthetic
    minimum jerk trajectories.
    """

    def __init__(self, keyboard: 'QWERTYKeyboard'):
        """
        Initialize the minimum jerk model.

        Args:
            keyboard: QWERTYKeyboard instance for key positions
        """
        self.keyboard = keyboard
        self.distributions = MinimumJerkDistributions()

    def fit(
        self,
        gestures_by_word: Dict[str, List[np.ndarray]],
        verbose: bool = True
    ) -> 'MinimumJerkModel':
        """
        Fit the model to training data by extracting offset distributions.

        Args:
            gestures_by_word: Dictionary mapping word -> list of gesture arrays
                              Each gesture is (seq_length, 3) with (x, y, t)
            verbose: Whether to print fitting progress

        Returns:
            self (for method chaining)
        """
        key_offsets_x = []
        key_offsets_y = []
        midpoint_angles = []

        for word, gestures in gestures_by_word.items():
            key_positions = self._get_key_positions(word)
            if len(key_positions) < 2:
                continue

            key_positions = np.array(key_positions)
            n_keys = len(key_positions)

            for gesture in gestures:
                # Extract key center offsets for interior keys
                if n_keys > 2:
                    offsets = self._extract_key_offsets(gesture, key_positions)
                    for ox, oy in offsets:
                        key_offsets_x.append(ox)
                        key_offsets_y.append(oy)

                # Extract midpoint angles for all consecutive key pairs
                angles = self._extract_midpoint_angles(gesture, key_positions)
                midpoint_angles.extend(angles)

        # Compute statistics
        if len(key_offsets_x) > 0:
            self.distributions.key_offset_mean_x = float(np.mean(key_offsets_x))
            self.distributions.key_offset_std_x = float(np.std(key_offsets_x))
            self.distributions.key_offset_mean_y = float(np.mean(key_offsets_y))
            self.distributions.key_offset_std_y = float(np.std(key_offsets_y))
            self.distributions.n_key_offset_samples = len(key_offsets_x)

        if len(midpoint_angles) > 0:
            self.distributions.midpoint_angle_mean = float(np.mean(midpoint_angles))
            self.distributions.midpoint_angle_std = float(np.std(midpoint_angles))
            self.distributions.n_midpoint_samples = len(midpoint_angles)

        if verbose:
            print(f"MinimumJerkModel fitted:")
            print(f"  Key offsets: mean=({self.distributions.key_offset_mean_x:.4f}, "
                  f"{self.distributions.key_offset_mean_y:.4f}), "
                  f"std=({self.distributions.key_offset_std_x:.4f}, "
                  f"{self.distributions.key_offset_std_y:.4f}) "
                  f"[n={self.distributions.n_key_offset_samples}]")
            print(f"  Midpoint angles: mean={self.distributions.midpoint_angle_mean:.4f}, "
                  f"std={self.distributions.midpoint_angle_std:.4f} "
                  f"[n={self.distributions.n_midpoint_samples}]")

        return self

    def _get_key_positions(self, word: str) -> List[Tuple[float, float]]:
        """Get key center positions for a word."""
        positions = []
        for letter in word.lower():
            center = self.keyboard.get_key_center(letter)
            if center is not None:
                positions.append(center)
        return positions

    def _extract_key_offsets(
        self,
        gesture: np.ndarray,
        key_positions: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Extract offsets from key centers for interior keys.

        For each interior key (not first or last), find the closest point
        on the gesture and compute the offset from the key center.

        Args:
            gesture: (seq_length, 3) array with (x, y, t)
            key_positions: (n_keys, 2) array of key centers

        Returns:
            List of (offset_x, offset_y) tuples for interior keys
        """
        offsets = []
        n_keys = len(key_positions)
        gesture_xy = gesture[:, :2]

        # For interior keys (indices 1 to n_keys-2)
        for key_idx in range(1, n_keys - 1):
            key_pos = key_positions[key_idx]

            # Find the gesture point closest to this key center
            distances = np.linalg.norm(gesture_xy - key_pos, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = gesture_xy[closest_idx]

            # Compute offset
            offset_x = closest_point[0] - key_pos[0]
            offset_y = closest_point[1] - key_pos[1]
            offsets.append((offset_x, offset_y))

        return offsets

    def _extract_midpoint_angles(
        self,
        gesture: np.ndarray,
        key_positions: np.ndarray
    ) -> List[float]:
        """
        Extract midpoint angles for consecutive key pairs.

        For each pair of consecutive keys, find the gesture point closest to
        the theoretical midpoint, and compute the perpendicular deviation
        (angle) from the key-to-key line.

        The angle is normalized by the segment length to be scale-invariant.

        Args:
            gesture: (seq_length, 3) array with (x, y, t)
            key_positions: (n_keys, 2) array of key centers

        Returns:
            List of normalized perpendicular deviations (angles)
        """
        angles = []
        n_keys = len(key_positions)
        gesture_xy = gesture[:, :2]

        for i in range(n_keys - 1):
            key_start = key_positions[i]
            key_end = key_positions[i + 1]

            # Theoretical midpoint
            midpoint = (key_start + key_end) / 2

            # Direction vector and segment length
            direction = key_end - key_start
            seg_length = np.linalg.norm(direction)

            if seg_length < 1e-6:
                continue

            # Perpendicular direction (90 degrees counterclockwise)
            perp = np.array([-direction[1], direction[0]]) / seg_length

            # Find gesture point closest to the theoretical midpoint
            distances = np.linalg.norm(gesture_xy - midpoint, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = gesture_xy[closest_idx]

            # Compute perpendicular deviation from the key-to-key line
            # This is the "angle" in Quinn & Zhai's terminology
            deviation_vec = closest_point - midpoint
            perp_deviation = np.dot(deviation_vec, perp)

            # Normalize by segment length to make it scale-invariant
            normalized_angle = perp_deviation / seg_length
            angles.append(normalized_angle)

        return angles

    def generate_trajectory(
        self,
        word: str,
        num_points: int = 128,
        include_midpoints: bool = True
    ) -> np.ndarray:
        """
        Generate a minimum jerk trajectory using learned distributions.

        Args:
            word: Target word
            num_points: Number of points in trajectory
            include_midpoints: Whether to include midpoints between keys

        Returns:
            (num_points, 3) array with (x, y, t) trajectory
        """
        key_positions = self._get_key_positions(word)

        if len(key_positions) < 2:
            if len(key_positions) == 1:
                return self.keyboard._make_single_point_prototype(
                    *key_positions[0], num_points
                )
            return np.zeros((num_points, 3), dtype=np.float32)

        key_positions = np.array(key_positions)

        return generate_minimum_jerk_trajectory_fitted(
            via_points=key_positions,
            num_points=num_points,
            include_midpoints=include_midpoints,
            key_offset_mean=(
                self.distributions.key_offset_mean_x,
                self.distributions.key_offset_mean_y
            ),
            key_offset_std=(
                self.distributions.key_offset_std_x,
                self.distributions.key_offset_std_y
            ),
            midpoint_angle_mean=self.distributions.midpoint_angle_mean,
            midpoint_angle_std=self.distributions.midpoint_angle_std
        )


def minimum_jerk_quintic(t: np.ndarray) -> np.ndarray:
    """
    Standard minimum jerk interpolation: s(t) = 10t³ - 15t⁴ + 6t⁵

    Satisfies boundary conditions:
    - s(0)=0, s(1)=1
    - s'(0)=s'(1)=0 (zero velocity)
    - s''(0)=s''(1)=0 (zero acceleration)
    """
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def quintic_hermite_segment(
    p0: np.ndarray, p1: np.ndarray,
    v0: np.ndarray, v1: np.ndarray,
    a0: np.ndarray, a1: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Quintic Hermite interpolation for a single segment.

    Given position, velocity, and acceleration at both endpoints,
    compute the trajectory satisfying all boundary conditions.

    Args:
        p0, p1: Position at t=0 and t=1
        v0, v1: Velocity at t=0 and t=1
        a0, a1: Acceleration at t=0 and t=1
        t: Array of parameter values in [0, 1]

    Returns:
        Array of shape (len(t), 2) with (x, y) positions
    """
    # Quintic Hermite basis functions
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Position basis functions
    h00 = 1 - 10*t3 + 15*t4 - 6*t5
    h01 = 10*t3 - 15*t4 + 6*t5

    # Velocity basis functions
    h10 = t - 6*t3 + 8*t4 - 3*t5
    h11 = -4*t3 + 7*t4 - 3*t5

    # Acceleration basis functions
    h20 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
    h21 = 0.5*t3 - t4 + 0.5*t5

    # Combine: p(t) = h00*p0 + h01*p1 + h10*v0 + h11*v1 + h20*a0 + h21*a1
    result = (np.outer(h00, p0) + np.outer(h01, p1) +
              np.outer(h10, v0) + np.outer(h11, v1) +
              np.outer(h20, a0) + np.outer(h21, a1))
    return result


def _generate_fine_trajectory_with_tau(
    points: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    num_fine: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory at fine τ resolution for arc-length/time mapping.

    The parameter τ in quintic Hermite interpolation IS normalized time.
    This function generates the trajectory at fine τ resolution so we can
    compute the s(τ) relationship (arc-length as function of time).

    Args:
        points: Array of shape (n, 2) with via-point positions
        velocities: Array of shape (n, 2) with velocities at each via-point
        accelerations: Array of shape (n, 2) with accelerations at each via-point
        num_fine: Number of fine resolution points

    Returns:
        trajectory: (num_fine, 2) array of (x, y) positions
        tau_values: (num_fine,) array of τ values in [0, 1]
    """
    n = len(points)
    tau_fine = np.linspace(0, 1, num_fine)

    # Map global τ to segment indices and local τ values (vectorized)
    segment_tau = tau_fine * (n - 1)
    seg_indices = np.minimum(segment_tau.astype(int), n - 2)
    local_tau = segment_tau - seg_indices

    # Process each segment's points in batch
    trajectory = np.zeros((num_fine, 2))
    for seg_idx in range(n - 1):
        mask = seg_indices == seg_idx
        if not np.any(mask):
            continue

        trajectory[mask] = quintic_hermite_segment(
            points[seg_idx], points[seg_idx + 1],
            velocities[seg_idx], velocities[seg_idx + 1],
            accelerations[seg_idx], accelerations[seg_idx + 1],
            local_tau[mask]
        )

    return trajectory, tau_fine


def generate_minimum_jerk_trajectory(
    via_points: np.ndarray,
    num_points: int = 128,
    include_midpoints: bool = True,
    offset_std: float = 0.0
) -> np.ndarray:
    """
    Generate C² continuous minimum jerk trajectory through via-points.

    Implements Quinn & Zhai (2018) approach:
    - Optional midpoints between key centers
    - Gaussian offset noise on key positions
    - C² continuity (position, velocity, acceleration) at all via-points
    - Zero velocity and acceleration only at start and end

    Time values are derived from the minimum jerk velocity profile by inverting
    the s(τ) relationship. This ensures time is consistent with how real gestures
    are processed (arc-length resampled with interpolated timestamps).

    Args:
        via_points: Array of shape (n, 2) with key center (x, y) positions
        num_points: Total points in output trajectory
        include_midpoints: If True, add midpoints between consecutive keys
        offset_std: Standard deviation of Gaussian noise on key positions

    Returns:
        Array of shape (num_points, 3) with (x, y, t) trajectory
        where t is derived from the minimum jerk velocity profile
    """
    n = len(via_points)
    if n < 2:
        # Single point or empty - return with uniform time
        xy = np.tile(via_points[0] if n == 1 else [0, 0], (num_points, 1))
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        return np.hstack([xy, times]).astype(np.float32)

    # Apply offset noise to key centers (except first and last)
    points = via_points.copy().astype(float)
    if offset_std > 0 and n > 2:
        noise = np.random.normal(0, offset_std, (n - 2, 2))
        points[1:-1] += noise

    # Add midpoints between consecutive keys (Quinn & Zhai approach)
    if include_midpoints and n > 2:
        expanded = [points[0]]
        for i in range(n - 1):
            # Midpoint with slight random offset toward next key
            mid = (points[i] + points[i + 1]) / 2
            if offset_std > 0:
                # Add angular noise to midpoint
                direction = points[i + 1] - points[i]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                mid += perp * np.random.normal(0, offset_std * 0.5)
            expanded.append(mid)
            expanded.append(points[i + 1])
        points = np.array(expanded)

    n = len(points)

    if n == 2:
        # Simple case: single segment with zero velocity at endpoints
        # For minimum jerk, τ IS time, and position follows quintic profile
        tau = np.linspace(0, 1, num_points)
        s = minimum_jerk_quintic(tau)
        xy = points[0] + np.outer(s, points[1] - points[0])
        # Time = τ (the minimum jerk parameter)
        return np.column_stack([xy, tau]).astype(np.float32)

    # Compute velocities at interior points for C² continuity
    # Using Catmull-Rom style tangents scaled by segment length
    velocities = np.zeros((n, 2))

    for i in range(1, n - 1):
        # Tangent direction from neighboring points
        d_before = points[i] - points[i - 1]
        d_after = points[i + 1] - points[i]
        len_before = np.linalg.norm(d_before)
        len_after = np.linalg.norm(d_after)

        if len_before > 1e-6 and len_after > 1e-6:
            # Catmull-Rom tangent: average of normalized directions, scaled
            t_before = d_before / len_before
            t_after = d_after / len_after
            tangent = (t_before + t_after) / 2
            # Scale by harmonic mean of segment lengths for smooth speed
            scale = 2 * len_before * len_after / (len_before + len_after)
            velocities[i] = tangent * scale

    # Accelerations at interior points (approximate from velocity changes)
    accelerations = np.zeros((n, 2))
    # Keep zero at endpoints and interior (natural spline-like behavior)

    # Generate trajectory at fine τ resolution to compute s(τ) mapping
    # τ is the time parameter in minimum jerk - we need to recover it after arc-length sampling
    traj_fine, tau_fine = _generate_fine_trajectory_with_tau(
        points, velocities, accelerations, num_fine=1000
    )

    # Compute s(τ) - cumulative arc-length as function of τ
    ds = np.linalg.norm(np.diff(traj_fine, axis=0), axis=1)
    s_of_tau = np.concatenate([[0], np.cumsum(ds)])
    total_length = s_of_tau[-1]

    if total_length < 1e-6:
        # Degenerate case - all points at same location
        xy = np.tile(points[0], (num_points, 1))
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        return np.hstack([xy, times]).astype(np.float32)

    # Sample at uniform arc-lengths for spatial uniformity
    s_target = np.linspace(0, total_length, num_points)

    # Invert s(τ) to find τ for each target arc-length
    # This recovers the time value that corresponds to each spatial position
    tau_for_points = np.interp(s_target, s_of_tau, tau_fine)

    # Get (x, y) at these arc-lengths by interpolating the fine trajectory
    x = np.interp(s_target, s_of_tau, traj_fine[:, 0])
    y = np.interp(s_target, s_of_tau, traj_fine[:, 1])

    # Time IS τ (the minimum jerk parameter)
    # This ensures time reflects the velocity profile: slower at corners, faster on straights
    times = tau_for_points

    return np.column_stack([x, y, times]).astype(np.float32)


def generate_minimum_jerk_trajectory_fitted(
    via_points: np.ndarray,
    num_points: int = 128,
    include_midpoints: bool = True,
    key_offset_mean: Tuple[float, float] = (0.0, 0.0),
    key_offset_std: Tuple[float, float] = (0.02, 0.02),
    midpoint_angle_mean: float = 0.0,
    midpoint_angle_std: float = 0.1
) -> np.ndarray:
    """
    Generate minimum jerk trajectory using learned distributions (Quinn & Zhai 2018).

    This version uses separately learned distributions for:
    1. Key center offsets (x and y components independently)
    2. Midpoint angles (perpendicular deviation from key-to-key line)

    Args:
        via_points: Array of shape (n, 2) with key center (x, y) positions
        num_points: Total points in output trajectory
        include_midpoints: If True, add midpoints between consecutive keys
        key_offset_mean: (mean_x, mean_y) for key center offsets
        key_offset_std: (std_x, std_y) for key center offsets
        midpoint_angle_mean: Mean of midpoint angle distribution
        midpoint_angle_std: Std of midpoint angle distribution

    Returns:
        Array of shape (num_points, 3) with (x, y, t) trajectory
    """
    n = len(via_points)
    if n < 2:
        xy = np.tile(via_points[0] if n == 1 else [0, 0], (num_points, 1))
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        return np.hstack([xy, times]).astype(np.float32)

    # Apply learned offset distribution to key centers (except first and last)
    points = via_points.copy().astype(float)
    if n > 2:
        # Sample from learned key offset distribution
        offset_x = np.random.normal(key_offset_mean[0], key_offset_std[0], n - 2)
        offset_y = np.random.normal(key_offset_mean[1], key_offset_std[1], n - 2)
        points[1:-1, 0] += offset_x
        points[1:-1, 1] += offset_y

    # Add midpoints with learned angle distribution
    if include_midpoints and n > 2:
        expanded = [points[0]]
        for i in range(n - 1):
            # Theoretical midpoint
            mid = (points[i] + points[i + 1]) / 2

            # Apply learned midpoint angle distribution
            direction = points[i + 1] - points[i]
            seg_length = np.linalg.norm(direction)

            if seg_length > 1e-6:
                # Perpendicular direction (90 degrees counterclockwise)
                perp = np.array([-direction[1], direction[0]]) / seg_length

                # Sample angle from learned distribution
                # Angle is normalized by segment length, so multiply back
                angle = np.random.normal(midpoint_angle_mean, midpoint_angle_std)
                mid += perp * angle * seg_length

            expanded.append(mid)
            expanded.append(points[i + 1])
        points = np.array(expanded)

    n = len(points)

    if n == 2:
        tau = np.linspace(0, 1, num_points)
        s = minimum_jerk_quintic(tau)
        xy = points[0] + np.outer(s, points[1] - points[0])
        return np.column_stack([xy, tau]).astype(np.float32)

    # Compute velocities at interior points for C² continuity
    velocities = np.zeros((n, 2))
    for i in range(1, n - 1):
        d_before = points[i] - points[i - 1]
        d_after = points[i + 1] - points[i]
        len_before = np.linalg.norm(d_before)
        len_after = np.linalg.norm(d_after)

        if len_before > 1e-6 and len_after > 1e-6:
            t_before = d_before / len_before
            t_after = d_after / len_after
            tangent = (t_before + t_after) / 2
            scale = 2 * len_before * len_after / (len_before + len_after)
            velocities[i] = tangent * scale

    accelerations = np.zeros((n, 2))

    # Generate fine trajectory
    traj_fine, tau_fine = _generate_fine_trajectory_with_tau(
        points, velocities, accelerations, num_fine=1000
    )

    # Compute arc-length mapping
    ds = np.linalg.norm(np.diff(traj_fine, axis=0), axis=1)
    s_of_tau = np.concatenate([[0], np.cumsum(ds)])
    total_length = s_of_tau[-1]

    if total_length < 1e-6:
        xy = np.tile(points[0], (num_points, 1))
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        return np.hstack([xy, times]).astype(np.float32)

    # Sample at uniform arc-lengths
    s_target = np.linspace(0, total_length, num_points)
    tau_for_points = np.interp(s_target, s_of_tau, tau_fine)
    x = np.interp(s_target, s_of_tau, traj_fine[:, 0])
    y = np.interp(s_target, s_of_tau, traj_fine[:, 1])

    return np.column_stack([x, y, tau_for_points]).astype(np.float32)


class QWERTYKeyboard:
    """
    QWERTY keyboard layout for generating word prototypes.

    The keyboard coordinates are in canonical space where:
    - x: left-to-right position based on key layout
    - y: -1 is top row, 0 is middle row, +1 is bottom row (approximately)

    Gesture data should be normalized to this canonical space during loading.
    """

    def __init__(self, config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG):
        """
        Initialize keyboard layout.

        Args:
            config: Keyboard configuration
        """
        self.config = config
        self.key_centers = self._compute_key_centers()

    def _compute_key_centers(self) -> Dict[str, Tuple[float, float]]:
        """Compute the center coordinates for each key in canonical space."""
        key_centers = {}
        rows = self.config.rows
        row_offsets = self.config.row_offsets

        for row_idx, (row, offset) in enumerate(zip(rows, row_offsets)):
            num_keys = len(row)
            # Calculate y position: -1 for top row, 0 for middle, +1 for bottom
            y = -1 + (row_idx + 0.5) * (2.0 / len(rows))

            for key_idx, key in enumerate(row):
                # Calculate x position - keys span [-0.9, 0.9] with row offset
                row_span = 1.8 - offset  # Available width for this row
                row_start = -0.9 + offset / 2
                x = row_start + (key_idx + 0.5) * (row_span / num_keys)

                key_centers[key.lower()] = (x, y)

        return key_centers

    def get_key_center(self, letter: str) -> Optional[Tuple[float, float]]:
        """Get the center coordinates for a letter key."""
        return self.key_centers.get(letter.lower())

    def _get_key_positions(self, word: str) -> list:
        """Extract valid key positions for a word."""
        positions = []
        for letter in word.lower():
            center = self.get_key_center(letter)
            if center is not None:
                positions.append(center)
        return positions

    def _make_single_point_prototype(self, x: float, y: float, num_points: int) -> np.ndarray:
        """Create prototype for single-letter or same-position words."""
        prototype = np.zeros((num_points, 3), dtype=np.float32)
        prototype[:, 0] = x
        prototype[:, 1] = y
        prototype[:, 2] = np.linspace(0, 1, num_points)
        return prototype

    def _finalize_prototype(self, trajectory: np.ndarray, num_points: int) -> np.ndarray:
        """Pad/trim trajectory to exact length and add time dimension."""
        # Ensure exactly num_points
        if len(trajectory) > num_points:
            indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
            trajectory = trajectory[indices]
        elif len(trajectory) < num_points:
            padding = np.tile(trajectory[-1], (num_points - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])

        # Add time dimension
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        return np.hstack([trajectory, times]).astype(np.float32)

    def get_word_prototype(self, word: str, num_points: int = 128) -> np.ndarray:
        """
        Generate a word prototype: straight lines connecting letter centroids.

        Uses arc-length sampling to ensure uniform spatial distribution of points,
        matching how user gestures are resampled in data.py.

        Args:
            word: The target word
            num_points: Total number of points in the prototype (default 128)

        Returns:
            Array of shape (num_points, 3) with (x, y, t) coordinates.
            Time values are uniformly distributed.
        """
        key_positions = self._get_key_positions(word)

        if len(key_positions) < 2:
            if len(key_positions) == 1:
                return self._make_single_point_prototype(*key_positions[0], num_points)
            return np.zeros((num_points, 3), dtype=np.float32)

        key_positions = np.array(key_positions)
        k = len(key_positions)

        # Calculate segment lengths and total arc length
        segment_lengths = np.linalg.norm(np.diff(key_positions, axis=0), axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < 1e-6:
            # All keys at same position
            return self._make_single_point_prototype(*key_positions[0], num_points)

        # Sample at uniform arc-length intervals
        target_lengths = np.linspace(0, total_length, num_points)
        trajectory = np.zeros((num_points, 2), dtype=np.float32)

        for i, target_len in enumerate(target_lengths):
            # Find segment containing this target length
            seg_idx = np.searchsorted(cumulative_lengths, target_len, side='right') - 1
            seg_idx = max(0, min(seg_idx, k - 2))

            # Interpolate within segment
            seg_start_len = cumulative_lengths[seg_idx]
            seg_len = segment_lengths[seg_idx]

            if seg_len > 1e-6:
                t = (target_len - seg_start_len) / seg_len
            else:
                t = 0.0

            t = max(0.0, min(1.0, t))
            trajectory[i] = key_positions[seg_idx] + t * (key_positions[seg_idx + 1] - key_positions[seg_idx])

        return self._finalize_prototype(trajectory, num_points)

    def get_key_centers_for_word(self, word: str) -> np.ndarray:
        """
        Get the (x, y) coordinates of key centers for a word.

        Args:
            word: The target word

        Returns:
            Array of shape (n_keys, 2) with key center coordinates.
        """
        positions = self._get_key_positions(word)
        return np.array(positions) if positions else np.zeros((0, 2))

    def get_key_indices(self, word: str, num_points: int = 128) -> np.ndarray:
        """
        Get the indices in a prototype sequence where key centers are located.

        Uses arc-length sampling logic matching get_word_prototype to ensure
        indices correctly correspond to key positions.

        Args:
            word: The target word
            num_points: Total number of points in the prototype

        Returns:
            Array of indices where key centers appear in the prototype.
        """
        key_positions = self._get_key_positions(word)
        k = len(key_positions)

        if k == 0:
            return np.array([], dtype=int)
        if k == 1:
            return np.array([0], dtype=int)

        key_positions = np.array(key_positions)

        # Calculate cumulative arc lengths to each key
        segment_lengths = np.linalg.norm(np.diff(key_positions, axis=0), axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < 1e-6:
            return np.array([0], dtype=int)

        # Key at cumulative_lengths[i] maps to index based on arc-length sampling
        # target_lengths = np.linspace(0, total_length, num_points)
        # Index i corresponds to arc-length = i * total_length / (num_points - 1)
        # So key at arc-length L is at index = L * (num_points - 1) / total_length
        indices = np.round(cumulative_lengths * (num_points - 1) / total_length).astype(int)
        indices = np.clip(indices, 0, num_points - 1)

        return indices

    def get_minimum_jerk_trajectory(
        self,
        word: str,
        num_points: int = 128,
        include_midpoints: bool = True,
        offset_std: float = 0.0
    ) -> np.ndarray:
        """
        Generate minimum jerk trajectory for a word following Quinn & Zhai (2018).

        Features:
        - C² continuity through all via-points
        - Optional midpoints between keys
        - Optional Gaussian noise on key positions
        - Zero velocity/acceleration only at start and end
        - Time derived from velocity profile (slower at corners, faster on straights)

        Args:
            word: The target word
            num_points: Total points in trajectory
            include_midpoints: Add midpoints between consecutive keys
            offset_std: Std dev of Gaussian noise on key positions

        Returns:
            Array of shape (num_points, 3) with (x, y, t) coordinates
            where t is derived from the minimum jerk velocity profile
        """
        key_positions = self._get_key_positions(word)

        if len(key_positions) < 2:
            if len(key_positions) == 1:
                return self._make_single_point_prototype(*key_positions[0], num_points)
            return np.zeros((num_points, 3), dtype=np.float32)

        key_positions = np.array(key_positions)

        # Generate minimum jerk trajectory with proper time from velocity profile
        # Time is now included in the output (not added separately)
        return generate_minimum_jerk_trajectory(
            key_positions,
            num_points=num_points,
            include_midpoints=include_midpoints,
            offset_std=offset_std
        )
