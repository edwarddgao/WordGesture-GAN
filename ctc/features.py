"""Feature extraction for gesture recognition.

Extracts rich features from raw gesture data used by the CTC model.
"""

from __future__ import annotations

import numpy as np

from shared import KeyboardLayout


class GestureFeatureExtractor:
    """Extract rich features from raw gesture data.

    Features include:
    - Base coordinates: x, y, dt (3 features)
    - Key proximity: Gaussian distance to each of 26 keys (26 features)
    - Velocity: vx, vy computed from position/time (2 features)
    - Approach angle: normalized movement direction (1 feature)
    - Acceleration: ax, ay computed from velocity/time (2 features)

    Total: up to 34 features per timestep depending on configuration.
    """

    def __init__(
        self,
        layout: KeyboardLayout | None = None,
        sigma: float = 0.3,
        use_key_proximity: bool = True,
        use_velocity: bool = True,
        use_approach_angle: bool = False,
        use_acceleration: bool = False,
    ):
        """Initialize the feature extractor.

        Args:
            layout: Keyboard layout for key positions. Defaults to QWERTY.
            sigma: Gaussian sigma for key proximity soft assignment.
            use_key_proximity: Whether to include 26-dim key proximity features.
            use_velocity: Whether to include 2-dim velocity features.
            use_approach_angle: Whether to include 1-dim approach angle feature.
            use_acceleration: Whether to include 2-dim acceleration features.
        """
        self.layout = layout or KeyboardLayout()
        self.sigma = sigma
        self.use_key_proximity = use_key_proximity
        self.use_velocity = use_velocity
        self.use_approach_angle = use_approach_angle
        self.use_acceleration = use_acceleration

        # Precompute key centers: (26, 2) array
        centers = self.layout.key_centers_normalized()
        self.key_order = sorted(centers.keys())  # a-z alphabetically
        self.key_centers = np.array(
            [centers[k] for k in self.key_order], dtype=np.float32
        )

    def __call__(self, gesture: np.ndarray) -> np.ndarray:
        """Extract features from a gesture.

        Args:
            gesture: (seq_len, 3) array with [x, y, dt] per timestep.

        Returns:
            features: (seq_len, n_features) enriched feature array.
        """
        features = [gesture.astype(np.float32)]  # Start with base (x, y, dt)

        if self.use_key_proximity:
            proximity = self._compute_key_proximity(gesture[:, :2])
            features.append(proximity)

        velocity = None
        if self.use_velocity or self.use_acceleration:
            velocity = self._compute_velocity(gesture)
            if self.use_velocity:
                features.append(velocity)

        if self.use_approach_angle:
            angle = self._compute_approach_angle(gesture)
            features.append(angle)

        if self.use_acceleration:
            acceleration = self._compute_acceleration(gesture, velocity)
            features.append(acceleration)

        return np.concatenate(features, axis=-1)

    def _compute_key_proximity(self, xy: np.ndarray) -> np.ndarray:
        """Compute Gaussian proximity to each key.

        Args:
            xy: (seq_len, 2) array of x, y coordinates.

        Returns:
            proximity: (seq_len, 26) array where proximity[t, k] is the
                Gaussian proximity of timestep t to key k.
        """
        # xy: (seq_len, 2), key_centers: (26, 2)
        # Compute distance from each point to each key
        diff = xy[:, None, :] - self.key_centers[None, :, :]  # (seq_len, 26, 2)
        dist_sq = np.sum(diff**2, axis=-1)  # (seq_len, 26)

        # Convert to Gaussian proximity
        proximity = np.exp(-dist_sq / (2 * self.sigma**2))
        return proximity.astype(np.float32)

    def _compute_velocity(self, gesture: np.ndarray) -> np.ndarray:
        """Compute velocity from position and time.

        Args:
            gesture: (seq_len, 3) array with [x, y, dt].

        Returns:
            velocity: (seq_len, 2) array with [vx, vy], normalized to [-1, 1].
        """
        # Compute position differences
        dx = np.diff(gesture[:, 0], prepend=gesture[0, 0])
        dy = np.diff(gesture[:, 1], prepend=gesture[0, 1])

        # Get time deltas, avoiding division by zero
        dt = np.maximum(gesture[:, 2], 1e-6)

        vx = dx / dt
        vy = dy / dt

        # Normalize to [-1, 1] range using 99th percentile values (~12)
        # This makes velocity scale compatible with other features (x,y,dt,key_proximity)
        velocity_scale = 12.0
        vx = np.clip(vx / velocity_scale, -1.0, 1.0)
        vy = np.clip(vy / velocity_scale, -1.0, 1.0)

        return np.stack([vx, vy], axis=-1).astype(np.float32)

    def _compute_approach_angle(self, gesture: np.ndarray) -> np.ndarray:
        """Compute normalized movement direction angle.

        Args:
            gesture: (seq_len, 3) array with [x, y, dt].

        Returns:
            angle: (seq_len, 1) array with angle normalized to [-1, 1].
        """
        # Compute position differences
        dx = np.diff(gesture[:, 0], prepend=gesture[0, 0])
        dy = np.diff(gesture[:, 1], prepend=gesture[0, 1])

        # Compute angle using atan2, normalized by pi to get [-1, 1]
        angle = np.arctan2(dy, dx) / np.pi

        # First timestep has no movement, set to 0
        angle[0] = 0.0

        return angle[:, None].astype(np.float32)

    def _compute_acceleration(
        self, gesture: np.ndarray, velocity: np.ndarray | None
    ) -> np.ndarray:
        """Compute acceleration from velocity and time.

        Args:
            gesture: (seq_len, 3) array with [x, y, dt].
            velocity: (seq_len, 2) array with [vx, vy], or None to compute.

        Returns:
            acceleration: (seq_len, 2) array with [ax, ay], normalized to [-1, 1].
        """
        if velocity is None:
            velocity = self._compute_velocity(gesture)

        # Compute velocity differences
        dvx = np.diff(velocity[:, 0], prepend=velocity[0, 0])
        dvy = np.diff(velocity[:, 1], prepend=velocity[0, 1])

        # Get time deltas, avoiding division by zero
        dt = np.maximum(gesture[:, 2], 1e-6)

        ax = dvx / dt
        ay = dvy / dt

        # Normalize to [-1, 1] range
        # Acceleration scale derived from velocity scale (~12) and typical dt (~0.01)
        accel_scale = 50.0
        ax = np.clip(ax / accel_scale, -1.0, 1.0)
        ay = np.clip(ay / accel_scale, -1.0, 1.0)

        return np.stack([ax, ay], axis=-1).astype(np.float32)

    @property
    def n_features(self) -> int:
        """Total number of features per timestep."""
        n = 3  # x, y, dt
        if self.use_key_proximity:
            n += 26
        if self.use_velocity:
            n += 2
        if self.use_approach_angle:
            n += 1
        if self.use_acceleration:
            n += 2
        return n
