"""Shared feature extraction for gesture recognition.

PyTorch-based feature extractor that can be used by both ctc and wg_gan models.
Supports differentiable feature extraction for end-to-end training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .keyboard import KeyboardLayout


class GestureFeatureExtractor(nn.Module):
    """PyTorch-based feature extractor for differentiable training.

    Features include:
    - Base coordinates: x, y, dt (3 features)
    - Key proximity: Gaussian distance to each of 26 keys (26 features)
    - Velocity: vx, vy computed from position/time (2 features)

    Total: 31 features per timestep (when all features enabled).
    """

    def __init__(
        self,
        layout: KeyboardLayout | None = None,
        sigma: float = 0.3,
        use_key_proximity: bool = True,
        use_velocity: bool = True,
    ):
        """Initialize the feature extractor.

        Args:
            layout: Keyboard layout for key positions. Defaults to QWERTY.
            sigma: Gaussian sigma for key proximity soft assignment.
            use_key_proximity: Whether to include 26-dim key proximity features.
            use_velocity: Whether to include 2-dim velocity features.
        """
        super().__init__()
        self.sigma = sigma
        self.use_key_proximity = use_key_proximity
        self.use_velocity = use_velocity

        # Precompute key centers and register as buffer
        layout = layout or KeyboardLayout()
        centers = layout.key_centers_normalized()
        key_order = sorted(centers.keys())  # a-z alphabetically
        key_centers = torch.tensor(
            [centers[k] for k in key_order], dtype=torch.float32
        )
        self.register_buffer("key_centers", key_centers)  # (26, 2)

    def forward(self, gesture: torch.Tensor) -> torch.Tensor:
        """Extract features from gesture batch.

        Args:
            gesture: (B, T, 3) tensor with [x, y, dt] per timestep.

        Returns:
            features: (B, T, n_features) enriched feature tensor.
        """
        features = [gesture]  # Start with base (x, y, dt)

        if self.use_key_proximity:
            proximity = self._compute_key_proximity(gesture[..., :2])
            features.append(proximity)

        if self.use_velocity:
            velocity = self._compute_velocity(gesture)
            features.append(velocity)

        return torch.cat(features, dim=-1)

    def _compute_key_proximity(self, xy: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian proximity to each key.

        Args:
            xy: (B, T, 2) tensor of x, y coordinates.

        Returns:
            proximity: (B, T, 26) tensor where proximity[b, t, k] is the
                Gaussian proximity of timestep t to key k.
        """
        # xy: (B, T, 2), key_centers: (26, 2)
        # Compute distance from each point to each key
        diff = xy.unsqueeze(-2) - self.key_centers  # (B, T, 26, 2)
        dist_sq = (diff**2).sum(dim=-1)  # (B, T, 26)

        # Convert to Gaussian proximity
        proximity = torch.exp(-dist_sq / (2 * self.sigma**2))
        return proximity

    def _compute_velocity(self, gesture: torch.Tensor) -> torch.Tensor:
        """Compute velocity from position and time.

        Args:
            gesture: (B, T, 3) tensor with [x, y, dt].

        Returns:
            velocity: (B, T, 2) tensor with [vx, vy], normalized to [-1, 1].
        """
        # Compute position differences
        dx = torch.diff(gesture[..., 0], dim=-1, prepend=gesture[..., :1, 0])
        dy = torch.diff(gesture[..., 1], dim=-1, prepend=gesture[..., :1, 1])

        # Get time deltas, avoiding division by zero
        dt = gesture[..., 2].clamp(min=1e-6)

        vx = dx / dt
        vy = dy / dt

        # Normalize to [-1, 1] range using 99th percentile values (~12)
        velocity_scale = 12.0
        vx = (vx / velocity_scale).clamp(-1.0, 1.0)
        vy = (vy / velocity_scale).clamp(-1.0, 1.0)

        return torch.stack([vx, vy], dim=-1)

    @property
    def n_features(self) -> int:
        """Total number of features per timestep."""
        n = 3  # x, y, dt
        if self.use_key_proximity:
            n += 26
        if self.use_velocity:
            n += 2
        return n


class NumpyFeatureExtractor:
    """NumPy wrapper around PyTorch feature extractor for dataset usage.

    This wrapper allows using the same feature extraction logic in contexts
    where numpy arrays are needed (e.g., dataset __getitem__).
    """

    def __init__(
        self,
        layout: KeyboardLayout | None = None,
        sigma: float = 0.3,
        use_key_proximity: bool = True,
        use_velocity: bool = True,
    ):
        """Initialize the numpy feature extractor.

        Args:
            layout: Keyboard layout for key positions. Defaults to QWERTY.
            sigma: Gaussian sigma for key proximity soft assignment.
            use_key_proximity: Whether to include 26-dim key proximity features.
            use_velocity: Whether to include 2-dim velocity features.
        """
        self._extractor = GestureFeatureExtractor(
            layout=layout,
            sigma=sigma,
            use_key_proximity=use_key_proximity,
            use_velocity=use_velocity,
        )

    def __call__(self, gesture: np.ndarray) -> np.ndarray:
        """Extract features from a gesture.

        Args:
            gesture: (seq_len, 3) array with [x, y, dt] per timestep.

        Returns:
            features: (seq_len, n_features) enriched feature array.
        """
        gesture_tensor = torch.from_numpy(gesture.astype(np.float32))
        # Add batch dimension, extract features, remove batch dimension
        features_tensor = self._extractor(gesture_tensor.unsqueeze(0)).squeeze(0)
        return features_tensor.numpy()

    @property
    def n_features(self) -> int:
        """Total number of features per timestep."""
        return self._extractor.n_features
