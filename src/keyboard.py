"""
Keyboard layout utilities for generating word prototypes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .config import KeyboardConfig, DEFAULT_KEYBOARD_CONFIG


class QWERTYKeyboard:
    """
    QWERTY keyboard layout for generating word prototypes.

    The keyboard coordinates are normalized to [-1, 1] where:
    - x: -1 is left edge, 1 is right edge
    - y: -1 is top edge, 1 is bottom edge
    """

    def __init__(self, config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG):
        self.config = config
        self.key_centers = self._compute_key_centers()

    def _compute_key_centers(self) -> Dict[str, Tuple[float, float]]:
        """Compute the center coordinates for each key on the keyboard."""
        key_centers = {}
        rows = self.config.rows
        row_offsets = self.config.row_offsets

        for row_idx, (row, offset) in enumerate(zip(rows, row_offsets)):
            num_keys = len(row)
            # Calculate y position (normalized to [-1, 1])
            y = -1 + (row_idx + 0.5) * (2.0 / len(rows))

            for key_idx, key in enumerate(row):
                # Calculate x position with offset (normalized to [-1, 1])
                # Account for varying row lengths
                row_width = num_keys * self.config.key_width
                start_x = -1 + offset + (1 - row_width / 2)
                x = start_x + (key_idx + 0.5) * self.config.key_width * 2

                # Clamp to [-1, 1]
                x = max(-1, min(1, x))
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

        # Distribute points: (n-k)/(k-1) "between" points per segment
        points_per_segment = (num_points - k) // (k - 1)
        remaining_points = (num_points - k) % (k - 1)

        trajectory = []
        for i in range(k - 1):
            start, end = key_positions[i], key_positions[i + 1]
            n_between = points_per_segment + (1 if i < remaining_points else 0)

            trajectory.append(start)
            for j in range(1, n_between + 1):
                t = j / (n_between + 1)
                trajectory.append(start + t * (end - start))

        trajectory.append(key_positions[-1])
        return self._finalize_prototype(np.array(trajectory), num_points)

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

        Uses the same distribution logic as get_word_prototype to ensure
        indices match the prototype point positions.

        Args:
            word: The target word
            num_points: Total number of points in the prototype

        Returns:
            Array of indices where key centers appear in the prototype.
        """
        word = word.lower()

        # Count valid letters
        valid_letters = [l for l in word if self.get_key_center(l) is not None]
        k = len(valid_letters)

        if k == 0:
            return np.array([], dtype=int)
        if k == 1:
            return np.array([0], dtype=int)

        # Same distribution logic as get_word_prototype
        points_per_segment = (num_points - k) // (k - 1)
        remaining_points = (num_points - k) % (k - 1)

        indices = []
        current_idx = 0

        for i in range(k - 1):
            # Key center is at current position
            indices.append(current_idx)

            # Number of intermediate points for this segment
            n_between = points_per_segment + (1 if i < remaining_points else 0)

            # Move to next key: 1 (for current key) + n_between (intermediate points)
            current_idx += 1 + n_between

        # Final key
        indices.append(min(current_idx, num_points - 1))

        return np.array(indices, dtype=int)
