"""
Keyboard layout utilities for generating word prototypes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .config import KeyboardConfig, DEFAULT_KEYBOARD_CONFIG


def _minimum_jerk_trajectory(start: np.ndarray, end: np.ndarray, n_points: int) -> np.ndarray:
    """
    Generate minimum jerk trajectory between two points.

    Uses the quintic polynomial that minimizes jerk (3rd derivative of position):
        x(t) = x0 + (x1 - x0) * (10*t^3 - 15*t^4 + 6*t^5)

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
    # Quintic polynomial for minimum jerk: s(t) = 10*t^3 - 15*t^4 + 6*t^5
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    trajectory = start + np.outer(s, (end - start))
    return trajectory


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
        word = word.lower()

        # Get key centers for valid letters
        key_positions = []
        for letter in word:
            center = self.get_key_center(letter)
            if center is not None:
                key_positions.append(center)

        if len(key_positions) < 2:
            # Single letter or no valid letters - return simple prototype
            if len(key_positions) == 1:
                x, y = key_positions[0]
                prototype = np.zeros((num_points, 3))
                prototype[:, 0] = x
                prototype[:, 1] = y
                prototype[:, 2] = np.linspace(0, 1, num_points)
                return prototype
            else:
                return np.zeros((num_points, 3))

        key_positions = np.array(key_positions)
        k = len(key_positions)  # Number of key centers

        # Distribute points: (n-k)/(k-1) "between" points per segment
        # Total: k key centers + (n-k) between points = n points
        points_per_segment = (num_points - k) // (k - 1)
        remaining_points = (num_points - k) % (k - 1)

        prototype_points = []

        for i in range(k - 1):
            start = key_positions[i]
            end = key_positions[i + 1]

            # Number of intermediate points for this segment
            n_between = points_per_segment + (1 if i < remaining_points else 0)

            # Add start point
            prototype_points.append(start)

            # Add intermediate points (uniformly distributed)
            for j in range(1, n_between + 1):
                t = j / (n_between + 1)
                point = start + t * (end - start)
                prototype_points.append(point)

        # Add final point
        prototype_points.append(key_positions[-1])

        # Ensure exactly num_points
        prototype_points = np.array(prototype_points[:num_points])

        # Pad if necessary
        if len(prototype_points) < num_points:
            last_point = prototype_points[-1]
            padding = np.tile(last_point, (num_points - len(prototype_points), 1))
            prototype_points = np.vstack([prototype_points, padding])

        # Add time dimension (uniformly distributed from 0 to 1)
        times = np.linspace(0, 1, num_points).reshape(-1, 1)

        # Combine (x, y, t)
        prototype = np.hstack([prototype_points, times])

        return prototype.astype(np.float32)

    def get_word_prototype_minimum_jerk(self, word: str, num_points: int = 128) -> np.ndarray:
        """
        Generate a minimum jerk word prototype connecting letter centroids.

        Unlike the straight-line prototype, this uses quintic polynomial
        trajectories that minimize jerk (third derivative of position),
        producing smoother curves between key positions.

        Args:
            word: The target word
            num_points: Total number of points in the prototype (default 128)

        Returns:
            Array of shape (num_points, 3) with (x, y, t) coordinates.
            Time values are uniformly distributed.
        """
        word = word.lower()

        # Get key centers for valid letters
        key_positions = []
        for letter in word:
            center = self.get_key_center(letter)
            if center is not None:
                key_positions.append(center)

        if len(key_positions) < 2:
            # Single letter or no valid letters - return simple prototype
            if len(key_positions) == 1:
                x, y = key_positions[0]
                prototype = np.zeros((num_points, 3))
                prototype[:, 0] = x
                prototype[:, 1] = y
                prototype[:, 2] = np.linspace(0, 1, num_points)
                return prototype.astype(np.float32)
            else:
                return np.zeros((num_points, 3), dtype=np.float32)

        key_positions = np.array(key_positions)
        k = len(key_positions)  # Number of key centers

        # Generate minimum jerk trajectory segments between consecutive keys
        # Distribute points proportionally to segment length
        segment_lengths = np.linalg.norm(np.diff(key_positions, axis=0), axis=1)
        total_length = segment_lengths.sum()

        if total_length < 1e-6:
            # All keys at same position
            prototype = np.zeros((num_points, 3))
            prototype[:, 0] = key_positions[0, 0]
            prototype[:, 1] = key_positions[0, 1]
            prototype[:, 2] = np.linspace(0, 1, num_points)
            return prototype.astype(np.float32)

        # Distribute points to segments proportionally to length
        # Ensure at least 2 points per segment
        segment_points = np.maximum(2, np.round(segment_lengths / total_length * num_points).astype(int))
        # Adjust to hit exact num_points
        diff = num_points - segment_points.sum()
        if diff > 0:
            # Add points to longest segments
            for _ in range(diff):
                idx = np.argmax(segment_lengths / segment_points)
                segment_points[idx] += 1
        elif diff < 0:
            # Remove points from longest segments (but keep min 2)
            for _ in range(-diff):
                valid = segment_points > 2
                if not valid.any():
                    break
                idx = np.argmax(np.where(valid, segment_lengths / segment_points, -np.inf))
                segment_points[idx] -= 1

        # Generate trajectories
        trajectory_points = []
        for i in range(k - 1):
            start = key_positions[i]
            end = key_positions[i + 1]
            n_pts = segment_points[i]

            # Generate minimum jerk segment
            segment = _minimum_jerk_trajectory(start, end, n_pts)

            # Skip first point to avoid duplication at junction (except for first segment)
            if i > 0:
                segment = segment[1:]

            trajectory_points.append(segment)

        trajectory = np.vstack(trajectory_points)

        # Ensure exactly num_points
        if len(trajectory) > num_points:
            # Uniformly sample to reduce
            indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
            trajectory = trajectory[indices]
        elif len(trajectory) < num_points:
            # Pad with last point
            padding = np.tile(trajectory[-1], (num_points - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])

        # Add uniform time dimension
        times = np.linspace(0, 1, num_points).reshape(-1, 1)
        prototype = np.hstack([trajectory, times])

        return prototype.astype(np.float32)

    def get_key_centers_for_word(self, word: str) -> np.ndarray:
        """
        Get the (x, y) coordinates of key centers for a word.

        Args:
            word: The target word

        Returns:
            Array of shape (n_keys, 2) with key center coordinates.
        """
        word = word.lower()
        centers = []
        for letter in word:
            center = self.get_key_center(letter)
            if center is not None:
                centers.append(center)
        return np.array(centers) if centers else np.zeros((0, 2))

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

    def visualize_keyboard(self) -> None:
        """Print keyboard layout with key centers (for debugging)."""
        print("QWERTY Keyboard Layout (normalized coordinates):")
        print("-" * 50)
        for row in self.config.rows:
            row_str = ""
            for key in row:
                x, y = self.key_centers[key]
                row_str += f"{key}({x:.2f},{y:.2f}) "
            print(row_str)
        print("-" * 50)


def create_word_prototypes_batch(
    words: List[str],
    keyboard: QWERTYKeyboard,
    num_points: int = 128
) -> np.ndarray:
    """
    Create word prototypes for a batch of words.

    Args:
        words: List of words
        keyboard: QWERTYKeyboard instance
        num_points: Number of points per prototype

    Returns:
        Array of shape (batch_size, num_points, 3)
    """
    prototypes = []
    for word in words:
        prototype = keyboard.get_word_prototype(word, num_points)
        prototypes.append(prototype)
    return np.stack(prototypes, axis=0)
