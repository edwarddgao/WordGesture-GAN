"""Dataset for contrastive learning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared import KeyboardLayout, build_word_prototype, load_dataset


def _path_length(points: np.ndarray) -> float:
    """Calculate total Euclidean path length."""
    return float(np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2).sum())


def filter_by_path_length(
    gestures: np.ndarray,
    words: List[str],
    n_points: int,
    layout: KeyboardLayout,
    path_ratio_range: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[np.ndarray, List[str]]:
    """Filter out gestures whose path length doesn't match the expected prototype.

    Args:
        gestures: (N, n_points, 3) array of gestures
        words: List of N word labels
        n_points: Number of points per gesture
        layout: Keyboard layout for building prototypes
        path_ratio_range: (min_ratio, max_ratio) for valid gesture/prototype length ratio

    Returns:
        Filtered (gestures, words) tuple
    """
    min_ratio, max_ratio = path_ratio_range
    expected_path_cache: Dict[str, float] = {}
    valid_indices = []

    for i, (gesture, word) in enumerate(zip(gestures, words)):
        gesture_path = _path_length(gesture[:, :2])
        if word not in expected_path_cache:
            proto = build_word_prototype(word, n_points, layout)[:, :2]
            expected_path_cache[word] = _path_length(proto)
        expected_path = expected_path_cache[word]
        if expected_path > 0:
            ratio = gesture_path / expected_path
            if min_ratio <= ratio <= max_ratio:
                valid_indices.append(i)

    return gestures[valid_indices], [words[i] for i in valid_indices]


class ContrastiveGestureDataset(Dataset):
    """Dataset that returns (gesture, prototype, word_idx) tuples.

    Gestures have shape (n_points, 3) with (x, y, dt).
    Prototypes have shape (n_points, 2) with (x, y) only - no timing info.
    """

    def __init__(
        self,
        npz_path: Path,
        n_points: int = 128,
        layout: KeyboardLayout | None = None,
        augment: bool = False,
        noise_std: float = 0.01,
        apply_path_filter: bool = True,
        path_ratio_range: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        gestures, words = load_dataset(npz_path)
        self.n_points = n_points
        self.layout = layout or KeyboardLayout()
        self.augment = augment
        self.noise_std = noise_std
        self._prototype_cache: Dict[str, np.ndarray] = {}

        if apply_path_filter:
            gestures, words = filter_by_path_length(
                gestures, words, n_points, self.layout, path_ratio_range
            )

        self.gestures = gestures
        self.words = words

        # Build word-to-index mapping
        self.unique_words = sorted(set(words))
        self.word_to_idx = {w: i for i, w in enumerate(self.unique_words)}

    def __len__(self) -> int:
        return len(self.gestures)

    @property
    def vocab_size(self) -> int:
        return len(self.unique_words)

    def _get_prototype(self, word: str) -> np.ndarray:
        """Get or build prototype for a word."""
        if word not in self._prototype_cache:
            proto = build_word_prototype(word, self.n_points, self.layout)
            self._prototype_cache[word] = proto
        return self._prototype_cache[word]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        gesture = self.gestures[idx].copy()
        word = self.words[idx]

        # Optional augmentation: add noise to gesture coordinates
        if self.augment:
            gesture[:, :2] += np.random.randn(*gesture[:, :2].shape).astype(np.float32) * self.noise_std

        # Get prototype (x, y only - drop dt channel)
        prototype = self._get_prototype(word)[:, :2]

        word_idx = self.word_to_idx[word]

        return (
            torch.from_numpy(gesture),
            torch.from_numpy(prototype),
            word_idx,
        )

    def get_word(self, idx: int) -> str:
        """Get the word label for a sample."""
        return self.words[idx]
