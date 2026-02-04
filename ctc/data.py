"""Dataset for CTC gesture-to-character training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared import KeyboardLayout, load_dataset

from .features import GestureFeatureExtractor


class CTCGestureDataset(Dataset):
    """Dataset for CTC training with gesture-to-character decoding.

    Each sample returns:
        - features: (seq_len, n_features) enriched gesture features
        - target: (target_len,) character indices
        - input_length: int, length of input sequence
        - target_length: int, length of target sequence
    """

    BLANK_IDX = 0  # CTC blank token
    CHAR_OFFSET = 1  # Characters start at index 1 (a=1, b=2, ..., z=26)

    def __init__(
        self,
        npz_path: Path,
        n_points: int = 128,
        feature_extractor: GestureFeatureExtractor | None = None,
        augment: bool = False,
        noise_std: float = 0.01,
    ):
        """Initialize the CTC dataset.

        Args:
            npz_path: Path to .npz file with gestures and words.
            n_points: Number of points per gesture (should match data).
            feature_extractor: Feature extractor instance. Creates default if None.
            augment: Whether to apply data augmentation.
            noise_std: Standard deviation of Gaussian noise for augmentation.
        """
        gestures, words = load_dataset(npz_path)
        self.n_points = n_points
        self.feature_extractor = feature_extractor or GestureFeatureExtractor(
            KeyboardLayout()
        )
        self.augment = augment
        self.noise_std = noise_std

        # Store data
        self.gestures = gestures
        self.words = words

        # Character mapping: a=1, b=2, ..., z=26 (0=blank)
        self.char_to_idx = {
            chr(ord("a") + i): i + self.CHAR_OFFSET for i in range(26)
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def word_to_target(self, word: str) -> List[int]:
        """Convert word to character index sequence.

        Args:
            word: Input word string.

        Returns:
            List of character indices.
        """
        return [
            self.char_to_idx[c] for c in word.lower() if c in self.char_to_idx
        ]

    def target_to_word(self, target: List[int]) -> str:
        """Convert character index sequence to word.

        Args:
            target: List of character indices.

        Returns:
            Decoded word string.
        """
        return "".join(
            self.idx_to_char.get(idx, "") for idx in target if idx != self.BLANK_IDX
        )

    def __len__(self) -> int:
        return len(self.gestures)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Get a single sample.

        Returns:
            Tuple of (features, target, input_length, target_length).
        """
        gesture = self.gestures[idx].copy()
        word = self.words[idx]

        # Apply augmentation (noise on x, y coordinates)
        if self.augment:
            noise = np.random.randn(*gesture[:, :2].shape).astype(np.float32)
            gesture[:, :2] += noise * self.noise_std

        # Extract features
        features = self.feature_extractor(gesture)

        # Convert word to target sequence
        target = self.word_to_target(word)

        return (
            torch.from_numpy(features),  # (seq_len, n_features)
            torch.tensor(target, dtype=torch.long),  # (target_len,)
            self.n_points,  # input_length (fixed)
            len(target),  # target_length
        )


def ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for variable-length targets.

    Args:
        batch: List of (features, target, input_length, target_length) tuples.

    Returns:
        Tuple of:
            - features: (batch, seq_len, n_features)
            - targets: (batch, max_target_len) padded target sequences
            - input_lengths: (batch,) input sequence lengths
            - target_lengths: (batch,) target sequence lengths
    """
    features, targets, input_lengths, target_lengths = zip(*batch)

    # Stack features (all same length due to fixed n_points)
    features = torch.stack(features, dim=0)  # (B, seq_len, n_features)

    # Pad targets to max length in batch
    max_target_len = max(target_lengths)
    batch_size = len(batch)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, target in enumerate(targets):
        padded_targets[i, : len(target)] = target

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return features, padded_targets, input_lengths, target_lengths
