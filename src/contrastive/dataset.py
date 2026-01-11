"""
Dataset and batch sampler for contrastive gesture learning.

Key feature: Each batch contains N words with K gestures each,
ensuring positive pairs exist for contrastive loss computation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from typing import Dict, List, Tuple, Iterator, Optional
from collections import defaultdict
import random

from .model import ContrastiveConfig, DEFAULT_CONTRASTIVE_CONFIG
from src.shared.keyboard import QWERTYKeyboard


def augment_with_minimum_jerk(
    gestures_by_word: Dict[str, List[np.ndarray]],
    keyboard: QWERTYKeyboard,
    num_augmentations: int = 2,
    offset_std: float = 0.02
) -> Dict[str, List[np.ndarray]]:
    """
    Add minimum jerk trajectories as synthetic positive examples.

    Following Quinn & Zhai (2018), generates smooth trajectories through
    key centers with optional Gaussian noise for variation.

    Args:
        gestures_by_word: Dictionary mapping word -> list of gesture arrays
        keyboard: QWERTYKeyboard instance for generating trajectories
        num_augmentations: Number of min jerk samples to add per word
        offset_std: Standard deviation of Gaussian noise on key positions

    Returns:
        Augmented dictionary with additional min jerk trajectories
    """
    augmented = {}
    for word, gestures in gestures_by_word.items():
        augmented[word] = list(gestures)  # Copy original gestures
        for _ in range(num_augmentations):
            traj = keyboard.get_minimum_jerk_trajectory(
                word,
                num_points=128,
                include_midpoints=True,
                offset_std=offset_std
            )
            augmented[word].append(traj)
    return augmented


class ContrastiveGestureDataset(Dataset):
    """
    Dataset that groups gestures by word for contrastive learning.

    Each sample returns a gesture tensor and its word label.
    The batch sampler ensures each batch has balanced word representation.
    """

    def __init__(
        self,
        gestures_by_word: Dict[str, List[np.ndarray]],
        min_gestures_per_word: int = 2
    ):
        """
        Args:
            gestures_by_word: Dictionary mapping word -> list of gesture arrays
            min_gestures_per_word: Minimum gestures required per word (for positive pairs)
        """
        # Filter words with enough gestures
        self.words = []
        self.gestures = []
        self.word_labels = []

        # Build mapping from word to indices for sampling
        self.word_to_indices: Dict[str, List[int]] = defaultdict(list)

        idx = 0
        for word, gesture_list in gestures_by_word.items():
            if len(gesture_list) >= min_gestures_per_word:
                for gesture in gesture_list:
                    self.gestures.append(gesture)
                    self.words.append(word)
                    self.word_to_indices[word].append(idx)
                    idx += 1

        # Create integer labels for words
        self.unique_words = list(self.word_to_indices.keys())
        self.word_to_label = {word: i for i, word in enumerate(self.unique_words)}
        self.word_labels = [self.word_to_label[w] for w in self.words]

        print(f"ContrastiveGestureDataset: {len(self.gestures)} gestures from {len(self.unique_words)} words")

    def __len__(self) -> int:
        return len(self.gestures)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single gesture sample.

        Returns:
            Tuple of (gesture_tensor, label_int, word_str)
        """
        gesture = torch.FloatTensor(self.gestures[idx])
        label = self.word_labels[idx]
        word = self.words[idx]
        return gesture, label, word

    def get_word_indices(self, word: str) -> List[int]:
        """Get all indices for a specific word."""
        return self.word_to_indices.get(word, [])

    def get_num_words(self) -> int:
        """Get number of unique words."""
        return len(self.unique_words)


class ContrastiveBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that creates batches with N words × K gestures.

    Each batch contains exactly batch_words unique words, with
    gestures_per_word samples from each word. This ensures
    positive pairs exist for contrastive learning.
    """

    def __init__(
        self,
        dataset: ContrastiveGestureDataset,
        batch_words: int = 32,
        gestures_per_word: int = 2,
        drop_last: bool = True
    ):
        """
        Args:
            dataset: ContrastiveGestureDataset instance
            batch_words: Number of unique words per batch
            gestures_per_word: Number of gestures sampled per word
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_words = batch_words
        self.gestures_per_word = gestures_per_word
        self.drop_last = drop_last

        # Only use words with enough samples
        self.eligible_words = [
            word for word in dataset.unique_words
            if len(dataset.get_word_indices(word)) >= gestures_per_word
        ]

        if len(self.eligible_words) < batch_words:
            raise ValueError(
                f"Not enough words with >= {gestures_per_word} gestures. "
                f"Have {len(self.eligible_words)}, need {batch_words}"
            )

        # Calculate number of batches (each word appears once per epoch)
        self.batches_per_epoch = len(self.eligible_words) // batch_words
        if not drop_last and len(self.eligible_words) % batch_words != 0:
            self.batches_per_epoch += 1

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches for one epoch."""
        # Shuffle words for each epoch
        words = self.eligible_words.copy()
        random.shuffle(words)

        for batch_idx in range(self.batches_per_epoch):
            start = batch_idx * self.batch_words
            end = start + self.batch_words

            if end > len(words):
                if self.drop_last:
                    break
                end = len(words)

            batch_words = words[start:end]
            batch_indices = []

            for word in batch_words:
                word_indices = self.dataset.get_word_indices(word)
                # Sample K gestures from this word
                sampled = random.sample(word_indices, min(self.gestures_per_word, len(word_indices)))
                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self) -> int:
        return self.batches_per_epoch


def create_contrastive_datasets(
    gestures_by_word: Dict[str, List[np.ndarray]],
    train_ratio: float = 0.8,
    min_gestures_per_word: int = 2,
    seed: int = 42,
    augment_min_jerk: bool = False,
    keyboard: Optional[QWERTYKeyboard] = None,
    min_jerk_augmentations: int = 2,
    min_jerk_noise: float = 0.02
) -> Tuple[ContrastiveGestureDataset, ContrastiveGestureDataset]:
    """
    Create train and test datasets for contrastive learning.

    Words are split (not individual gestures) to ensure no word overlap.

    Args:
        gestures_by_word: Dictionary mapping word -> list of gesture arrays
        train_ratio: Fraction of words for training
        min_gestures_per_word: Minimum gestures per word to include
        seed: Random seed
        augment_min_jerk: If True, add minimum jerk trajectories to training set
        keyboard: QWERTYKeyboard instance (required if augment_min_jerk=True)
        min_jerk_augmentations: Number of min jerk samples per word
        min_jerk_noise: Std dev of Gaussian noise on key positions

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter words with enough gestures
    eligible_words = [
        word for word, gestures in gestures_by_word.items()
        if len(gestures) >= min_gestures_per_word
    ]

    random.shuffle(eligible_words)

    split_idx = int(len(eligible_words) * train_ratio)
    train_words = set(eligible_words[:split_idx])
    test_words = set(eligible_words[split_idx:])

    print(f"Train words: {len(train_words)}, Test words: {len(test_words)}")

    # Split gestures by word
    train_gestures_by_word = {
        word: gestures for word, gestures in gestures_by_word.items()
        if word in train_words
    }
    test_gestures_by_word = {
        word: gestures for word, gestures in gestures_by_word.items()
        if word in test_words
    }

    # Apply minimum jerk augmentation to training set only
    if augment_min_jerk:
        if keyboard is None:
            raise ValueError("keyboard is required when augment_min_jerk=True")
        print(f"Augmenting training set with {min_jerk_augmentations} min jerk trajectories per word (noise={min_jerk_noise})")
        train_gestures_by_word = augment_with_minimum_jerk(
            train_gestures_by_word,
            keyboard,
            num_augmentations=min_jerk_augmentations,
            offset_std=min_jerk_noise
        )

    train_dataset = ContrastiveGestureDataset(train_gestures_by_word, min_gestures_per_word)
    test_dataset = ContrastiveGestureDataset(test_gestures_by_word, min_gestures_per_word)

    return train_dataset, test_dataset


def create_contrastive_data_loader(
    dataset: ContrastiveGestureDataset,
    config: ContrastiveConfig = DEFAULT_CONTRASTIVE_CONFIG,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create data loader with contrastive batch sampling.

    Args:
        dataset: ContrastiveGestureDataset instance
        config: ContrastiveConfig with batch_words and gestures_per_word
        shuffle: Whether to shuffle (only affects word order, not batch sampling)
        num_workers: Number of data loading workers

    Returns:
        DataLoader with custom batch sampler
    """
    if shuffle:
        batch_sampler = ContrastiveBatchSampler(
            dataset,
            batch_words=config.batch_words,
            gestures_per_word=config.gestures_per_word,
            drop_last=True
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=contrastive_collate_fn
        )
    else:
        # For evaluation, use standard batching (all gestures)
        return DataLoader(
            dataset,
            batch_size=config.batch_words * config.gestures_per_word,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=contrastive_collate_fn
        )


def contrastive_collate_fn(
    batch: List[Tuple[torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function for contrastive batches.

    Args:
        batch: List of (gesture, label, word) tuples

    Returns:
        Tuple of (gestures_batch, labels_batch, words_list)
    """
    gestures = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    words = [item[2] for item in batch]

    return gestures, labels, words
