"""
Data loading and preprocessing for word-gesture dataset.
"""

import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from .keyboard import QWERTYKeyboard
from .config import TrainingConfig, ModelConfig, DEFAULT_TRAINING_CONFIG, DEFAULT_MODEL_CONFIG


class GestureDataset(Dataset):
    """
    Dataset for word-gesture data.

    Each sample contains:
    - gesture: (seq_length, 3) array of (x, y, t) coordinates
    - prototype: (seq_length, 3) array of word prototype
    - word: the target word string
    """

    def __init__(
        self,
        gestures: List[np.ndarray],
        prototypes: List[np.ndarray],
        words: List[str],
    ):
        """
        Args:
            gestures: List of gesture arrays, each (seq_length, 3)
            prototypes: List of prototype arrays, each (seq_length, 3)
            words: List of corresponding words
        """
        self.gestures = gestures
        self.prototypes = prototypes
        self.words = words

    def __len__(self) -> int:
        return len(self.gestures)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gesture = torch.FloatTensor(self.gestures[idx])
        prototype = torch.FloatTensor(self.prototypes[idx])
        return {
            'gesture': gesture,
            'prototype': prototype,
            'word': self.words[idx]
        }


def parse_log_file(log_content: str) -> Dict[str, List[Dict]]:
    """
    Parse a swipe log file and extract gestures grouped by word.

    Args:
        log_content: Content of the log file

    Returns:
        Dictionary mapping word -> list of gesture point dictionaries
    """
    gestures_by_word = defaultdict(list)
    current_word = None
    current_gesture = []

    lines = log_content.strip().split('\n')

    for line in lines[1:]:  # Skip header
        try:
            parts = line.split()
            if len(parts) < 12:
                continue

            event = parts[4]
            x = float(parts[5])
            y = float(parts[6])
            timestamp = int(parts[1])
            word = parts[10]
            is_err = int(parts[11])

            # Skip error gestures
            if is_err == 1:
                continue

            # Skip single-letter words
            if len(word) <= 1:
                continue

            if event == 'touchstart':
                current_word = word.lower()
                current_gesture = [{
                    'x': x, 'y': y, 't': timestamp,
                    'keyb_width': float(parts[2]),
                    'keyb_height': float(parts[3])
                }]
            elif event == 'touchmove' and current_word:
                current_gesture.append({
                    'x': x, 'y': y, 't': timestamp,
                    'keyb_width': float(parts[2]),
                    'keyb_height': float(parts[3])
                })
            elif event == 'touchend' and current_word and current_gesture:
                current_gesture.append({
                    'x': x, 'y': y, 't': timestamp,
                    'keyb_width': float(parts[2]),
                    'keyb_height': float(parts[3])
                })
                if len(current_gesture) >= 3:  # Minimum gesture length
                    gestures_by_word[current_word].append(current_gesture)
                current_word = None
                current_gesture = []

        except (ValueError, IndexError):
            continue

    return dict(gestures_by_word)


def normalize_gesture(
    gesture_points: List[Dict],
    seq_length: int = 128
) -> np.ndarray:
    """
    Normalize and resample a gesture to fixed length.

    Coordinates are normalized to [-1, 1].
    Timestamps are converted to time deltas in seconds.

    Args:
        gesture_points: List of point dictionaries with x, y, t, keyb_width, keyb_height
        seq_length: Target sequence length

    Returns:
        Array of shape (seq_length, 3) with (x, y, t) values
    """
    if len(gesture_points) < 2:
        return np.zeros((seq_length, 3), dtype=np.float32)

    # Extract coordinates and normalize
    keyb_width = gesture_points[0]['keyb_width']
    keyb_height = gesture_points[0]['keyb_height']

    points = []
    for p in gesture_points:
        # Normalize x to [-1, 1]
        x = (p['x'] / keyb_width) * 2 - 1
        # Normalize y to [-1, 1]
        y = (p['y'] / keyb_height) * 2 - 1
        # Keep timestamp in ms for now
        t = p['t']
        points.append([x, y, t])

    points = np.array(points, dtype=np.float32)

    # Convert timestamps to cumulative time from start, normalized to [0, 1]
    # This matches the prototype format (linspace 0 to 1) and allows the
    # generator to learn timing within its [-1, 1] output range
    start_time = points[0, 2]
    end_time = points[-1, 2]
    duration_ms = end_time - start_time
    if duration_ms > 0:
        points[:, 2] = (points[:, 2] - start_time) / duration_ms  # Normalize to [0, 1]
    else:
        points[:, 2] = np.linspace(0, 1, len(points))

    # Resample to seq_length points
    if len(points) == seq_length:
        return points

    # Interpolate/resample
    resampled = np.zeros((seq_length, 3), dtype=np.float32)

    # Calculate arc lengths for proper resampling
    diffs = np.diff(points[:, :2], axis=0)
    arc_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(arc_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-6:
        # Very short gesture - duplicate first point
        resampled[:, 0] = points[0, 0]
        resampled[:, 1] = points[0, 1]
        resampled[:, 2] = np.linspace(points[0, 2], points[-1, 2], seq_length)
        return resampled

    # Sample at uniform arc-length intervals
    target_lengths = np.linspace(0, total_length, seq_length)

    for i, target_len in enumerate(target_lengths):
        # Find segment containing this target length
        idx = np.searchsorted(cumulative_length, target_len, side='right') - 1
        idx = max(0, min(idx, len(points) - 2))

        # Interpolate within segment
        segment_start = cumulative_length[idx]
        segment_end = cumulative_length[idx + 1]
        segment_len = segment_end - segment_start

        if segment_len > 1e-6:
            t_interp = (target_len - segment_start) / segment_len
        else:
            t_interp = 0

        t_interp = max(0, min(1, t_interp))

        resampled[i] = points[idx] + t_interp * (points[idx + 1] - points[idx])

    return resampled


def load_dataset_from_zip(
    zip_path: str,
    keyboard: QWERTYKeyboard,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    max_files: Optional[int] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray]]:
    """
    Load gesture dataset from zip file.

    Args:
        zip_path: Path to swipelogs.zip
        keyboard: QWERTYKeyboard instance for generating prototypes
        model_config: Model configuration
        training_config: Training configuration
        max_files: Maximum number of log files to process (for debugging)

    Returns:
        Tuple of (gestures_by_word, prototypes_by_word)
    """
    gestures_by_word = defaultdict(list)
    processed_files = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        log_files = [f for f in zf.namelist() if f.endswith('.log')]

        if max_files:
            log_files = log_files[:max_files]

        for log_file in log_files:
            try:
                with zf.open(log_file) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    file_gestures = parse_log_file(content)

                    for word, gesture_list in file_gestures.items():
                        for gesture_points in gesture_list:
                            normalized = normalize_gesture(
                                gesture_points,
                                model_config.seq_length
                            )
                            gestures_by_word[word].append(normalized)

                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files...")

            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue

    print(f"Processed {processed_files} log files")
    print(f"Found {len(gestures_by_word)} unique words")

    # Cap samples per word to balance dataset
    max_samples = training_config.max_samples_per_word
    for word in gestures_by_word:
        if len(gestures_by_word[word]) > max_samples:
            gestures_by_word[word] = random.sample(
                gestures_by_word[word], max_samples
            )

    # Generate prototypes for each word
    prototypes_by_word = {}
    for word in gestures_by_word:
        prototypes_by_word[word] = keyboard.get_word_prototype(word, model_config.seq_length)

    return dict(gestures_by_word), prototypes_by_word


def create_train_test_split(
    gestures_by_word: Dict[str, List[np.ndarray]],
    prototypes_by_word: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[GestureDataset, GestureDataset]:
    """
    Split dataset into train and test sets by word.

    Words are split, not individual gestures, ensuring no word overlap.

    Args:
        gestures_by_word: Dictionary of word -> gesture list
        prototypes_by_word: Dictionary of word -> prototype
        train_ratio: Fraction of words for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    random.seed(seed)
    np.random.seed(seed)

    words = list(gestures_by_word.keys())
    random.shuffle(words)

    split_idx = int(len(words) * train_ratio)
    train_words = set(words[:split_idx])
    test_words = set(words[split_idx:])

    print(f"Training words: {len(train_words)}, Test words: {len(test_words)}")

    # Create train dataset
    train_gestures = []
    train_prototypes = []
    train_word_list = []

    for word in train_words:
        prototype = prototypes_by_word[word]
        for gesture in gestures_by_word[word]:
            train_gestures.append(gesture)
            train_prototypes.append(prototype)
            train_word_list.append(word)

    # Create test dataset
    test_gestures = []
    test_prototypes = []
    test_word_list = []

    for word in test_words:
        prototype = prototypes_by_word[word]
        for gesture in gestures_by_word[word]:
            test_gestures.append(gesture)
            test_prototypes.append(prototype)
            test_word_list.append(word)

    print(f"Training samples: {len(train_gestures)}, Test samples: {len(test_gestures)}")

    train_dataset = GestureDataset(train_gestures, train_prototypes, train_word_list)
    test_dataset = GestureDataset(test_gestures, test_prototypes, test_word_list)

    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: GestureDataset,
    test_dataset: GestureDataset,
    batch_size: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
