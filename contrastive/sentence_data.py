"""Build sentence-level dataset from raw JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from shared import KeyboardLayout, build_word_prototype
from shared.data import RawSample, filter_sample, to_fixed_length


@dataclass
class SentenceData:
    """A single sentence attempt with all word gestures."""

    sentence_id: str  # e.g., "what_about_jay"
    username: str
    words: List[str]  # ["what", "about", "jay"]
    gestures: List[np.ndarray]  # Corresponding (n_points, 3) arrays


@dataclass
class _WordAttempt:
    """Internal: tracks a word attempt within a sentence."""

    word: str
    gesture: np.ndarray
    position: int  # Position in sentence


def _parse_sentence_words(sentence_id: str) -> List[str]:
    """Parse words from sentence ID (underscore-separated)."""
    return sentence_id.split("_")


def _path_length(points: np.ndarray) -> float:
    """Calculate total Euclidean path length."""
    return float(np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2).sum())


def iter_raw_samples_with_user(raw_dir: Path) -> Iterator[Tuple[str, RawSample]]:
    """Iterate raw samples, yielding (username, sample) pairs."""
    for jsonl in sorted(raw_dir.rglob("*.jsonl")):
        if jsonl.name == "manifest.json":
            continue
        username = jsonl.stem
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield username, RawSample(**data)


def load_sentence_dataset(
    raw_dir: Path,
    n_points: int = 128,
    filter_errors: bool = True,
    min_complete_fraction: float = 0.5,
    layout: KeyboardLayout | None = None,
) -> List[SentenceData]:
    """Load sentence-level dataset from raw JSONL files.

    Args:
        raw_dir: Directory containing per-user JSONL files.
        n_points: Number of points to resample each gesture to.
        filter_errors: If True, only include successful gestures (is_err=0).
        min_complete_fraction: Minimum fraction of words that must be present
            for a sentence to be included.
        layout: If provided, filter out gestures where path length ratio
            (gesture/expected) is outside 0.5-2.0 range.

    Returns:
        List of SentenceData objects, each representing a sentence attempt
        with all (or most) word gestures present.
    """
    # Group samples by (username, sentence_id)
    # Key: (username, sentence_id) -> Dict[word -> List[_WordAttempt]]
    sentence_attempts: Dict[Tuple[str, str], Dict[str, List[_WordAttempt]]] = {}

    for username, sample in iter_raw_samples_with_user(raw_dir):
        # Apply filtering
        if filter_errors and sample.is_err:
            continue
        if not filter_sample(sample, min_points=5):
            continue

        sentence_id = sample.sentence
        word = sample.word.lower()

        # Get expected position of this word in the sentence
        expected_words = _parse_sentence_words(sentence_id)
        try:
            position = expected_words.index(word)
        except ValueError:
            # Word not found in sentence (shouldn't happen, but skip if so)
            continue

        # Convert to fixed-length gesture
        gesture = to_fixed_length(sample, n_points)

        # Filter by path length ratio
        if layout is not None:
            gesture_path = _path_length(gesture[:, :2])
            proto = build_word_prototype(word, n_points, layout)[:, :2]
            expected_path = _path_length(proto)
            if expected_path > 0:
                ratio = gesture_path / expected_path
                if not (0.5 <= ratio <= 2.0):
                    continue

        # Store attempt
        key = (username, sentence_id)
        if key not in sentence_attempts:
            sentence_attempts[key] = {}
        if word not in sentence_attempts[key]:
            sentence_attempts[key][word] = []
        sentence_attempts[key][word].append(
            _WordAttempt(word=word, gesture=gesture, position=position)
        )

    # Build SentenceData objects
    sentences: List[SentenceData] = []

    for (username, sentence_id), word_attempts in sentence_attempts.items():
        expected_words = _parse_sentence_words(sentence_id)
        n_expected = len(expected_words)

        # Check if we have enough words
        if len(word_attempts) < min_complete_fraction * n_expected:
            continue

        # Build ordered list of words and gestures
        words: List[str] = []
        gestures: List[np.ndarray] = []

        for expected_word in expected_words:
            if expected_word not in word_attempts:
                # Skip sentences with missing words
                break
            # Take the first successful attempt for this word
            attempt = word_attempts[expected_word][0]
            words.append(attempt.word)
            gestures.append(attempt.gesture)
        else:
            # All words present
            sentences.append(
                SentenceData(
                    sentence_id=sentence_id,
                    username=username,
                    words=words,
                    gestures=gestures,
                )
            )

    return sentences


def load_enron_sentence_ids(stats_path: Path) -> set:
    """Load sentence IDs from enron dataset (real sentences)."""
    enron_ids = set()
    with stats_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        sentence_idx = header.index("sentence")
        dataset_idx = header.index("dataset")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(sentence_idx, dataset_idx):
                if parts[dataset_idx] == "enron":
                    enron_ids.add(parts[sentence_idx])
    return enron_ids


def load_sentence_dataset_subset(
    raw_dir: Path,
    n_points: int = 128,
    max_sentences: int | None = None,
    seed: int = 42,
    natural_only: bool = False,
) -> List[SentenceData]:
    """Load a subset of sentence data for quick evaluation.

    Args:
        raw_dir: Directory containing per-user JSONL files.
        n_points: Number of points to resample each gesture to.
        max_sentences: Maximum number of sentences to return.
        seed: Random seed for reproducible sampling.
        natural_only: If True, only include sentences from enron dataset
            (real English sentences vs random word combinations).

    Returns:
        List of SentenceData objects.
    """
    sentences = load_sentence_dataset(raw_dir, n_points, layout=KeyboardLayout())

    # Filter to natural sentences if requested
    if natural_only:
        stats_path = raw_dir.parent.parent / "stats-sentences.tsv"
        if stats_path.exists():
            enron_ids = load_enron_sentence_ids(stats_path)
            sentences = [s for s in sentences if s.sentence_id in enron_ids]

    if max_sentences is not None and len(sentences) > max_sentences:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(sentences), size=max_sentences, replace=False)
        sentences = [sentences[i] for i in indices]

    return sentences


def get_sentence_stats(sentences: List[SentenceData]) -> Dict[str, float]:
    """Compute statistics about the sentence dataset."""
    n_sentences = len(sentences)
    n_words = sum(len(s.words) for s in sentences)
    avg_words = n_words / n_sentences if n_sentences > 0 else 0
    unique_sentences = len(set(s.sentence_id for s in sentences))
    unique_users = len(set(s.username for s in sentences))

    return {
        "n_sentences": n_sentences,
        "n_words": n_words,
        "avg_words_per_sentence": avg_words,
        "unique_sentences": unique_sentences,
        "unique_users": unique_users,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test sentence data loading.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/processed/raw"),
        help="Directory containing raw JSONL files.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=128,
        help="Number of points per gesture.",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Maximum sentences to load (for testing).",
    )
    args = parser.parse_args()

    print(f"Loading sentences from {args.raw_dir}...")
    sentences = load_sentence_dataset_subset(
        args.raw_dir, args.n_points, args.max_sentences
    )

    stats = get_sentence_stats(sentences)
    print(f"\nDataset statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    if sentences:
        print(f"\nExample sentence:")
        ex = sentences[0]
        print(f"  ID: {ex.sentence_id}")
        print(f"  User: {ex.username}")
        print(f"  Words: {ex.words}")
        print(f"  Gesture shapes: {[g.shape for g in ex.gestures]}")
