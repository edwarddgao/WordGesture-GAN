"""Preprocess raw JSONL samples into fixed-length tensors."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

WORD_RE = re.compile(r"^[a-z]+$")


@dataclass
class RawSample:
    sentence: str
    word: str
    is_err: int
    keyb_width: int
    keyb_height: int
    timestamps: List[int]
    xs: List[float]
    ys: List[float]


def iter_raw_samples(raw_dir: Path) -> Iterator[RawSample]:
    for jsonl in sorted(raw_dir.rglob("*.jsonl")):
        if jsonl.name == "manifest.json":
            continue
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield RawSample(**data)


def normalize_xy(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    x_norm = (xs / width) * 2.0 - 1.0
    y_norm = (ys / height) * 2.0 - 1.0
    return x_norm, y_norm


def resample_gesture(
    timestamps: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(xs)
    if count == n_points:
        return timestamps, xs, ys

    if count > n_points:
        indices = np.linspace(0, count - 1, num=n_points, dtype=int)
        return timestamps[indices], xs[indices], ys[indices]

    # Linear interpolation on index positions.
    new_idx = np.linspace(0, count - 1, num=n_points)
    timestamps_new = np.interp(new_idx, np.arange(count), timestamps)
    xs_new = np.interp(new_idx, np.arange(count), xs)
    ys_new = np.interp(new_idx, np.arange(count), ys)
    return timestamps_new, xs_new, ys_new


def to_fixed_length(sample: RawSample, n_points: int) -> np.ndarray:
    timestamps = np.asarray(sample.timestamps, dtype=np.float64)
    xs = np.asarray(sample.xs, dtype=np.float64)
    ys = np.asarray(sample.ys, dtype=np.float64)

    timestamps, xs, ys = resample_gesture(timestamps, xs, ys, n_points)
    xs, ys = normalize_xy(xs, ys, sample.keyb_width, sample.keyb_height)

    dt = np.diff(timestamps, prepend=timestamps[0]) / 1000.0
    gesture = np.stack([xs, ys, dt], axis=-1)
    return gesture.astype(np.float32)


def filter_sample(sample: RawSample, min_points: int = 5) -> bool:
    if sample.is_err:
        return False
    word = sample.word.lower()
    if len(word) < 2:
        return False
    if not WORD_RE.match(word):
        return False
    if len(sample.timestamps) < min_points:
        return False
    return True


def preprocess_dataset(
    raw_dir: Path,
    out_dir: Path,
    n_points: int,
    max_per_word: int,
    train_split: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    word_to_samples: Dict[str, List[np.ndarray]] = {}

    for sample in iter_raw_samples(raw_dir):
        if not filter_sample(sample):
            continue
        gesture = to_fixed_length(sample, n_points)
        word_to_samples.setdefault(sample.word.lower(), []).append(gesture)

    for word, samples in list(word_to_samples.items()):
        rng.shuffle(samples)
        word_to_samples[word] = samples[:max_per_word]
        if not word_to_samples[word]:
            word_to_samples.pop(word, None)

    words = list(word_to_samples.keys())
    rng.shuffle(words)
    split_idx = int(len(words) * train_split)
    train_words = set(words[:split_idx])
    test_words = set(words[split_idx:])

    train_gestures, train_labels = [], []
    test_gestures, test_labels = [], []
    for word, samples in word_to_samples.items():
        target_gestures = train_gestures if word in train_words else test_gestures
        target_labels = train_labels if word in train_words else test_labels
        for sample in samples:
            target_gestures.append(sample)
            target_labels.append(word)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "train.npz",
        gestures=np.asarray(train_gestures, dtype=np.float32),
        words=np.asarray(train_labels, dtype=object),
    )
    np.savez_compressed(
        out_dir / "test.npz",
        gestures=np.asarray(test_gestures, dtype=np.float32),
        words=np.asarray(test_labels, dtype=object),
    )

    meta = {
        "n_points": n_points,
        "max_per_word": max_per_word,
        "train_split": train_split,
        "seed": seed,
        "train_words": sorted(train_words),
        "test_words": sorted(test_words),
        "train_count": len(train_gestures),
        "test_count": len(test_gestures),
    }
    with (out_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_dataset(path: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    gestures = data["gestures"].astype(np.float32)
    words = data["words"].tolist()
    return gestures, words


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess swipe logs into fixed-length tensors.")
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--max_per_word", type=int, default=5)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        n_points=args.n_points,
        max_per_word=args.max_per_word,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
