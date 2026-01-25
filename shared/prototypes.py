"""Word prototype construction."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .keyboard import KeyboardLayout


def build_word_prototype(word: str, n_points: int, layout: KeyboardLayout | None = None) -> np.ndarray:
    if layout is None:
        layout = KeyboardLayout()
    centers = layout.key_centers_normalized()
    keys = [ch for ch in word.lower() if ch in centers]
    if not keys:
        raise ValueError(f"Word '{word}' has no valid keys")

    points = [np.array(centers[ch], dtype=np.float32) for ch in keys]
    if len(points) == 1:
        xy = np.repeat(points[0][None, :], n_points, axis=0)
    else:
        total_extra = n_points - len(points)
        segments = len(points) - 1
        base = total_extra // segments
        remainder = total_extra % segments
        xy_list = [points[0]]
        for idx in range(segments):
            extra = base + (1 if idx < remainder else 0)
            start = points[idx]
            end = points[idx + 1]
            interp = np.linspace(start, end, num=extra + 2, dtype=np.float32)
            xy_list.extend(interp[1:])  # skip duplicate start
        xy = np.stack(xy_list, axis=0)

    dt = np.zeros((n_points, 1), dtype=np.float32)
    return np.concatenate([xy, dt], axis=1)


def build_batch_prototypes(words: Sequence[str], n_points: int, layout: KeyboardLayout | None = None) -> np.ndarray:
    if layout is None:
        layout = KeyboardLayout()
    return np.stack([build_word_prototype(word, n_points, layout) for word in words], axis=0)
