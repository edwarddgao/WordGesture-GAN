"""Duration metrics and CLC baseline."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from ..keyboard.qwerty import KeyboardLayout


def gesture_duration(gesture: np.ndarray) -> float:
    return float(np.sum(gesture[:, 2]))


def word_durations(gestures: np.ndarray, words: List[str]) -> Dict[str, List[float]]:
    durations: Dict[str, List[float]] = {}
    for gesture, word in zip(gestures, words):
        durations.setdefault(word, []).append(gesture_duration(gesture))
    return durations


def _prototype_segment_lengths(word: str, layout: KeyboardLayout) -> np.ndarray:
    centers = layout.key_centers_normalized()
    keys = [ch for ch in word.lower() if ch in centers]
    if len(keys) < 2:
        return np.zeros(0, dtype=np.float64)
    points = np.array([centers[ch] for ch in keys], dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def fit_clc(
    gestures: np.ndarray,
    words: List[str],
    layout: KeyboardLayout | None = None,
) -> Tuple[float, float]:
    if layout is None:
        layout = KeyboardLayout()
    durations = word_durations(gestures, words)
    word_list = list(durations.keys())
    segment_lengths = [_prototype_segment_lengths(word, layout) for word in word_list]
    y = np.array([np.mean(durations[word]) for word in word_list], dtype=np.float64)

    def rmse(params: np.ndarray) -> float:
        log_m, log_n = params
        m = np.exp(log_m)
        n = np.exp(log_n)
        pred = np.array([m * np.sum(lengths ** n) for lengths in segment_lengths], dtype=np.float64)
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    result = minimize(rmse, x0=np.log([300.0, 0.1]), method="Nelder-Mead")
    log_m, log_n = result.x
    return float(np.exp(log_m)), float(np.exp(log_n))


def clc_predict(word: str, m: float, n: float, layout: KeyboardLayout | None = None) -> float:
    if layout is None:
        layout = KeyboardLayout()
    lengths = _prototype_segment_lengths(word, layout)
    return float(m * np.sum(lengths ** n))
