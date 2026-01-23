"""Duration metrics and CLC baseline."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from ..keyboard.qwerty import KeyboardLayout
from ..prototypes import build_word_prototype


def gesture_duration(gesture: np.ndarray) -> float:
    return float(np.sum(gesture[:, 2]))


def word_durations(gestures: np.ndarray, words: List[str]) -> Dict[str, List[float]]:
    durations: Dict[str, List[float]] = {}
    for gesture, word in zip(gestures, words):
        durations.setdefault(word, []).append(gesture_duration(gesture))
    return durations


def _prototype_length(word: str, layout: KeyboardLayout, n_points: int = 128) -> float:
    proto = build_word_prototype(word, n_points, layout)
    xy = proto[:, :2]
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def fit_clc(
    gestures: np.ndarray,
    words: List[str],
    layout: KeyboardLayout | None = None,
) -> Tuple[float, float]:
    if layout is None:
        layout = KeyboardLayout()
    durations = word_durations(gestures, words)
    word_lengths = {word: _prototype_length(word, layout) for word in durations.keys()}
    y = np.array([np.mean(vals) for vals in durations.values()], dtype=np.float64)
    x = np.array([word_lengths[word] for word in durations.keys()], dtype=np.float64)

    def rmse(params: np.ndarray) -> float:
        log_m, log_n = params
        m = np.exp(log_m)
        n = np.exp(log_n)
        pred = m * (x ** n)
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    result = minimize(rmse, x0=np.log([300.0, 0.1]), method="Nelder-Mead")
    log_m, log_n = result.x
    return float(np.exp(log_m)), float(np.exp(log_n))


def clc_predict(word: str, m: float, n: float, layout: KeyboardLayout | None = None) -> float:
    if layout is None:
        layout = KeyboardLayout()
    length = _prototype_length(word, layout)
    return float(m * (length ** n))
