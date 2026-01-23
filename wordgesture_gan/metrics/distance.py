"""Distance metrics and Wasserstein matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(a - b, axis=-1)))


@njit(cache=True)
def _dtw_core(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    m = b.shape[0]
    cost = np.full((n + 1, m + 1), 1e20, dtype=np.float64)
    cost[0, 0] = 0.0
    dims = a.shape[1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = 0.0
            for k in range(dims):
                diff = a[i - 1, k] - b[j - 1, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            best = cost[i - 1, j]
            if cost[i, j - 1] < best:
                best = cost[i, j - 1]
            if cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
            cost[i, j] = dist + best
    return cost[n, m]


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(_dtw_core(a, b))


def _pairwise_cost(
    real: List[np.ndarray],
    fake: List[np.ndarray],
    metric: str,
) -> np.ndarray:
    metric_fn = l2_distance if metric == "l2" else dtw_distance
    cost = np.zeros((len(real), len(fake)), dtype=np.float32)
    for i, r in enumerate(real):
        for j, f in enumerate(fake):
            cost[i, j] = metric_fn(r, f)
    return cost


def wasserstein_matching(
    real: List[np.ndarray],
    fake: List[np.ndarray],
    metric: str = "l2",
) -> float:
    if not real or not fake:
        return float("nan")
    cost = _pairwise_cost(real, fake, metric)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def per_word_wasserstein(
    real: np.ndarray,
    real_words: List[str],
    fake: np.ndarray,
    fake_words: List[str],
    metric: str = "l2",
    show_progress: bool = True,
) -> Tuple[float, float]:
    words = sorted(set(real_words))
    word_to_real: Dict[str, List[np.ndarray]] = {w: [] for w in words}
    word_to_fake: Dict[str, List[np.ndarray]] = {w: [] for w in words}
    for gesture, word in zip(real, real_words):
        word_to_real[word].append(gesture)
    for gesture, word in zip(fake, fake_words):
        if word in word_to_fake:
            word_to_fake[word].append(gesture)
    scores = []
    iterator = tqdm(words, desc=f"         {metric.upper()} matching", leave=False) if show_progress else words
    for word in iterator:
        score = wasserstein_matching(word_to_real[word], word_to_fake[word], metric)
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))
