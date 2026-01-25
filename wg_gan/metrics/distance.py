"""Distance metrics and Wasserstein matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numba import njit, prange
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


@njit(cache=True)
def _dtw_core(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    m = b.shape[0]
    cost = np.full((n + 1, m + 1), 1e20, dtype=np.float64)
    cost[0, 0] = 0.0
    dims = a.shape[1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist_sq = 0.0
            for k in range(dims):
                diff = a[i - 1, k] - b[j - 1, k]
                dist_sq += diff * diff
            best = cost[i - 1, j]
            if cost[i, j - 1] < best:
                best = cost[i, j - 1]
            if cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
            cost[i, j] = dist_sq + best
    return np.sqrt(cost[n, m])


@njit(cache=True)
def _dtw_core_banded(a: np.ndarray, b: np.ndarray, band_width: int) -> float:
    """DTW with Sakoe-Chiba band constraint."""
    n, m = a.shape[0], b.shape[0]
    dims = a.shape[1]
    cost = np.full((n + 1, m + 1), 1e20, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band_width)
        j_end = min(m + 1, i + band_width + 1)
        for j in range(j_start, j_end):
            dist_sq = 0.0
            for k in range(dims):
                diff = a[i - 1, k] - b[j - 1, k]
                dist_sq += diff * diff
            best = cost[i - 1, j]
            if cost[i, j - 1] < best:
                best = cost[i, j - 1]
            if cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
            cost[i, j] = dist_sq + best
    return np.sqrt(cost[n, m])


@njit(parallel=True, cache=True)
def _pairwise_dtw_parallel(real: np.ndarray, fake: np.ndarray, band_width: int) -> np.ndarray:
    """Compute pairwise DTW distances in parallel."""
    n_real, n_fake = real.shape[0], fake.shape[0]
    cost = np.zeros((n_real, n_fake), dtype=np.float64)
    total = n_real * n_fake

    for idx in prange(total):
        i, j = idx // n_fake, idx % n_fake
        if band_width > 0:
            cost[i, j] = _dtw_core_banded(real[i], fake[j], band_width)
        else:
            cost[i, j] = _dtw_core(real[i], fake[j])
    return cost


def _pairwise_cost(
    real: List[np.ndarray],
    fake: List[np.ndarray],
    metric: str,
    band_width: int = 0,
) -> np.ndarray:
    if metric == "l2":
        cost = np.zeros((len(real), len(fake)), dtype=np.float32)
        for i, r in enumerate(real):
            for j, f in enumerate(fake):
                cost[i, j] = l2_distance(r, f)
        return cost
    # DTW: use parallel implementation
    real_arr = np.stack(real, axis=0)
    fake_arr = np.stack(fake, axis=0)
    return _pairwise_dtw_parallel(real_arr, fake_arr, band_width).astype(np.float32)


def wasserstein_matching(
    real: List[np.ndarray],
    fake: List[np.ndarray],
    metric: str = "l2",
    band_width: int = 0,
) -> float:
    if not real or not fake:
        return float("nan")
    cost = _pairwise_cost(real, fake, metric, band_width)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def per_word_wasserstein(
    real: np.ndarray,
    real_words: List[str],
    fake: np.ndarray,
    fake_words: List[str],
    metric: str = "l2",
    show_progress: bool = True,
    band_width: int = 0,
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
        score = wasserstein_matching(word_to_real[word], word_to_fake[word], metric, band_width)
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))
