"""Precision/recall for generated gesture manifolds."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise L2 distances between two sets of gestures.

    Args:
        a: shape (n, seq_len, dims)
        b: shape (m, seq_len, dims)

    Returns:
        Distance matrix of shape (n, m)
    """
    n, seq_len, dims = a.shape
    m = b.shape[0]
    result = np.zeros((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            total = 0.0
            for t in range(seq_len):
                dist_sq = 0.0
                for d in range(dims):
                    diff = a[i, t, d] - b[j, t, d]
                    dist_sq += diff * diff
                total += np.sqrt(dist_sq)
            result[i, j] = total

    return result


def _kth_neighbor_distances(gestures: np.ndarray, k: int) -> np.ndarray:
    """Compute k-th nearest neighbor distance for each gesture."""
    n = len(gestures)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    # Compute all pairwise distances: (n, n)
    dists = _pairwise_l2(gestures, gestures)

    # Set diagonal to inf so we don't count self-distance
    np.fill_diagonal(dists, np.inf)

    # Get k-th smallest distance for each row
    k_idx = min(k - 1, n - 2)
    kth_dists = np.partition(dists, k_idx, axis=1)[:, k_idx]

    return kth_dists.astype(np.float32)


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> Tuple[float, float]:
    if len(real) == 0 or len(fake) == 0:
        return float("nan"), float("nan")

    real_r = _kth_neighbor_distances(real, k)
    fake_r = _kth_neighbor_distances(fake, k)

    # Compute cross-distances: (n_fake, n_real)
    fake_to_real = _pairwise_l2(fake, real)
    # Compute cross-distances: (n_real, n_fake)
    real_to_fake = _pairwise_l2(real, fake)

    # Precision: for each fake, check if any real is within its radius
    # fake_to_real[i, j] <= real_r[j] means fake[i] is within real[j]'s neighborhood
    precision_hits = np.any(fake_to_real <= real_r[None, :], axis=1).sum()

    # Recall: for each real, check if any fake is within its radius
    recall_hits = np.any(real_to_fake <= fake_r[None, :], axis=1).sum()

    precision = precision_hits / len(fake)
    recall = recall_hits / len(real)
    return float(precision), float(recall)
