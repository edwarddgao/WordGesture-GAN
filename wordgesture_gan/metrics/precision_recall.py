"""Precision/recall for generated gesture manifolds."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .distance import l2_distance


def _kth_neighbor_distances(gestures: np.ndarray, k: int) -> np.ndarray:
    n = len(gestures)
    distances = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append(l2_distance(gestures[i], gestures[j]))
        if not dists:
            distances[i] = 0.0
        else:
            distances[i] = float(np.partition(dists, min(k - 1, len(dists) - 1))[min(k - 1, len(dists) - 1)])
    return distances


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> Tuple[float, float]:
    if len(real) == 0 or len(fake) == 0:
        return float("nan"), float("nan")

    real_r = _kth_neighbor_distances(real, k)
    fake_r = _kth_neighbor_distances(fake, k)

    precision_hits = 0
    for f in fake:
        hit = False
        for r, radius in zip(real, real_r):
            if l2_distance(f, r) <= radius:
                hit = True
                break
        precision_hits += int(hit)

    recall_hits = 0
    for r in real:
        hit = False
        for f, radius in zip(fake, fake_r):
            if l2_distance(r, f) <= radius:
                hit = True
                break
        recall_hits += int(hit)

    precision = precision_hits / len(fake)
    recall = recall_hits / len(real)
    return float(precision), float(recall)
