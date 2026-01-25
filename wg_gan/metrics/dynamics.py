"""Velocity, acceleration, and jerk metrics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(pearsonr(a, b)[0])


def derivatives(gesture: np.ndarray, window: int = 5, poly: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = gesture[:, :2]
    dt = gesture[:, 2]
    delta = float(np.mean(dt[1:])) if len(dt) > 1 else 1.0
    velocity = np.zeros_like(positions)
    accel = np.zeros_like(positions)
    jerk = np.zeros_like(positions)
    for axis in range(2):
        velocity[:, axis] = savgol_filter(positions[:, axis], window, poly, deriv=1, delta=delta)
        accel[:, axis] = savgol_filter(positions[:, axis], window, poly, deriv=2, delta=delta)
        jerk[:, axis] = savgol_filter(positions[:, axis], window, poly, deriv=3, delta=delta)
    return velocity, accel, jerk


def dynamics_correlation(real: np.ndarray, fake: np.ndarray) -> Tuple[float, float]:
    real_v, real_a, _ = derivatives(real)
    fake_v, fake_a, _ = derivatives(fake)
    v_corr = (_safe_corr(real_v[:, 0], fake_v[:, 0]) + _safe_corr(real_v[:, 1], fake_v[:, 1])) / 2.0
    a_corr = (_safe_corr(real_a[:, 0], fake_a[:, 0]) + _safe_corr(real_a[:, 1], fake_a[:, 1])) / 2.0
    return float(v_corr), float(a_corr)


def jerk_stat(gesture: np.ndarray) -> float:
    _, _, jerk = derivatives(gesture)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))
