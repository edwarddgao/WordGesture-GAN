"""Sample minimum jerk trajectories from fitted distributions."""

from __future__ import annotations

from typing import List

import numpy as np

from shared import KeyboardLayout, coalesce_repeats
from .fit import MinJerkFit


def _rotate(vec: np.ndarray, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    return rot @ vec


def _poly_terms(t: float) -> np.ndarray:
    return np.array([1.0, t, t**2, t**3, t**4, t**5], dtype=np.float64)


def _poly_d1(t: float) -> np.ndarray:
    return np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4], dtype=np.float64)


def _poly_d2(t: float) -> np.ndarray:
    return np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3], dtype=np.float64)


def _segment_jerk_cost(t: float) -> np.ndarray:
    q = np.zeros((6, 6), dtype=np.float64)
    for i in range(3, 6):
        for j in range(3, 6):
            coef = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2)
            q[i, j] = coef * (t ** (i + j - 5)) / (i + j - 5)
    return q


def _segment_row(seg_idx: int, terms: np.ndarray, n_segments: int) -> np.ndarray:
    row = np.zeros(6 * n_segments, dtype=np.float64)
    start = seg_idx * 6
    row[start : start + 6] = terms
    return row


def _build_constraints(positions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_segments = len(times)
    rows: List[np.ndarray] = []
    b: List[float] = []

    # Start position
    rows.append(_segment_row(0, _poly_terms(0.0), n_segments))
    b.append(float(positions[0]))

    # End positions
    for idx, t in enumerate(times):
        rows.append(_segment_row(idx, _poly_terms(float(t)), n_segments))
        b.append(float(positions[idx + 1]))

    # Position continuity
    for idx in range(1, n_segments):
        row = _segment_row(idx - 1, _poly_terms(float(times[idx - 1])), n_segments)
        row -= _segment_row(idx, _poly_terms(0.0), n_segments)
        rows.append(row)
        b.append(0.0)

    # Velocity continuity
    for idx in range(1, n_segments):
        row = _segment_row(idx - 1, _poly_d1(float(times[idx - 1])), n_segments)
        row -= _segment_row(idx, _poly_d1(0.0), n_segments)
        rows.append(row)
        b.append(0.0)

    # Acceleration continuity
    for idx in range(1, n_segments):
        row = _segment_row(idx - 1, _poly_d2(float(times[idx - 1])), n_segments)
        row -= _segment_row(idx, _poly_d2(0.0), n_segments)
        rows.append(row)
        b.append(0.0)

    # Boundary velocity and acceleration constraints
    rows.append(_segment_row(0, _poly_d1(0.0), n_segments))
    b.append(0.0)
    rows.append(_segment_row(n_segments - 1, _poly_d1(float(times[-1])), n_segments))
    b.append(0.0)

    rows.append(_segment_row(0, _poly_d2(0.0), n_segments))
    b.append(0.0)
    rows.append(_segment_row(n_segments - 1, _poly_d2(float(times[-1])), n_segments))
    b.append(0.0)

    return np.vstack(rows), np.asarray(b, dtype=np.float64)


def _solve_minimum_jerk(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    n_segments = len(times)
    q = np.zeros((6 * n_segments, 6 * n_segments), dtype=np.float64)
    for idx, t in enumerate(times):
        block = _segment_jerk_cost(float(t))
        start = idx * 6
        q[start : start + 6, start : start + 6] = block

    a, b = _build_constraints(positions, times)
    zeros = np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
    kkt = np.block([[q, a.T], [a, zeros]])
    rhs = np.concatenate([np.zeros(q.shape[0], dtype=np.float64), b], axis=0)

    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[: q.shape[0]]


def _segment_durations(via_points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(via_points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total < 1e-6:
        return np.full(len(lengths), 1.0 / max(len(lengths), 1), dtype=np.float64)
    return (lengths / total).astype(np.float64)


def _sample_trajectory(
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    times: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_segments = len(times)
    coeffs_x = coeffs_x.reshape(n_segments, 6)
    coeffs_y = coeffs_y.reshape(n_segments, 6)
    boundaries = np.concatenate([[0.0], np.cumsum(times)])
    t_global = np.linspace(0.0, float(boundaries[-1]), num=n_points, dtype=np.float64)

    xy = np.zeros((n_points, 2), dtype=np.float64)
    for idx, t in enumerate(t_global):
        seg = int(np.searchsorted(boundaries, t, side="right") - 1)
        seg = min(max(seg, 0), n_segments - 1)
        local_t = float(t - boundaries[seg])
        terms = _poly_terms(local_t)
        xy[idx, 0] = float(np.dot(coeffs_x[seg], terms))
        xy[idx, 1] = float(np.dot(coeffs_y[seg], terms))

    dt = np.diff(t_global, prepend=t_global[0])
    return xy.astype(np.float32), dt.astype(np.float32)


def sample_via_points(word: str, fit: MinJerkFit, layout: KeyboardLayout) -> List[np.ndarray]:
    centers = layout.key_centers_normalized()
    keys = coalesce_repeats(word.lower())
    if not keys:
        raise ValueError(f"Invalid word '{word}' for minimum jerk sampling")

    via_points: List[np.ndarray] = []
    for key in keys:
        offset = np.random.multivariate_normal(fit.offset_mean, fit.offset_cov)
        via_points.append(np.array(centers[key]) + offset)

    enriched: List[np.ndarray] = []
    for idx in range(len(via_points) - 1):
        current = via_points[idx]
        nxt = via_points[idx + 1]
        enriched.append(current)
        if not layout.are_adjacent(keys[idx], keys[idx + 1]):
            v1 = np.array(centers[keys[idx + 1]]) - current
            if np.linalg.norm(v1) > 1e-6:
                v1_unit = v1 / np.linalg.norm(v1)
                theta = np.random.normal(fit.angle_mean, fit.angle_std)
                span = float(np.linalg.norm(nxt - current))
                mid = current + _rotate(v1_unit, theta) * (span * 0.5)
                enriched.append(mid)
    enriched.append(via_points[-1])
    return enriched


def sample_minimum_jerk(
    word: str,
    fit: MinJerkFit,
    n_points: int,
    layout: KeyboardLayout | None = None,
) -> np.ndarray:
    if layout is None:
        layout = KeyboardLayout()
    via_points = sample_via_points(word, fit, layout)
    via_points = np.asarray(via_points, dtype=np.float32)

    if len(via_points) < 2:
        xy = np.repeat(via_points[:1], n_points, axis=0)
        dt = np.zeros((n_points, 1), dtype=np.float32)
    else:
        times = _segment_durations(via_points)
        coeffs_x = _solve_minimum_jerk(via_points[:, 0].astype(np.float64), times)
        coeffs_y = _solve_minimum_jerk(via_points[:, 1].astype(np.float64), times)
        xy, dt = _sample_trajectory(coeffs_x, coeffs_y, times, n_points)

    dt = dt.reshape(-1, 1)
    return np.concatenate([xy, dt], axis=1)
