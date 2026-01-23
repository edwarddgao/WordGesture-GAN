"""Sample minimum jerk trajectories from fitted distributions."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.interpolate import make_interp_spline

from ..keyboard.qwerty import KeyboardLayout, coalesce_repeats
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
                mid = current + _rotate(v1_unit, theta) * (np.linalg.norm(v1) * 0.5)
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
    else:
        t = np.linspace(0.0, 1.0, num=len(via_points))
        k = min(5, len(via_points) - 1)
        if k < 2:
            xy = np.linspace(via_points[0], via_points[-1], num=n_points, dtype=np.float32)
        else:
            t_new = np.linspace(0.0, 1.0, num=n_points)
            if k == 5:
                bc = ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)])
                spline_x = make_interp_spline(t, via_points[:, 0], k=k, bc_type=bc)
                spline_y = make_interp_spline(t, via_points[:, 1], k=k, bc_type=bc)
            else:
                spline_x = make_interp_spline(t, via_points[:, 0], k=k)
                spline_y = make_interp_spline(t, via_points[:, 1], k=k)
            xy = np.stack([spline_x(t_new), spline_y(t_new)], axis=-1).astype(np.float32)

    dt = np.full((n_points, 1), 1.0 / max(n_points - 1, 1), dtype=np.float32)
    return np.concatenate([xy, dt], axis=1)
