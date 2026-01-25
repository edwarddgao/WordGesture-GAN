"""Fit aggregate distributions for minimum jerk sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from shared import KeyboardLayout
from .via_points import extract_via_points, validate_gesture


@dataclass
class MinJerkFit:
    offset_mean: np.ndarray
    offset_cov: np.ndarray
    angle_mean: float
    angle_std: float

    def to_dict(self) -> Dict:
        return {
            "offset_mean": self.offset_mean.tolist(),
            "offset_cov": self.offset_cov.tolist(),
            "angle_mean": float(self.angle_mean),
            "angle_std": float(self.angle_std),
        }

    @staticmethod
    def from_dict(data: Dict) -> "MinJerkFit":
        return MinJerkFit(
            offset_mean=np.asarray(data["offset_mean"], dtype=np.float32),
            offset_cov=np.asarray(data["offset_cov"], dtype=np.float32),
            angle_mean=float(data["angle_mean"]),
            angle_std=float(data["angle_std"]),
        )


def _signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return float(np.sign(cross) * angle)


def fit_distributions(
    gestures: np.ndarray,
    words: List[str],
    layout: KeyboardLayout | None = None,
) -> MinJerkFit:
    if layout is None:
        layout = KeyboardLayout()
    centers = layout.key_centers_normalized()

    offsets: List[np.ndarray] = []
    angles: List[float] = []

    for gesture, word in zip(gestures, words):
        points = gesture[:, :2]
        if not validate_gesture(points, word, layout):
            continue
        via_points = extract_via_points(points, word, layout)

        for vp in via_points:
            if vp.kind != "char" or vp.key is None:
                continue
            offsets.append(vp.xy - np.array(centers[vp.key]))

        for idx, vp in enumerate(via_points):
            if vp.kind != "midpoint":
                continue
            prev_char = next((p for p in reversed(via_points[:idx]) if p.kind == "char"), None)
            next_char = next((p for p in via_points[idx + 1 :] if p.kind == "char"), None)
            if prev_char is None or next_char is None:
                continue
            v1 = np.array(centers[next_char.key]) - prev_char.xy
            v2 = vp.xy - prev_char.xy
            angles.append(_signed_angle(v1, v2))

    if not offsets:
        raise ValueError("No valid offsets found for minimum jerk fitting.")
    offsets_arr = np.stack(offsets, axis=0)
    offset_mean = offsets_arr.mean(axis=0)
    offset_cov = np.cov(offsets_arr.T)

    if angles:
        angles_arr = np.asarray(angles, dtype=np.float32)
        lower, upper = np.percentile(angles_arr, [1, 99])
        clipped = angles_arr[(angles_arr >= lower) & (angles_arr <= upper)]
        angle_mean = float(np.mean(clipped))
        angle_std = float(np.std(clipped) + 1e-6)
    else:
        angle_mean = 0.0
        angle_std = 1.0

    return MinJerkFit(offset_mean=offset_mean, offset_cov=offset_cov, angle_mean=angle_mean, angle_std=angle_std)
