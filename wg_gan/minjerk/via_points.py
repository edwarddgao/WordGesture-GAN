"""Via-point extraction for minimum jerk model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from shared import KeyboardLayout, coalesce_repeats


@dataclass
class ViaPoint:
    idx: int
    xy: np.ndarray
    key: str | None
    kind: str  # "char" or "midpoint"


def _curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.where(denom < 1e-6, 1e-6, denom)
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return curvature


def segments(points: np.ndarray, word: str, layout: KeyboardLayout) -> Tuple[List[List[int]], str]:
    keys = coalesce_repeats(word.lower())
    centers = layout.key_centers_normalized()

    if len(points) < 4 or len(keys) == 0:
        return [], keys

    point_indices = list(range(len(points)))
    working = point_indices[2:-2]
    segments_out: List[List[int]] = []
    cursor = 0

    for idx in range(len(keys) - 1):
        key = keys[idx]
        next_key = keys[idx + 1]
        if key == next_key:
            continue
        segment: List[int] = []
        d = float("inf")
        while cursor < len(working):
            p_idx = working[cursor]
            p = points[p_idx]
            d_prime = np.linalg.norm(p - centers[key])
            if d_prime < d or d_prime < np.linalg.norm(p - centers[next_key]):
                segment.append(p_idx)
                d = d_prime
                cursor += 1
            else:
                break
        segments_out.append(segment)

    if not segments_out:
        segments_out = [working]

    segments_out[0] = [0, 1] + segments_out[0]
    remaining = working[cursor:]
    segments_out[-1] = segments_out[-1] + remaining + [len(points) - 2, len(points) - 1]
    return segments_out, keys


def _midpoint_on_path(points: np.ndarray, idx_a: int, idx_b: int) -> int:
    if idx_a == idx_b:
        return idx_a
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    sub = points[idx_a : idx_b + 1]
    if len(sub) < 2:
        return idx_a
    dists = np.linalg.norm(np.diff(sub, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(dists)])
    target = cumulative[-1] / 2.0
    mid_local = int(np.argmin(np.abs(cumulative - target)))
    return idx_a + mid_local


def extract_via_points(points: np.ndarray, word: str, layout: KeyboardLayout) -> List[ViaPoint]:
    segments_out, keys = segments(points, word, layout)
    if not segments_out:
        return []
    centers = layout.key_centers_normalized()
    via_points: List[ViaPoint] = []

    first_idx = segments_out[0][0]
    via_points.append(ViaPoint(first_idx, points[first_idx], keys[0], "char"))

    for idx in range(1, len(keys) - 1):
        segment = segments_out[idx]
        if not segment:
            continue
        segment_points = points[segment]
        curvature = _curvature(segment_points)
        max_curv = float(np.max(curvature))
        if max_curv > 0.01:
            sel = int(np.argmax(curvature))
            seg_idx = segment[sel]
        else:
            key_center = np.array(centers[keys[idx]])
            dists = np.linalg.norm(segment_points - key_center, axis=1)
            seg_idx = segment[int(np.argmin(dists))]
        via_points.append(ViaPoint(seg_idx, points[seg_idx], keys[idx], "char"))

    last_idx = segments_out[-1][-1]
    via_points.append(ViaPoint(last_idx, points[last_idx], keys[-1], "char"))

    # Midpoints between non-adjacent keys.
    enriched: List[ViaPoint] = []
    for i in range(len(via_points) - 1):
        current = via_points[i]
        nxt = via_points[i + 1]
        enriched.append(current)
        if current.key and nxt.key and not layout.are_adjacent(current.key, nxt.key):
            mid_idx = _midpoint_on_path(points, current.idx, nxt.idx)
            enriched.append(ViaPoint(mid_idx, points[mid_idx], None, "midpoint"))
    enriched.append(via_points[-1])
    return enriched


def validate_gesture(points: np.ndarray, word: str, layout: KeyboardLayout) -> bool:
    segments_out, keys = segments(points, word, layout)
    if not segments_out:
        return False
    if any(len(segment) < 2 for segment in segments_out):
        return False
    centers = layout.key_centers_normalized()
    aw_dist = np.linalg.norm(np.array(centers["a"]) - np.array(centers["w"]))
    via_points = extract_via_points(points, word, layout)
    for vp in via_points:
        if vp.kind != "char" or vp.key is None:
            continue
        if np.linalg.norm(vp.xy - np.array(centers[vp.key])) > aw_dist:
            return False
    return True
