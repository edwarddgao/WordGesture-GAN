"""QWERTY key-centers and adjacency helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
ROW_OFFSETS = [0.0, 0.5, 1.5]  # in key widths
ROW_SPACING = 1.0
KEY_WIDTH = 1.0
KEY_HEIGHT = 1.0


def coalesce_repeats(word: str) -> str:
    if not word:
        return word
    out = [word[0]]
    for ch in word[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


@dataclass
class KeyboardLayout:
    key_width: float = KEY_WIDTH
    key_height: float = KEY_HEIGHT
    row_spacing: float = ROW_SPACING
    row_offsets: Tuple[float, float, float] = tuple(ROW_OFFSETS)

    def _key_positions_unit(self) -> Dict[str, Tuple[float, float]]:
        positions: Dict[str, Tuple[float, float]] = {}
        for row_idx, row in enumerate(QWERTY_ROWS):
            offset = self.row_offsets[row_idx] * self.key_width
            y = (row_idx + 0.5) * self.row_spacing
            for col_idx, ch in enumerate(row):
                x = offset + (col_idx + 0.5) * self.key_width
                positions[ch] = (x, y)
        return positions

    def key_centers_normalized(self) -> Dict[str, Tuple[float, float]]:
        positions = self._key_positions_unit()
        width = 10 * self.key_width
        height = 3 * self.row_spacing
        normalized = {}
        for key, (x, y) in positions.items():
            x_norm = (x / width) * 2.0 - 1.0
            y_norm = (y / height) * 2.0 - 1.0
            normalized[key] = (x_norm, y_norm)
        return normalized

    def key_centers_unit(self) -> Dict[str, Tuple[float, float]]:
        return self._key_positions_unit()

    def adjacency(self, threshold: float = 1.45) -> Set[Tuple[str, str]]:
        positions = self._key_positions_unit()
        keys = list(positions.keys())
        adj: Set[Tuple[str, str]] = set()
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                p1 = np.array(positions[k1])
                p2 = np.array(positions[k2])
                if np.linalg.norm(p1 - p2) <= threshold:
                    adj.add((k1, k2))
                    adj.add((k2, k1))
        return adj

    def are_adjacent(self, key_a: str, key_b: str, threshold: float = 1.45) -> bool:
        if key_a == key_b:
            return True
        positions = self._key_positions_unit()
        p1 = np.array(positions[key_a])
        p2 = np.array(positions[key_b])
        return np.linalg.norm(p1 - p2) <= threshold


def word_key_centers(word: str, layout: KeyboardLayout) -> List[Tuple[float, float]]:
    centers = layout.key_centers_normalized()
    return [centers[ch] for ch in word]
