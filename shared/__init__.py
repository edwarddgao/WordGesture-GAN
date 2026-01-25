"""Shared utilities for gesture recognition."""

from .data import load_dataset, preprocess_dataset
from .keyboard import QWERTY_ROWS, KeyboardLayout, coalesce_repeats, word_key_centers
from .prototypes import build_batch_prototypes, build_word_prototype
from .utils import get_device

__all__ = [
    "QWERTY_ROWS",
    "KeyboardLayout",
    "coalesce_repeats",
    "word_key_centers",
    "load_dataset",
    "preprocess_dataset",
    "build_word_prototype",
    "build_batch_prototypes",
    "get_device",
]
