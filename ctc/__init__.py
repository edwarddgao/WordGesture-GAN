"""CTC-based gesture recognition module.

Includes feature extraction, model training, decoding, and evaluation.
"""

from shared.features import NumpyFeatureExtractor as GestureFeatureExtractor
from .models import BLSTMCTCModel
from .decode import CTCDecoder
from .trie import Trie
from .beam_search import ctc_beam_search_trie
from .reranker import OpenRouterReranker, NoopReranker, RerankerResult
from .sentence_data import (
    SentenceData,
    load_sentence_dataset,
    load_sentence_dataset_subset,
    get_sentence_stats,
)

__all__ = [
    "GestureFeatureExtractor",
    "BLSTMCTCModel",
    "CTCDecoder",
    "Trie",
    "ctc_beam_search_trie",
    "OpenRouterReranker",
    "NoopReranker",
    "RerankerResult",
    "SentenceData",
    "load_sentence_dataset",
    "load_sentence_dataset_subset",
    "get_sentence_stats",
]
