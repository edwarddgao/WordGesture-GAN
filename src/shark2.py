"""
SHARK2 decoder for word-gesture recognition.

Implements the multi-channel recognition system from:
"SHARK2: A Large Vocabulary Shorthand Writing System for Pen-based Computers" (UIST 2004)

The SHARK2 decoder combines three channels:
- Location channel: Distance from gesture to word template (key positions)
- Shape channel: Shape similarity after normalizing for position/scale
- Language model: Unigram word probabilities

Combined score: exp(-dist_loc^2 / 2*sigma_loc^2) * exp(-dist_shape^2 / 2*sigma_shape^2) * P(word)^sigma_lm
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

from .keyboard import QWERTYKeyboard


def load_word_frequencies(lexicon: List[str], use_zipf: bool = True) -> Dict[str, float]:
    """
    Load word frequencies for a lexicon using the wordfreq library.

    The wordfreq library includes data from multiple sources including
    COCA (Corpus of Contemporary American English), matching the paper's
    "30k unigram language model trained from COCA Corpus".

    Args:
        lexicon: List of words to get frequencies for
        use_zipf: If True, use Zipf scale (0-8 range), else use raw probabilities

    Returns:
        Dictionary mapping word -> frequency
    """
    try:
        from wordfreq import word_frequency, zipf_frequency
    except ImportError:
        print("Warning: wordfreq not installed. Using uniform distribution.")
        print("Install with: pip install wordfreq")
        return {w: 1.0 for w in lexicon}

    frequencies = {}
    for word in lexicon:
        if use_zipf:
            # Zipf frequency: ranges from 0 (very rare) to ~8 (most common)
            # This provides better spread for language model weighting
            freq = zipf_frequency(word.lower(), 'en')
            # Convert to pseudo-probability: 10^(zipf - 8) gives ~1 for "the"
            frequencies[word] = 10 ** (freq - 8) if freq > 0 else 1e-9
        else:
            # Raw probability (0 to ~0.07 for "the")
            freq = word_frequency(word.lower(), 'en')
            frequencies[word] = max(freq, 1e-9)

    return frequencies


@dataclass
class SHARK2Config:
    """Configuration for SHARK2 decoder parameters."""
    # Default sigma parameters
    sigma_loc: float = 0.3
    sigma_shape: float = 0.1
    sigma_lm: float = 0.5

    # Grid search parameter ranges (for optimization)
    # Note: sigma_lm must be > 0 to use language model; range tuned for COCA frequencies
    sigma_loc_range: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    sigma_shape_range: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    sigma_lm_range: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)


DEFAULT_SHARK2_CONFIG = SHARK2Config()


def normalize_shape_batch(templates: np.ndarray) -> np.ndarray:
    """
    Normalize shapes for a batch of gesture templates.

    Centers gestures at origin and scales by path length for
    position/scale-invariant shape comparison.

    Args:
        templates: Array of shape (n_gestures, seq_len, 2) containing x,y coordinates

    Returns:
        Normalized templates of shape (n_gestures, seq_len, 2)
    """
    # Center at origin
    centered = templates - templates.mean(axis=1, keepdims=True)
    # Path length for each gesture
    diffs = np.diff(centered, axis=1)
    path_lengths = np.sum(np.sqrt(np.sum(diffs**2, axis=2)), axis=1, keepdims=True)
    path_lengths = np.maximum(path_lengths, 1e-6)  # Avoid division by zero
    # Scale by path length
    return centered / path_lengths[:, :, np.newaxis]


def normalize_shape(gesture: np.ndarray) -> np.ndarray:
    """
    Normalize a single gesture shape.

    Args:
        gesture: Array of shape (seq_len, 2) containing x,y coordinates

    Returns:
        Normalized gesture of shape (seq_len, 2)
    """
    return normalize_shape_batch(gesture[np.newaxis])[0]


class SHARK2Decoder:
    """
    SHARK2 multi-channel decoder for word-gesture recognition.

    The decoder matches input gestures against a lexicon of word templates
    using three weighted channels: location, shape, and language model.

    Example usage:
        keyboard = QWERTYKeyboard()
        decoder = SHARK2Decoder(
            lexicon=['the', 'quick', 'brown', 'fox'],
            keyboard=keyboard,
            seq_length=128
        )

        # Decode single gesture
        word = decoder.decode(gesture_xy)

        # Batch decode
        predictions = decoder.decode_batch(gestures_xy)

        # Optimize parameters on training data
        decoder.optimize_parameters(train_gestures, train_labels)
    """

    def __init__(
        self,
        lexicon: List[str],
        keyboard: QWERTYKeyboard,
        seq_length: int = 128,
        config: SHARK2Config = DEFAULT_SHARK2_CONFIG,
        word_frequencies: Optional[Dict[str, float]] = None
    ):
        """
        Initialize SHARK2 decoder.

        Args:
            lexicon: List of words in the vocabulary
            keyboard: QWERTYKeyboard instance for generating word templates
            seq_length: Number of points in gesture sequences
            config: SHARK2Config with sigma parameters
            word_frequencies: Optional dict mapping word -> frequency for language model.
                            If None, uses uniform distribution.
        """
        self.lexicon = list(lexicon)
        self.word_to_idx = {w: i for i, w in enumerate(self.lexicon)}
        self.keyboard = keyboard
        self.seq_length = seq_length
        self.config = config

        # Current parameters
        self._sigma_loc = config.sigma_loc
        self._sigma_shape = config.sigma_shape
        self._sigma_lm = config.sigma_lm

        # Precompute templates: (n_words, seq_len, 2)
        self._templates_xy = np.stack([
            keyboard.get_word_prototype(w, seq_length)[:, :2]
            for w in self.lexicon
        ], axis=0)

        # Precompute normalized shapes
        self._templates_shape = normalize_shape_batch(self._templates_xy)

        # Precompute key centers and indices for location channel
        # Location channel measures distance from gesture to actual key centers
        self._key_centers_list = [keyboard.get_key_centers_for_word(w) for w in self.lexicon]
        self._key_indices_list = [keyboard.get_key_indices(w, seq_length) for w in self.lexicon]

        # Pad to fixed size for vectorized operations
        self._max_keys = max(len(kc) for kc in self._key_centers_list) if self._key_centers_list else 1
        self._key_centers_padded = np.zeros((len(self.lexicon), self._max_keys, 2))
        self._key_indices_padded = np.zeros((len(self.lexicon), self._max_keys), dtype=int)
        self._key_mask = np.zeros((len(self.lexicon), self._max_keys), dtype=bool)

        for i, (kc, ki) in enumerate(zip(self._key_centers_list, self._key_indices_list)):
            n = len(kc)
            if n > 0:
                self._key_centers_padded[i, :n] = kc
                self._key_indices_padded[i, :n] = ki
                self._key_mask[i, :n] = True

        # Language model (log probabilities)
        if word_frequencies is None:
            # Uniform prior
            self._word_log_prob = np.full(len(self.lexicon), np.log(1.0 / len(self.lexicon)))
        else:
            # Normalize frequencies to probabilities
            total = sum(word_frequencies.get(w, 1e-6) for w in self.lexicon)
            self._word_log_prob = np.array([
                np.log(word_frequencies.get(w, 1e-6) / total)
                for w in self.lexicon
            ])

    @property
    def n_words(self) -> int:
        """Number of words in lexicon."""
        return len(self.lexicon)

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """Current (sigma_loc, sigma_shape, sigma_lm) parameters."""
        return (self._sigma_loc, self._sigma_shape, self._sigma_lm)

    def set_parameters(
        self,
        sigma_loc: float,
        sigma_shape: float,
        sigma_lm: float
    ) -> None:
        """
        Set decoder parameters.

        Args:
            sigma_loc: Location channel standard deviation
            sigma_shape: Shape channel standard deviation
            sigma_lm: Language model weight
        """
        self._sigma_loc = sigma_loc
        self._sigma_shape = sigma_shape
        self._sigma_lm = sigma_lm

    def decode(
        self,
        gesture_xy: np.ndarray,
        return_scores: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Decode a single gesture to a word.

        Args:
            gesture_xy: Gesture array of shape (seq_len, 2) with x,y coordinates
            return_scores: If True, return all word scores along with prediction

        Returns:
            If return_scores=False: Predicted word string
            If return_scores=True: Tuple of (predicted_word, scores_array)
        """
        results = self.decode_batch(gesture_xy[np.newaxis], return_indices=False)
        if return_scores:
            _, _, scores = self._compute_scores_single(gesture_xy)
            return results[0], scores
        return results[0]

    def decode_batch(
        self,
        gestures_xy: np.ndarray,
        return_indices: bool = False
    ) -> Union[List[str], List[int]]:
        """
        Decode multiple gestures using vectorized operations.

        Args:
            gestures_xy: Array of shape (n_gestures, seq_len, 2) with x,y coordinates
            return_indices: If True, return word indices instead of strings

        Returns:
            List of predicted words (or indices if return_indices=True)
        """
        n_gestures = len(gestures_xy)

        # Normalize gesture shapes
        gestures_shape = normalize_shape_batch(gestures_xy)

        predictions = []
        for i in range(n_gestures):
            g_xy = gestures_xy[i]  # (seq_len, 2)
            g_shape = gestures_shape[i]  # (seq_len, 2)

            # Location channel: L2 distance to all templates (full trajectory comparison)
            loc_dists = np.sqrt(np.mean((self._templates_xy - g_xy)**2, axis=(1, 2)))
            loc_scores = -loc_dists**2 / (2 * self._sigma_loc**2)

            # Shape channel: L2 distance to all normalized templates
            shape_dists = np.sqrt(np.mean((self._templates_shape - g_shape)**2, axis=(1, 2)))
            shape_scores = -shape_dists**2 / (2 * self._sigma_shape**2)

            # Total score
            scores = loc_scores + shape_scores + self._sigma_lm * self._word_log_prob

            # Best word
            best_idx = np.argmax(scores)
            predictions.append(best_idx if return_indices else self.lexicon[best_idx])

        return predictions

    def _compute_scores_single(
        self,
        gesture_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute individual channel scores for a gesture."""
        g_shape = normalize_shape(gesture_xy)

        # Location channel
        loc_dists = np.sqrt(np.mean((self._templates_xy - gesture_xy)**2, axis=(1, 2)))
        loc_scores = -loc_dists**2 / (2 * self._sigma_loc**2)

        # Shape channel
        shape_dists = np.sqrt(np.mean((self._templates_shape - g_shape)**2, axis=(1, 2)))
        shape_scores = -shape_dists**2 / (2 * self._sigma_shape**2)

        # Total
        total_scores = loc_scores + shape_scores + self._sigma_lm * self._word_log_prob

        return loc_scores, shape_scores, total_scores

    def get_top_k(
        self,
        gesture_xy: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-k word predictions with scores.

        Args:
            gesture_xy: Gesture array of shape (seq_len, 2)
            k: Number of top predictions to return

        Returns:
            List of (word, score) tuples sorted by score descending
        """
        _, _, scores = self._compute_scores_single(gesture_xy)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [(self.lexicon[i], scores[i]) for i in top_indices]

    def optimize_parameters(
        self,
        gestures: List[np.ndarray],
        labels: List[str],
        config: Optional[SHARK2Config] = None,
        max_samples: int = 200,
        verbose: bool = True
    ) -> Tuple[float, float, float]:
        """
        Optimize sigma parameters using grid search on training data.

        Args:
            gestures: List of gesture arrays, each (seq_len, 2) or (seq_len, 3)
            labels: List of ground truth word labels
            config: Optional config with search ranges (uses self.config if None)
            max_samples: Maximum samples to use for optimization (for speed)
            verbose: If True, print progress

        Returns:
            Tuple of (best_sigma_loc, best_sigma_shape, best_sigma_lm)

        Note:
            Updates internal parameters to best found values.
        """
        config = config or self.config

        # Use subset for speed
        if len(gestures) > max_samples:
            indices = np.random.choice(len(gestures), max_samples, replace=False)
            gestures = [gestures[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Extract xy coordinates (handle both 2D and 3D gestures)
        gestures_xy = np.array([g[:, :2] if g.shape[1] > 2 else g for g in gestures])
        true_indices = [self.word_to_idx[w] for w in labels]

        best_wer = 1.0
        best_params = (config.sigma_loc_range[0], config.sigma_shape_range[0], config.sigma_lm_range[0])

        for sigma_loc in config.sigma_loc_range:
            for sigma_shape in config.sigma_shape_range:
                for sigma_lm in config.sigma_lm_range:
                    self.set_parameters(sigma_loc, sigma_shape, sigma_lm)
                    pred_indices = self.decode_batch(gestures_xy, return_indices=True)
                    wer = sum(p != t for p, t in zip(pred_indices, true_indices)) / len(labels)

                    if wer < best_wer:
                        best_wer = wer
                        best_params = (sigma_loc, sigma_shape, sigma_lm)
                        if verbose:
                            print(f'  New best: loc={sigma_loc}, shape={sigma_shape}, lm={sigma_lm}, WER={wer*100:.1f}%')

        self.set_parameters(*best_params)
        return best_params


def evaluate_decoder(
    decoder: SHARK2Decoder,
    gestures: List[np.ndarray],
    labels: List[str],
    batch_size: int = 500,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate decoder on a test set.

    Args:
        decoder: SHARK2Decoder instance
        gestures: List of gesture arrays
        labels: List of ground truth word labels
        batch_size: Batch size for processing
        verbose: If True, print progress

    Returns:
        Dictionary with evaluation metrics:
        - 'wer': Word error rate
        - 'accuracy': Recognition accuracy (1 - wer)
        - 'n_correct': Number of correct predictions
        - 'n_total': Total number of samples
    """
    # Extract xy coordinates
    gestures_xy = np.array([g[:, :2] if g.shape[1] > 2 else g for g in gestures])

    total_errors = 0
    for batch_start in range(0, len(gestures), batch_size):
        batch_end = min(batch_start + batch_size, len(gestures))
        batch_gestures = gestures_xy[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]

        predictions = decoder.decode_batch(batch_gestures)
        batch_errors = sum(p != t for p, t in zip(predictions, batch_labels))
        total_errors += batch_errors

        if verbose:
            current_wer = total_errors / batch_end
            print(f'  Evaluated {batch_end}/{len(gestures)}, current WER: {current_wer*100:.1f}%')

    wer = total_errors / len(gestures)
    return {
        'wer': wer,
        'accuracy': 1.0 - wer,
        'n_correct': len(gestures) - total_errors,
        'n_total': len(gestures)
    }
