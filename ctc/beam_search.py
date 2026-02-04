"""CTC beam search with trie constraint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .trie import Trie, TrieNode


@dataclass
class Beam:
    """A single beam hypothesis in CTC decoding.

    Attributes:
        prefix: The decoded character sequence so far.
        trie_node: Current position in the vocabulary trie.
        log_prob_blank: Log probability ending in blank.
        log_prob_non_blank: Log probability ending in non-blank.
    """

    prefix: str
    trie_node: TrieNode
    log_prob_blank: float = float("-inf")
    log_prob_non_blank: float = float("-inf")

    @property
    def log_prob_total(self) -> float:
        """Total log probability (logsumexp of blank and non-blank)."""
        return np.logaddexp(self.log_prob_blank, self.log_prob_non_blank)


def ctc_beam_search_trie(
    log_probs: np.ndarray,
    trie: Trie,
    beam_width: int = 100,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """CTC beam search constrained to vocabulary trie.

    Args:
        log_probs: (T, 27) array of log probabilities.
            Index 0 = blank, indices 1-26 = a-z.
        trie: Vocabulary trie for constraint.
        beam_width: Number of beams to maintain at each timestep.
        top_k: Number of top candidates to return.

    Returns:
        List of (word, score) tuples sorted by score descending.
        Score is the log probability of the path.
    """
    T = log_probs.shape[0]

    # Initialize with empty prefix at trie root
    beams: Dict[str, Beam] = {"": Beam(prefix="", trie_node=trie.root, log_prob_blank=0.0)}

    for t in range(T):
        new_beams: Dict[str, Beam] = {}

        for prefix, beam in beams.items():
            node = beam.trie_node

            # Case 1: Emit blank (stay at same prefix)
            blank_log_prob = log_probs[t, 0]
            new_blank_prob = beam.log_prob_total + blank_log_prob

            if prefix in new_beams:
                new_beams[prefix].log_prob_blank = np.logaddexp(
                    new_beams[prefix].log_prob_blank, new_blank_prob
                )
            else:
                new_beams[prefix] = Beam(
                    prefix=prefix,
                    trie_node=node,
                    log_prob_blank=new_blank_prob,
                )

            # Case 2: Emit character (extend prefix if valid in trie)
            for char_idx in range(1, 27):
                char = chr(ord("a") + char_idx - 1)

                # Check trie constraint
                if char not in node.children:
                    continue

                char_log_prob = log_probs[t, char_idx]
                new_node = node.children[char]
                new_prefix = prefix + char

                # CTC rule: repeating same char requires blank in between
                if prefix and char == prefix[-1]:
                    # Must come from blank state
                    new_non_blank_prob = beam.log_prob_blank + char_log_prob
                else:
                    # Can come from either blank or non-blank
                    new_non_blank_prob = beam.log_prob_total + char_log_prob

                if new_prefix in new_beams:
                    new_beams[new_prefix].log_prob_non_blank = np.logaddexp(
                        new_beams[new_prefix].log_prob_non_blank,
                        new_non_blank_prob,
                    )
                else:
                    new_beams[new_prefix] = Beam(
                        prefix=new_prefix,
                        trie_node=new_node,
                        log_prob_non_blank=new_non_blank_prob,
                    )

        # Prune to beam_width
        if len(new_beams) > beam_width:
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: x[1].log_prob_total,
                reverse=True,
            )
            beams = dict(sorted_beams[:beam_width])
        else:
            beams = new_beams

    # Filter to complete words only
    complete_words = [
        (beam.prefix, beam.log_prob_total) for beam in beams.values() if beam.trie_node.is_word
    ]

    # Sort by score and return top-k
    complete_words.sort(key=lambda x: x[1], reverse=True)
    return complete_words[:top_k]
