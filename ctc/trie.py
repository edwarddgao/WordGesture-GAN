"""Trie data structure for vocabulary-constrained CTC decoding."""

from __future__ import annotations

from typing import Dict, List, Optional, Set


class TrieNode:
    """Node in the vocabulary trie."""

    __slots__ = ["children", "is_word", "word"]

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.word: Optional[str] = None


class Trie:
    """Trie for efficient vocabulary prefix lookup.

    Used to constrain CTC beam search to only produce valid vocabulary words.
    """

    def __init__(self, vocabulary: List[str] | None = None):
        """Initialize the trie.

        Args:
            vocabulary: List of words to insert. If None, creates empty trie.
        """
        self.root = TrieNode()
        self._size = 0

        if vocabulary:
            for word in vocabulary:
                self.insert(word)

    def insert(self, word: str) -> None:
        """Insert a word into the trie.

        Args:
            word: Word to insert (will be lowercased).
        """
        word = word.lower()
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            node.word = word
            self._size += 1

    def search(self, word: str) -> bool:
        """Check if exact word exists in trie.

        Args:
            word: Word to search for.

        Returns:
            True if word is in vocabulary.
        """
        node = self._get_node(word.lower())
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix.

        Args:
            prefix: Prefix to check.

        Returns:
            True if prefix exists in trie.
        """
        return self._get_node(prefix.lower()) is not None

    def get_node(self, prefix: str) -> Optional[TrieNode]:
        """Get the trie node for a given prefix.

        Args:
            prefix: Prefix string.

        Returns:
            TrieNode if prefix exists, None otherwise.
        """
        return self._get_node(prefix.lower())

    def _get_node(self, prefix: str) -> Optional[TrieNode]:
        """Internal method to traverse to a node."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_valid_next_chars(self, prefix: str) -> Set[str]:
        """Get all valid next characters from a prefix.

        Args:
            prefix: Current prefix.

        Returns:
            Set of valid next characters.
        """
        node = self._get_node(prefix.lower())
        if node is None:
            return set()
        return set(node.children.keys())

    def __len__(self) -> int:
        """Return number of words in trie."""
        return self._size

    def __contains__(self, word: str) -> bool:
        """Check if word is in trie."""
        return self.search(word)
