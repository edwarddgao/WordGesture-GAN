"""Gemini-based reranker for gesture recognition candidates."""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

RERANK_PROMPT_TEMPLATE = """You are proofreading swipe keyboard output. Select words to form a grammatically correct English sentence.

Current: {top1_sequence}

Candidates (* = gesture system's choice):
{candidates_formatted}

Carefully read the sentence. For each position, select the word that makes the sentence grammatically correct and natural. Output one word per line:
"""

INCREMENTAL_PROMPT_TEMPLATE = """You are proofreading swipe keyboard output. Select the next word.

Partial sentence so far: {context}

Candidates for next word (* = gesture system's top choice):
{candidates}

Select ONE word that continues the sentence grammatically and naturally. Output only the word:"""


def _load_optional_deps():
    """Load optional dependencies for Gemini reranker.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        from google import genai
        from google.genai import types

        return genai, types
    except ImportError as e:
        raise ImportError(
            "GeminiReranker requires google-genai and python-dotenv. "
            "Install with: pip install google-genai>=1.51.0 python-dotenv\n"
            "See README.md for setup instructions."
        ) from e


class GeminiReranker:
    """Reranker using Gemini 3.0 Flash via Vertex AI."""

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent: int = 10,
        max_candidates_display: int = 10,
        load_dotenv: bool = True,
    ):
        """Initialize the Gemini reranker.

        Args:
            project: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            location: GCP location. If None, uses GOOGLE_CLOUD_LOCATION env var or 'global'.
            model: Model name. Defaults to gemini-3-flash-preview.
            max_retries: Maximum retries on API errors.
            retry_delay: Base delay between retries (exponential backoff).
            max_concurrent: Maximum concurrent API calls for async batch processing.
            max_candidates_display: Maximum candidates to show in prompt per position.
            load_dotenv: If True, load .env file from project root if it exists.
        """
        # Load optional dependencies
        genai, types = _load_optional_deps()
        self._types = types

        # Optionally load .env file
        if load_dotenv:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                try:
                    from dotenv import load_dotenv as _load_dotenv

                    _load_dotenv(env_path)
                except ImportError:
                    pass  # dotenv is optional if env vars are set directly

        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError(
                "Project ID required. Set GOOGLE_CLOUD_PROJECT env var or pass project="
            )

        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent
        self.max_candidates_display = max_candidates_display
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

    def _format_candidates(
        self,
        candidates: List[List[Tuple[str, float]]],
        max_per_position: int | None = None,
    ) -> str:
        """Format candidates for the prompt.

        Args:
            candidates: List of (word, score) tuples per position.
            max_per_position: Maximum candidates to include per position.

        Returns:
            Formatted string for prompt.
        """
        if max_per_position is None:
            max_per_position = self.max_candidates_display
        lines = []
        for i, position_candidates in enumerate(candidates, start=1):
            # Take top candidates
            top_candidates = position_candidates[:max_per_position]
            # Format: Position 1: *"word1" (0.92), "word2" (0.87), ...
            # Mark top candidate with asterisk
            parts = []
            for j, (word, score) in enumerate(top_candidates):
                if j == 0:
                    parts.append(f'*"{word}" ({score:.2f})')
                else:
                    parts.append(f'"{word}" ({score:.2f})')
            lines.append(f"Position {i}: {', '.join(parts)}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        candidates: List[List[Tuple[str, float]]],
    ) -> str:
        """Build the full prompt for reranking."""
        top1_sequence = " ".join(cands[0][0] for cands in candidates)
        candidates_formatted = self._format_candidates(candidates)
        return RERANK_PROMPT_TEMPLATE.format(
            top1_sequence=top1_sequence,
            candidates_formatted=candidates_formatted,
        )

    def _parse_response(
        self,
        response_text: str,
        n_positions: int,
        candidates: List[List[Tuple[str, float]]],
    ) -> List[str]:
        """Parse LLM response into selected words.

        Args:
            response_text: Raw LLM response.
            n_positions: Expected number of positions.
            candidates: Original candidates (for fallback).

        Returns:
            List of selected words, one per position.
        """
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        selected = []

        for i in range(n_positions):
            if i < len(lines):
                word = lines[i].strip().strip('"').strip("'").lower()
                # Validate word is in candidates
                valid_words = {w for w, _ in candidates[i]}
                if word in valid_words:
                    selected.append(word)
                else:
                    # Fallback to top candidate
                    selected.append(candidates[i][0][0])
            else:
                # Missing line, use top candidate
                selected.append(candidates[i][0][0])

        return selected

    def rerank(
        self,
        candidates: List[List[Tuple[str, float]]],
    ) -> List[str]:
        """Rerank candidates for each position using Gemini.

        Args:
            candidates: List of (word, score) tuples per position,
                sorted by score descending.

        Returns:
            List of selected words, one per position.
        """
        if not candidates:
            return []

        prompt = self._build_prompt(candidates)

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=0.0,  # Deterministic for consistent reranking
                        max_output_tokens=256,
                    ),
                )
                return self._parse_response(response.text, len(candidates), candidates)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries failed, fall back to top-1
        warnings.warn(
            f"Reranker failed after {self.max_retries} attempts: {last_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return [cands[0][0] for cands in candidates]

    def rerank_batch(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
    ) -> List[List[str]]:
        """Rerank multiple sentences (sequential).

        Args:
            batch_candidates: List of candidate lists, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        return [self.rerank(candidates) for candidates in batch_candidates]

    async def rerank_async(
        self,
        candidates: List[List[Tuple[str, float]]],
    ) -> List[str]:
        """Async version of rerank using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rerank(candidates)
        )

    async def rerank_batch_async(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
    ) -> List[List[str]]:
        """Rerank multiple sentences concurrently.

        Args:
            batch_candidates: List of candidate lists, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_rerank(
            candidates: List[List[Tuple[str, float]]],
        ) -> List[str]:
            async with semaphore:
                return await self.rerank_async(candidates)

        tasks = [limited_rerank(c) for c in batch_candidates]
        return await asyncio.gather(*tasks)

    def _build_incremental_prompt(
        self,
        context: str,
        candidates: List[Tuple[str, float]],
    ) -> str:
        """Build prompt for single position with history context.

        Args:
            context: Previously selected words as context.
            candidates: (word, score) tuples for this position.

        Returns:
            Formatted prompt string.
        """
        top_candidates = candidates[: self.max_candidates_display]
        parts = []
        for j, (word, score) in enumerate(top_candidates):
            if j == 0:
                parts.append(f'*"{word}" ({score:.2f})')
            else:
                parts.append(f'"{word}" ({score:.2f})')
        candidates_str = ", ".join(parts)

        return INCREMENTAL_PROMPT_TEMPLATE.format(
            context=context,
            candidates=candidates_str,
        )

    def _get_single_word(
        self,
        prompt: str,
        candidates: List[Tuple[str, float]],
    ) -> str:
        """Get single word selection from LLM.

        Args:
            prompt: Formatted prompt for single position.
            candidates: Valid candidates for validation.

        Returns:
            Selected word (falls back to top-1 on error).
        """
        valid_words = {w for w, _ in candidates}
        top1_word = candidates[0][0]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=32,
                    ),
                )
                # Parse single word from response
                if response.text is None:
                    return top1_word
                word = response.text.strip().strip('"').strip("'").lower()
                # Take first word if multiple returned
                word = word.split()[0] if word else top1_word
                word = word.strip('"').strip("'")

                if word in valid_words:
                    return word
                else:
                    return top1_word
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        warnings.warn(
            f"Incremental reranker failed after {self.max_retries} attempts: {last_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return top1_word

    def rerank_incremental(
        self,
        candidates: List[List[Tuple[str, float]]],
        ground_truth: List[str],
    ) -> List[str]:
        """Rerank word-by-word, using ground truth previous words as context.

        Simulates user correcting mistakes before typing the next word
        (perfect context assumption).

        Args:
            candidates: List of (word, score) tuples per position.
            ground_truth: Ground truth words for perfect context.

        Returns:
            List of selected words, one per position.
        """
        if not candidates:
            return []

        selected = []
        for i, position_candidates in enumerate(candidates):
            # Use ground truth words as context (perfect context assumption)
            context = " ".join(ground_truth[:i]) if i > 0 else "(start of sentence)"
            prompt = self._build_incremental_prompt(context, position_candidates)
            word = self._get_single_word(prompt, position_candidates)
            selected.append(word)

        return selected

    async def _get_single_word_async(
        self,
        prompt: str,
        candidates: List[Tuple[str, float]],
    ) -> str:
        """Async version of _get_single_word using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self._get_single_word(prompt, candidates)
        )

    async def rerank_incremental_async(
        self,
        candidates: List[List[Tuple[str, float]]],
        ground_truth: List[str],
    ) -> List[str]:
        """Async version of rerank_incremental.

        Note: Still processes positions sequentially since each position
        depends on ground_truth context. The async is for non-blocking I/O.
        """
        if not candidates:
            return []

        selected = []
        for i, position_candidates in enumerate(candidates):
            context = " ".join(ground_truth[:i]) if i > 0 else "(start of sentence)"
            prompt = self._build_incremental_prompt(context, position_candidates)
            word = await self._get_single_word_async(prompt, position_candidates)
            selected.append(word)

        return selected

    async def rerank_incremental_batch_async(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        batch_ground_truth: List[List[str]],
    ) -> List[List[str]]:
        """Rerank multiple sentences incrementally with concurrency control.

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            batch_ground_truth: List of ground truth word lists, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_rerank(
            candidates: List[List[Tuple[str, float]]],
            ground_truth: List[str],
        ) -> List[str]:
            async with semaphore:
                return await self.rerank_incremental_async(candidates, ground_truth)

        tasks = [
            limited_rerank(c, gt)
            for c, gt in zip(batch_candidates, batch_ground_truth)
        ]
        return await asyncio.gather(*tasks)


class NoopReranker:
    """Baseline reranker that just returns top-1 candidates."""

    def rerank(
        self,
        candidates: List[List[Tuple[str, float]]],
    ) -> List[str]:
        """Return top-1 candidate for each position."""
        return [cands[0][0] for cands in candidates]

    def rerank_batch(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
    ) -> List[List[str]]:
        """Return top-1 candidates for each sentence."""
        return [self.rerank(candidates) for candidates in batch_candidates]
