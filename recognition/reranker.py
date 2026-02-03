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

RERANK_PROMPT_TEMPLATE = """You are fixing swipe keyboard output. Select the correct word for each position. Output one word per line.

Example:
Sentence: ill call you ibm three morning
Position 1: *"ill", "all"
Position 2: *"call", "caller"
Position 3: *"you", "your"
Position 4: *"ibm", "in", "inn"
Position 5: *"three", "the", "there"
Position 6: *"morning", "modeling"
Answer:
ill
call
you
in
the
morning

Now fix this:
Sentence: {top1_sequence}

{candidates_formatted}

Answer:
"""

RERANK_PROMPT_TEMPLATE_WITH_CTC = """You are fixing swipe keyboard output. For each position you have:
- Decoded: direct character-by-character decode of the gesture (may contain errors but preserves intended spelling)
- Candidates: words from vocabulary ranked by gesture similarity

Select the best word for each position. The decoded word is often correct, especially for names or rare words not in vocabulary. Output one word per line.

Example:
Sentence: ill call you ibm three morning
Position 1: Decoded: "ill" | Candidates: *"ill", "all"
Position 2: Decoded: "call" | Candidates: *"call", "caller"
Position 3: Decoded: "you" | Candidates: *"you", "your"
Position 4: Decoded: "in" | Candidates: *"ibm", "in", "inn"
Position 5: Decoded: "the" | Candidates: *"three", "the", "there"
Position 6: Decoded: "morning" | Candidates: *"morning", "modeling"
Answer:
ill
call
you
in
the
morning

Now fix this:
Sentence: {top1_sequence}

{candidates_formatted}

Answer:
"""


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
        ctc_words: List[str] | None = None,
    ) -> str:
        """Format candidates for the prompt.

        Args:
            candidates: List of (word, score) tuples per position.
            max_per_position: Maximum candidates to include per position.
            ctc_words: Optional list of CTC decoded words per position.

        Returns:
            Formatted string for prompt.
        """
        if max_per_position is None:
            max_per_position = self.max_candidates_display
        lines = []
        for i, position_candidates in enumerate(candidates, start=1):
            # Take top candidates
            top_candidates = position_candidates[:max_per_position]
            # Format candidates part
            parts = []
            for j, (word, score) in enumerate(top_candidates):
                if j == 0:
                    parts.append(f'*"{word}" ({score:.2f})')
                else:
                    parts.append(f'"{word}" ({score:.2f})')
            cands_str = ", ".join(parts)

            if ctc_words is not None:
                ctc_word = ctc_words[i - 1] if i - 1 < len(ctc_words) else ""
                ctc_word = ctc_word or ""  # Handle None
                lines.append(f'Position {i}: Decoded: "{ctc_word}" | Candidates: {cands_str}')
            else:
                lines.append(f"Position {i}: {cands_str}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str] | None = None,
    ) -> str:
        """Build the full prompt for reranking."""
        top1_sequence = " ".join(cands[0][0] for cands in candidates)
        candidates_formatted = self._format_candidates(candidates, ctc_words=ctc_words)

        if ctc_words is not None:
            return RERANK_PROMPT_TEMPLATE_WITH_CTC.format(
                top1_sequence=top1_sequence,
                candidates_formatted=candidates_formatted,
            )
        return RERANK_PROMPT_TEMPLATE.format(
            top1_sequence=top1_sequence,
            candidates_formatted=candidates_formatted,
        )

    def _parse_response(
        self,
        response_text: str,
        n_positions: int,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str] | None = None,
    ) -> List[str]:
        """Parse LLM response into selected words.

        Args:
            response_text: Raw LLM response.
            n_positions: Expected number of positions.
            candidates: Original candidates (for fallback).
            ctc_words: Optional CTC decoded words (also valid choices).

        Returns:
            List of selected words, one per position.
        """
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        selected = []

        for i in range(n_positions):
            if i < len(lines):
                word = lines[i].strip().strip('"').strip("'").lower()
                # Validate word is in candidates or is the CTC decoded word
                valid_words = {w for w, _ in candidates[i]}
                if ctc_words is not None and i < len(ctc_words) and ctc_words[i]:
                    valid_words.add(ctc_words[i])
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
        ctc_words: List[str] | None = None,
    ) -> List[str]:
        """Rerank candidates for each position using Gemini.

        Args:
            candidates: List of (word, score) tuples per position,
                sorted by score descending.
            ctc_words: Optional CTC decoded words per position.

        Returns:
            List of selected words, one per position.
        """
        if not candidates:
            return []

        prompt = self._build_prompt(candidates, ctc_words=ctc_words)

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
                return self._parse_response(response.text, len(candidates), candidates, ctc_words)
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
        batch_ctc_words: List[List[str] | None] | None = None,
    ) -> List[List[str]]:
        """Rerank multiple sentences (sequential).

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            batch_ctc_words: Optional list of CTC word lists, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        if batch_ctc_words is None:
            batch_ctc_words = [None] * len(batch_candidates)
        return [
            self.rerank(candidates, ctc_words)
            for candidates, ctc_words in zip(batch_candidates, batch_ctc_words)
        ]

    async def rerank_async(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str] | None = None,
    ) -> List[str]:
        """Async version of rerank using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rerank(candidates, ctc_words)
        )

    async def rerank_batch_async(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        batch_ctc_words: List[List[str] | None] | None = None,
    ) -> List[List[str]]:
        """Rerank multiple sentences concurrently.

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            batch_ctc_words: Optional list of CTC word lists, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        if batch_ctc_words is None:
            batch_ctc_words = [None] * len(batch_candidates)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_rerank(
            candidates: List[List[Tuple[str, float]]],
            ctc_words: List[str] | None,
        ) -> List[str]:
            async with semaphore:
                return await self.rerank_async(candidates, ctc_words)

        tasks = [
            limited_rerank(c, ctc)
            for c, ctc in zip(batch_candidates, batch_ctc_words)
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
