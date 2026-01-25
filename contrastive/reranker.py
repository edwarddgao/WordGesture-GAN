"""Gemini-based reranker for gesture recognition candidates."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


RERANK_PROMPT_TEMPLATE = """You are proofreading swipe keyboard output. Select words to form a grammatically correct English sentence.

Current: {top1_sequence}

Candidates (* = gesture system's choice):
{candidates_formatted}

Carefully read the sentence. For each position, select the word that makes the sentence grammatically correct and natural. Output one word per line:
"""

RERANK_PROMPT_TEMPLATE_WITH_WRITEIN = """You are proofreading swipe keyboard output. Select words to form a grammatically correct English sentence.

Current: {top1_sequence}

Candidates (* = gesture system's choice):
{candidates_formatted}

Carefully read the sentence. For each position:
- Pick from the candidates if one fits grammatically
- If NO candidate makes sense, suggest a common English word with a SIMILAR SHAPE on a QWERTY keyboard (same general left-to-right swipe pattern)

Output one word per line:
"""


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
        allow_writein: bool = False,
        max_candidates_display: int = 10,
    ):
        """Initialize the Gemini reranker.

        Args:
            project: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            location: GCP location. If None, uses GOOGLE_CLOUD_LOCATION env var or 'global'.
            model: Model name. Defaults to gemini-3-flash-preview.
            max_retries: Maximum retries on API errors.
            retry_delay: Base delay between retries (exponential backoff).
            max_concurrent: Maximum concurrent API calls for async batch processing.
            allow_writein: If True, allow LLM to suggest words not in candidates.
            max_candidates_display: Maximum candidates to show in prompt per position.
        """
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
        self.allow_writein = allow_writein
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
        template = RERANK_PROMPT_TEMPLATE_WITH_WRITEIN if self.allow_writein else RERANK_PROMPT_TEMPLATE
        return template.format(
            top1_sequence=top1_sequence,
            candidates_formatted=candidates_formatted,
        )

    def _parse_response(
        self,
        response_text: str,
        n_positions: int,
        candidates: List[List[Tuple[str, float]]],
        word_validator: Optional[Callable[[str, int], Optional[str]]] = None,
    ) -> List[str]:
        """Parse LLM response into selected words.

        Args:
            response_text: Raw LLM response.
            n_positions: Expected number of positions.
            candidates: Original candidates (for fallback).
            word_validator: Optional callback to validate write-in words.
                Takes (word, position_idx) and returns the word if valid, or None.

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
                elif self.allow_writein and word_validator:
                    # Try to validate write-in word
                    validated = word_validator(word, i)
                    if validated:
                        selected.append(validated)
                    else:
                        # Fallback to top candidate
                        selected.append(candidates[i][0][0])
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
        word_validator: Optional[Callable[[str, int], Optional[str]]] = None,
    ) -> List[str]:
        """Rerank candidates for each position using Gemini.

        Args:
            candidates: List of (word, score) tuples per position,
                sorted by score descending.
            word_validator: Optional callback to validate write-in words.
                Takes (word, position_idx) and returns the word if valid, or None.

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
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Deterministic for consistent reranking
                        max_output_tokens=256,
                    ),
                )
                return self._parse_response(response.text, len(candidates), candidates, word_validator)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries failed, fall back to top-1
        print(f"Warning: Reranker failed after {self.max_retries} attempts: {last_error}")
        return [cands[0][0] for cands in candidates]

    def rerank_batch(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        word_validators: Optional[List[Callable[[str, int], Optional[str]]]] = None,
    ) -> List[List[str]]:
        """Rerank multiple sentences (sequential).

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            word_validators: Optional list of validators, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        if word_validators:
            return [self.rerank(c, v) for c, v in zip(batch_candidates, word_validators)]
        return [self.rerank(candidates) for candidates in batch_candidates]

    async def rerank_async(
        self,
        candidates: List[List[Tuple[str, float]]],
        word_validator: Optional[Callable[[str, int], Optional[str]]] = None,
    ) -> List[str]:
        """Async version of rerank using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rerank(candidates, word_validator)
        )

    async def rerank_batch_async(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        word_validators: Optional[List[Callable[[str, int], Optional[str]]]] = None,
    ) -> List[List[str]]:
        """Rerank multiple sentences concurrently.

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            word_validators: Optional list of validators, one per sentence.

        Returns:
            List of selected word lists, one per sentence.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_rerank(
            candidates: List[List[Tuple[str, float]]],
            validator: Optional[Callable[[str, int], Optional[str]]] = None,
        ) -> List[str]:
            async with semaphore:
                return await self.rerank_async(candidates, validator)

        if word_validators:
            tasks = [limited_rerank(c, v) for c, v in zip(batch_candidates, word_validators)]
        else:
            tasks = [limited_rerank(c) for c in batch_candidates]
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
