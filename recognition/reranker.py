"""OpenRouter-based reranker for gesture recognition candidates."""

from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


def _make_words_schema(n_words: int) -> dict:
    """Create JSON schema for exactly n words."""
    return {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n_words,
                "maxItems": n_words,
                "description": f"Exactly {n_words} corrected words in order",
            }
        },
        "required": ["words"],
        "additionalProperties": False,
    }


@dataclass
class RerankerResult:
    """Result from reranking with debugging info."""
    predictions: List[str]
    raw_response: str = ""
    parse_details: List[dict] = field(default_factory=list)

RERANK_PROMPT_TEMPLATE = """You are fixing swipe keyboard output. For each position you have:
- Decoded: direct character-by-character decode of the gesture (may contain errors but preserves intended spelling)
- Candidates: words from vocabulary ranked by gesture similarity

Select the best word for each position. Use sentence context to pick words that form grammatical sentences. IMPORTANT: If Decoded matches a word in Candidates, strongly prefer it - especially for names and less common words.

Example 1:
Position 1: Decoded: "are" | Candidates: *"ate", "are", "agree"
Position 2: Decoded: "yoy" | Candidates: *"toy", "you", "youth"
Position 3: Decoded: "going" | Candidates: *"found", "going", "find"
Position 4: Decoded: "to" | Candidates: *"tip", "too", "top"
Position 5: Decoded: "cal" | Candidates: *"call", "cal", "carl"
Answer:
are
you
going
to
call

Example 2:
Position 1: Decoded: "thursay" | Candidates: *"thursday", "tuesday", "treasury"
Position 2: Decoded: "works" | Candidates: *"world", "worlds", "works"
Position 3: Decoded: "beter" | Candidates: *"better", "brett", "bet"
Position 4: Decoded: "for" | Candidates: *"four", "for", "foot"
Position 5: Decoded: "me" | Candidates: *"me", "ne", "mere"
Answer:
thursday
works
better
for
me

Example 3:
Position 1: Decoded: "no" | Candidates: *"no", "ni", "boo"
Position 2: Decoded: "suprise" | Candidates: *"surprise", "suppose", "super"
Position 3: Decoded: "there" | Candidates: *"threw", "three", "there"
Answer:
no
surprise
there

Now fix this:
{candidates_formatted}

Answer:
"""


def _load_optional_deps():
    """Load optional dependencies for OpenRouter reranker.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        from openai import AsyncOpenAI, OpenAI

        return OpenAI, AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenRouterReranker requires openai. "
            "Install with: pip install openai python-dotenv\n"
            "See README.md for setup instructions."
        ) from e


class OpenRouterReranker:
    """Reranker using OpenRouter API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "google/gemini-3-flash-preview",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent: int = 100,
        max_candidates_display: int = 10,
        load_dotenv: bool = True,
    ):
        """Initialize the OpenRouter reranker.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            model: Model name. Defaults to google/gemini-3-flash-preview.
            max_retries: Maximum retries on API errors.
            retry_delay: Base delay between retries (exponential backoff).
            max_concurrent: Maximum concurrent API calls for async batch processing.
            max_candidates_display: Maximum candidates to show in prompt per position.
            load_dotenv: If True, load .env file from project root if it exists.
        """
        # Load optional dependencies
        OpenAI, AsyncOpenAI = _load_optional_deps()

        # Optionally load .env file
        if load_dotenv:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                try:
                    from dotenv import load_dotenv as _load_dotenv

                    _load_dotenv(env_path)
                except ImportError:
                    pass  # dotenv is optional if env vars are set directly

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY env var or pass api_key="
            )

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent
        self.max_candidates_display = max_candidates_display

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def _format_candidates(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str],
        max_per_position: int | None = None,
    ) -> str:
        """Format candidates for the prompt.

        Args:
            candidates: List of (word, score) tuples per position.
            ctc_words: CTC decoded words per position.
            max_per_position: Maximum candidates to include per position.

        Returns:
            Formatted string for prompt.
        """
        if max_per_position is None:
            max_per_position = self.max_candidates_display
        lines = []
        for i, position_candidates in enumerate(candidates, start=1):
            top_candidates = position_candidates[:max_per_position]
            # Format candidates with * for top candidate
            parts = []
            for j, (word, score) in enumerate(top_candidates):
                if j == 0:
                    parts.append(f'*"{word}"')
                else:
                    parts.append(f'"{word}"')
            cands_str = ", ".join(parts)
            ctc_word = ctc_words[i - 1] if i - 1 < len(ctc_words) else ""
            ctc_word = ctc_word or ""  # Handle None
            lines.append(f'Position {i}: Decoded: "{ctc_word}" | Candidates: {cands_str}')
        return "\n".join(lines)

    def _build_prompt(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str],
    ) -> str:
        """Build the full prompt for reranking."""
        candidates_formatted = self._format_candidates(candidates, ctc_words)
        return RERANK_PROMPT_TEMPLATE.format(candidates_formatted=candidates_formatted)

    def _parse_response(
        self,
        response_text: str,
        n_positions: int,
        candidates: List[List[Tuple[str, float]]],
    ) -> Tuple[List[str], List[dict]]:
        """Parse LLM response into selected words with debugging details.

        Args:
            response_text: Raw LLM response.
            n_positions: Expected number of positions.
            candidates: Original candidates (for fallback on missing lines only).

        Returns:
            Tuple of (selected words, parse details per position).
        """
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        selected = []
        details = []

        for i in range(n_positions):
            detail = {"position": i}

            if i < len(lines):
                raw_line = lines[i]
                detail["raw_line"] = raw_line

                word = raw_line.strip().strip('"').lower()
                # Clean up any numbering or punctuation
                word = word.lstrip("0123456789.-) ")
                word = word.rstrip(".,;:!?")
                if " " in word:
                    word = word.split()[0]  # Take first word if multiple
                # Normalize contractions: don't -> dont, i'll -> ill, etc.
                word = word.replace("'", "")
                detail["normalized"] = word

                # Trust the LLM output - no validation against candidates
                # This allows LLM to fix CTC errors and suggest OOV words
                selected.append(word)
                detail["fallback"] = False
            else:
                # Missing line only - use top candidate
                fallback = candidates[i][0][0].replace("'", "")
                selected.append(fallback)
                detail["raw_line"] = None
                detail["normalized"] = None
                detail["fallback"] = True
                detail["fallback_reason"] = "missing_line"
                detail["fallback_to"] = fallback

            details.append(detail)

        return selected, details

    def rerank(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str],
    ) -> RerankerResult:
        """Rerank candidates for each position using OpenRouter.

        Args:
            candidates: List of (word, score) tuples per position,
                sorted by score descending.
            ctc_words: CTC decoded words per position.

        Returns:
            RerankerResult with predictions and debugging info.
        """
        if not candidates:
            return RerankerResult(predictions=[])

        prompt = self._build_prompt(candidates, ctc_words)

        # Retry loop with exponential backoff
        last_error = None
        n_words = len(candidates)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "words_response",
                            "strict": True,
                            "schema": _make_words_schema(n_words),
                        }
                    },
                )
                # Parse structured JSON response
                raw_text = response.choices[0].message.content
                try:
                    data = json.loads(raw_text)
                    words = data.get("words", [])
                except json.JSONDecodeError:
                    words = []

                predictions, parse_details = self._parse_response(
                    "\n".join(words), n_words, candidates
                )
                return RerankerResult(
                    predictions=predictions,
                    raw_response=raw_text,
                    parse_details=parse_details,
                )
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
        return RerankerResult(
            predictions=[cands[0][0] for cands in candidates],
            raw_response="",
            parse_details=[{"position": i, "fallback": True, "fallback_reason": "api_error"}
                          for i in range(len(candidates))],
        )

    def rerank_batch(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        batch_ctc_words: List[List[str]],
    ) -> List[RerankerResult]:
        """Rerank multiple sentences (sequential).

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            batch_ctc_words: CTC word lists, one per sentence.

        Returns:
            List of RerankerResult, one per sentence.
        """
        return [
            self.rerank(candidates, ctc_words)
            for candidates, ctc_words in zip(batch_candidates, batch_ctc_words)
        ]

    async def rerank_async(
        self,
        candidates: List[List[Tuple[str, float]]],
        ctc_words: List[str],
    ) -> RerankerResult:
        """Async version of rerank using native async API."""
        if not candidates:
            return RerankerResult(predictions=[])

        prompt = self._build_prompt(candidates, ctc_words)
        n_words = len(candidates)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "words_response",
                            "strict": True,
                            "schema": _make_words_schema(n_words),
                        }
                    },
                )
                raw_text = response.choices[0].message.content
                try:
                    data = json.loads(raw_text)
                    words = data.get("words", [])
                except json.JSONDecodeError:
                    words = []

                predictions, parse_details = self._parse_response(
                    "\n".join(words), n_words, candidates
                )
                return RerankerResult(
                    predictions=predictions,
                    raw_response=raw_text,
                    parse_details=parse_details,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        warnings.warn(
            f"Reranker failed after {self.max_retries} attempts: {last_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return RerankerResult(
            predictions=[cands[0][0] for cands in candidates],
            raw_response="",
            parse_details=[{"position": i, "fallback": True, "fallback_reason": "api_error"}
                          for i in range(len(candidates))],
        )

    async def rerank_batch_async(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        batch_ctc_words: List[List[str]],
    ) -> List[RerankerResult]:
        """Rerank multiple sentences concurrently.

        Args:
            batch_candidates: List of candidate lists, one per sentence.
            batch_ctc_words: CTC word lists, one per sentence.

        Returns:
            List of RerankerResult, one per sentence.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_rerank(
            candidates: List[List[Tuple[str, float]]],
            ctc_words: List[str],
        ) -> RerankerResult:
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
        ctc_words: List[str] | None = None,  # Ignored, for API compat
    ) -> RerankerResult:
        """Return top-1 candidate for each position."""
        predictions = [cands[0][0] for cands in candidates]
        return RerankerResult(
            predictions=predictions,
            raw_response="",
            parse_details=[{"position": i, "fallback": False, "top1": True}
                          for i in range(len(candidates))],
        )

    def rerank_batch(
        self,
        batch_candidates: List[List[List[Tuple[str, float]]]],
        batch_ctc_words: List[List[str]] | None = None,  # Ignored, for API compat
    ) -> List[RerankerResult]:
        """Return top-1 candidates for each sentence."""
        if batch_ctc_words is None:
            batch_ctc_words = [None] * len(batch_candidates)
        return [
            self.rerank(candidates, ctc_words)
            for candidates, ctc_words in zip(batch_candidates, batch_ctc_words)
        ]
