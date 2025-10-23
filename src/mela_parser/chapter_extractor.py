#!/usr/bin/env python3
"""Async chapter-based recipe extraction from EPUB cookbooks.

Simple parallel extraction using OpenAI's structured output API.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam

from .parse import CookbookRecipes, MelaRecipe

# Configure module logger
logger = logging.getLogger(__name__)

# Global debug directory for this run
DEBUG_DIR: Path | None = None


# ============================================================================
# Custom Exceptions
# ============================================================================


class ChapterExtractionError(Exception):
    """Base exception for chapter extraction errors."""

    pass


class RecipeExtractionError(ChapterExtractionError):
    """Raised when recipe extraction from a chapter fails."""

    pass


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class Chapter:
    """Represents a single EPUB chapter with its markdown content."""

    name: str
    content: str
    index: int

    def __post_init__(self) -> None:
        """Validate chapter data."""
        if not self.name:
            raise ValueError("Chapter name cannot be empty")
        if not self.content:
            logger.warning(f"Chapter '{self.name}' has empty content")


@dataclass
class ExtractionResult:
    """Result of recipe extraction from a single chapter."""

    chapter_name: str
    recipes: list[MelaRecipe]
    error: str | None = None
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return self.error is None

    @property
    def recipe_count(self) -> int:
        """Get number of recipes extracted."""
        return len(self.recipes)


# ============================================================================
# AsyncChapterExtractor: Extract Recipes from Chapters
# ============================================================================


class AsyncChapterExtractor:
    """Asynchronously extracts recipes from individual chapters with retry logic.

    This class provides parallel recipe extraction from multiple chapters using
    async processing with comprehensive retry logic and exponential backoff.

    Attributes:
        client: Async OpenAI client for parallel LLM interactions
        model: Model to use for extraction (default: gpt-5-nano)
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay for exponential backoff (seconds)

    Example:
        >>> extractor = AsyncChapterExtractor(model="gpt-5-nano")
        >>> results = await extractor.extract_from_chapters(
        ...     chapters,
        ...     expected_titles=None,
        ...     max_concurrent=200
        ... )
        >>> for result in results:
        ...     print(f"{result.chapter_name}: {result.recipe_count} recipes")
    """

    # Constants
    DEFAULT_MODEL: Final[str] = "gpt-5-nano"
    MAX_RETRIES: Final[int] = 3
    INITIAL_RETRY_DELAY: Final[float] = 1.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
        initial_retry_delay: float = INITIAL_RETRY_DELAY,
        debug: bool = True,
    ) -> None:
        """Initialize the async chapter extractor.

        Args:
            model: OpenAI model to use for extraction
            max_retries: Maximum retry attempts for failed extractions
            initial_retry_delay: Initial delay in seconds for exponential backoff
            debug: Whether to save debug output to disk
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.debug = debug

        # Initialize debug directory if debug is enabled
        if self.debug:
            global DEBUG_DIR
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            DEBUG_DIR = Path(f"debug/{timestamp}")
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug output will be saved to: {DEBUG_DIR}")

        logger.info(f"Initialized AsyncChapterExtractor with model: {model}")

    async def extract_from_chapters(
        self,
        chapters: list[Chapter],
        expected_titles: list[str] | None = None,
        max_concurrent: int = 5,
    ) -> list[ExtractionResult]:
        """Extract recipes from multiple chapters in parallel.

        Args:
            chapters: List of chapters to process
            expected_titles: Optional list of expected recipe titles (currently unused)
            max_concurrent: Maximum number of concurrent extractions

        Returns:
            List of ExtractionResult objects for each chapter
        """
        logger.info(
            f"Starting extraction from {len(chapters)} chapters (max concurrent: {max_concurrent})"
        )

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(chapter: Chapter) -> ExtractionResult:
            """Extract with concurrency control."""
            async with semaphore:
                return await self.extract_from_chapter(chapter)

        # Process all chapters in parallel (with concurrency limit)
        results = await asyncio.gather(
            *[extract_with_semaphore(ch) for ch in chapters], return_exceptions=True
        )

        # Handle exceptions in results
        final_results = []
        for chapter, result in zip(chapters, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for chapter '{chapter.name}': {result}")
                final_results.append(
                    ExtractionResult(chapter_name=chapter.name, recipes=[], error=str(result))
                )
            else:
                final_results.append(result)

        # Log summary
        total_recipes = sum(r.recipe_count for r in final_results)
        successful = sum(1 for r in final_results if r.is_success)

        logger.info(
            f"Extraction complete - "
            f"Successful: {successful}/{len(chapters)} chapters, "
            f"Total recipes: {total_recipes}"
        )

        return final_results

    async def extract_from_chapter(self, chapter: Chapter) -> ExtractionResult:
        """Extract recipes from a single chapter with retry logic.

        Args:
            chapter: Chapter to extract from

        Returns:
            ExtractionResult with extracted recipes or error information
        """
        retry_count = 0
        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                recipes = await self._extract_with_llm(chapter)

                logger.debug(
                    f"Chapter '{chapter.name}': Extracted {len(recipes)} recipes "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                return ExtractionResult(
                    chapter_name=chapter.name, recipes=recipes, retry_count=retry_count
                )

            except Exception as e:
                retry_count += 1
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    logger.warning(
                        f"Chapter '{chapter.name}' extraction failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Chapter '{chapter.name}' extraction failed "
                        f"after {self.max_retries} attempts: {e}"
                    )

        return ExtractionResult(
            chapter_name=chapter.name, recipes=[], error=last_error, retry_count=retry_count
        )

    async def _extract_with_llm(self, chapter: Chapter) -> list[MelaRecipe]:
        """Use LLM to extract recipes from chapter content with pagination support.

        Args:
            chapter: Chapter to process

        Returns:
            List of all extracted MelaRecipe objects (may require multiple API calls)
        """
        all_recipes = []
        remaining_content = chapter.content
        page = 1
        max_pages = 10  # Safety limit to prevent infinite loops

        while page <= max_pages:
            # Build extraction prompt
            is_continuation = page > 1
            last_title = all_recipes[-1].title if all_recipes else None

            prompt = self._build_pagination_prompt(
                remaining_content, is_continuation=is_continuation, last_title=last_title
            )

            # Save debug input if enabled
            if self.debug and DEBUG_DIR:
                self._save_debug_input(chapter, page, prompt, remaining_content)

            # Call LLM API with structured output
            response = await self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=CookbookRecipes,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
            )

            # Save debug output if enabled
            if self.debug and DEBUG_DIR:
                self._save_debug_output(chapter, page, response)

            # Handle case where parsing failed - stop pagination gracefully
            if response.output_parsed is None:
                logger.warning(
                    f"Chapter '{chapter.name}' page {page}: Got None response, stopping pagination"
                )
                break  # Stop pagination, return recipes collected so far

            batch = response.output_parsed.recipes
            has_more = response.output_parsed.has_more

            # Log batch results
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.debug(
                    f"Chapter '{chapter.name}' page {page} - "
                    f"Input: {getattr(usage, 'input_tokens', 0)} tokens, "
                    f"Output: {getattr(usage, 'output_tokens', 0)} tokens, "
                    f"Recipes: {len(batch)}, "
                    f"has_more: {has_more}"
                )

            # Add batch to results
            all_recipes.extend(batch)

            # Check if we should continue
            if not has_more or not batch:
                logger.info(
                    f"Chapter '{chapter.name}': Completed after {page} page(s), "
                    f"total {len(all_recipes)} recipes"
                )
                break

            # Find continuation point
            last_recipe_title = batch[-1].title
            continuation_pos = self._find_continuation_point(remaining_content, last_recipe_title)

            if continuation_pos == -1:
                logger.warning(
                    f"Chapter '{chapter.name}': Could not find continuation point "
                    f"after '{last_recipe_title}', stopping"
                )
                break

            # Update remaining content for next iteration
            remaining_content = remaining_content[continuation_pos:]
            page += 1

        if page > max_pages:
            logger.warning(
                f"Chapter '{chapter.name}': Hit maximum page limit ({max_pages}), "
                f"extracted {len(all_recipes)} recipes"
            )

        return all_recipes

    def _build_pagination_prompt(
        self, content: str, is_continuation: bool = False, last_title: str | None = None
    ) -> str:
        """Build extraction prompt for initial or continuation request."""
        if is_continuation and last_title:
            return (
                f"Continue extracting recipes from this section, "
                f'starting AFTER the recipe "{last_title}".\n\n'
                "Extract up to 15 complete recipes in order of appearance.\n\n"
                "The schema requires:\n"
                "- At least 1 ingredient with measurements\n"
                "- At least 2 instruction steps\n\n"
                "Copy titles exactly. Preserve ingredient groupings.\n"
                "For images, capture markdown image paths that appear near "
                "each recipe (e.g., '![Recipe Name](../images/pg_65.jpg)').\n"
                "Set has_more=true if more recipes exist after this batch.\n\n"
                f"<section>\n{content}\n</section>"
            )
        else:
            return (
                "Extract complete recipes from this section.\n\n"
                "Extract up to 15 recipes in order of appearance.\n\n"
                "The schema requires:\n"
                "- At least 1 ingredient with measurements\n"
                "- At least 2 instruction steps\n\n"
                "This naturally filters out incomplete content. "
                "Extract anything that meets these requirements.\n\n"
                "Copy titles exactly. Preserve ingredient groupings. "
                "Leave time/yield blank if not stated.\n"
                "For images, capture markdown image paths that appear near "
                "each recipe (e.g., '![Recipe Name](../images/pg_65.jpg)').\n"
                "Set has_more=true if there are more recipes after this batch "
                "that can't fit.\n\n"
                f"<section>\n{content}\n</section>"
            )

    def _find_continuation_point(self, content: str, last_title: str) -> int:
        """Find where to continue extraction after the last recipe.

        Returns position in content after the last title, or -1 if not found.
        """
        # Try to find the first occurrence of the last title in the content
        # (We use find() not rfind() because the content has already been sliced
        # from a previous continuation point, so we want the FIRST occurrence,
        # not the last which might be in a cross-reference or caption)
        pos = content.find(last_title)

        if pos == -1:
            return -1

        # Move past the title to avoid re-extracting it
        return pos + len(last_title)

    def _save_debug_input(self, chapter: Chapter, page: int, prompt: str, content: str) -> None:
        """Save debug information about the input sent to OpenAI."""
        if not DEBUG_DIR:
            return

        # Create chapter directory
        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = DEBUG_DIR / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)

        # Save the prompt
        prompt_file = chapter_dir / f"page_{page:02d}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        # Save the markdown content being processed
        content_file = chapter_dir / f"page_{page:02d}_content.md"
        content_file.write_text(content, encoding="utf-8")

    def _save_debug_output(self, chapter: Chapter, page: int, response: Any) -> None:
        """Save debug information about the response from OpenAI."""
        if not DEBUG_DIR:
            return

        # Create chapter directory
        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = DEBUG_DIR / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)

        # Save the raw response
        response_file = chapter_dir / f"page_{page:02d}_response.json"

        if response.output_parsed is None:
            # Save error information
            response_data = {"error": "output_parsed is None", "raw_response": str(response)}
        else:
            # Save parsed recipes
            response_data = {
                "recipes": [r.model_dump() for r in response.output_parsed.recipes],
                "has_more": response.output_parsed.has_more,
                "recipe_count": len(response.output_parsed.recipes),
            }

            # Add usage info if available
            if hasattr(response, "usage") and response.usage:
                response_data["usage"] = {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }

        response_file.write_text(json.dumps(response_data, indent=2), encoding="utf-8")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core classes
    "AsyncChapterExtractor",
    # Data models
    "Chapter",
    # Exceptions
    "ChapterExtractionError",
    "ExtractionResult",
    "RecipeExtractionError",
]
