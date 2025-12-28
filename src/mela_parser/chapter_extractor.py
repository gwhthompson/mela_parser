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

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import EasyInputMessageParam

from .config import ExtractionConfig
from .parse import ChapterTitles, CookbookRecipes, MelaRecipe

# Configure module logger
logger = logging.getLogger(__name__)


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
    expected_count: int | None = None  # From title enumeration
    titles_found: list[str] | None = None  # Verified titles from Stage 1

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return self.error is None

    @property
    def recipe_count(self) -> int:
        """Get number of recipes extracted."""
        return len(self.recipes)

    @property
    def completeness(self) -> float | None:
        """Calculate extraction completeness (extracted/expected)."""
        if self.expected_count is None or self.expected_count == 0:
            return None
        return self.recipe_count / self.expected_count


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

    # Constants (defaults when no config provided)
    DEFAULT_MODEL: Final[str] = "gpt-5-nano"
    MAX_RETRIES: Final[int] = 3
    INITIAL_RETRY_DELAY: Final[float] = 1.0

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
        initial_retry_delay: float = INITIAL_RETRY_DELAY,
        debug: bool = True,
        debug_dir: Path | None = None,
        use_grounded_extraction: bool = True,
        config: ExtractionConfig | None = None,
    ) -> None:
        """Initialize the async chapter extractor.

        Args:
            client: Async OpenAI client (created if not provided)
            model: OpenAI model to use for extraction
            max_retries: Maximum retry attempts for failed extractions
            initial_retry_delay: Initial delay in seconds for exponential backoff
            debug: Whether to save debug output to disk
            debug_dir: Directory for debug output (auto-created if None)
            use_grounded_extraction: If True, use title-grounded two-stage extraction.
                If False, use legacy pagination-based extraction.
            config: Optional ExtractionConfig for grounded extraction settings.
                If not provided, uses default values.
        """
        self.client = client if client is not None else AsyncOpenAI()
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.debug = debug
        self.use_grounded_extraction = use_grounded_extraction
        self.config = config if config is not None else ExtractionConfig()

        # Initialize debug directory if debug is enabled
        if self.debug:
            if debug_dir is not None:
                self.debug_dir = debug_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.debug_dir = Path(f"debug/{timestamp}")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug output will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = None

        extraction_mode = "grounded (two-stage)" if use_grounded_extraction else "pagination"
        logger.info(
            f"Initialized AsyncChapterExtractor with model: {model}, mode: {extraction_mode}"
        )

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
        final_results: list[ExtractionResult] = []
        for chapter, result in zip(chapters, results, strict=False):
            if isinstance(result, BaseException):
                logger.error(f"Extraction failed for chapter '{chapter.name}': {result}")
                final_results.append(
                    ExtractionResult(chapter_name=chapter.name, recipes=[], error=str(result))
                )
            else:
                # Type narrowing: result is ExtractionResult after BaseException check
                assert isinstance(result, ExtractionResult)
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
        """Extract recipes from a single chapter.

        Uses title-grounded two-stage extraction by default, or falls back to
        pagination-based extraction if use_grounded_extraction=False.

        Args:
            chapter: Chapter to extract from

        Returns:
            ExtractionResult with extracted recipes or error information
        """
        if self.use_grounded_extraction:
            return await self._extract_grounded(chapter)
        else:
            return await self._extract_paginated(chapter)

    async def _extract_grounded(self, chapter: Chapter) -> ExtractionResult:
        """Two-stage title-grounded extraction.

        Stage 1: Enumerate all recipe titles (lightweight)
        Stage 2: Extract each recipe by title (parallel)

        Args:
            chapter: Chapter to extract from

        Returns:
            ExtractionResult with extracted recipes and verification metrics
        """
        # Stage 1: Enumerate titles
        titles = await self.enumerate_titles(chapter)

        # Deduplicate while preserving order (some chapters list titles multiple times)
        original_count = len(titles)
        titles = list(dict.fromkeys(titles))
        if len(titles) < original_count:
            logger.info(
                f"Chapter '{chapter.name}': Deduplicated "
                f"{original_count} -> {len(titles)} unique titles"
            )

        if not titles:
            logger.info(f"Chapter '{chapter.name}': No recipe titles found")
            return ExtractionResult(
                chapter_name=chapter.name,
                recipes=[],
                expected_count=0,
                titles_found=[],
            )

        logger.info(f"Chapter '{chapter.name}': Extracting {len(titles)} recipes in parallel")

        # Stage 2: Extract each recipe (parallel with concurrency control)
        semaphore = asyncio.Semaphore(self.config.extraction_concurrency_per_chapter)

        async def extract_with_semaphore(title: str, index: int) -> MelaRecipe | None:
            async with semaphore:
                return await self.extract_by_title(chapter, title, index)

        tasks = [extract_with_semaphore(title, i) for i, title in enumerate(titles)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful extractions
        recipes: list[MelaRecipe] = []
        failed_titles: list[str] = []
        for title, result in zip(titles, results, strict=False):
            if isinstance(result, BaseException):
                logger.error(f"Chapter '{chapter.name}': Exception extracting '{title}': {result}")
                failed_titles.append(title)
            elif result is None:
                failed_titles.append(title)
            else:
                # Type narrowing: result is MelaRecipe after BaseException/None check
                recipes.append(result)

        # Log summary
        if failed_titles:
            logger.warning(
                f"Chapter '{chapter.name}': Failed to extract {len(failed_titles)} recipes: "
                f"{failed_titles[:3]}{'...' if len(failed_titles) > 3 else ''}"
            )

        completeness = len(recipes) / len(titles) if titles else 0
        logger.info(
            f"Chapter '{chapter.name}': Extracted {len(recipes)}/{len(titles)} recipes "
            f"({completeness:.0%} complete)"
        )

        return ExtractionResult(
            chapter_name=chapter.name,
            recipes=recipes,
            expected_count=len(titles),
            titles_found=titles,
        )

    async def _extract_paginated(self, chapter: Chapter) -> ExtractionResult:
        """Legacy pagination-based extraction with retry logic.

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

            except (OpenAIError, ValueError) as e:
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
        all_recipes: list[MelaRecipe] = []
        remaining_content = chapter.content
        page = 1
        max_pages = self.config.max_pagination_pages
        last_content_marker: str | None = None

        while page <= max_pages:
            # Build extraction prompt
            is_continuation = page > 1

            prompt = self._build_pagination_prompt(
                remaining_content, is_continuation=is_continuation, last_marker=last_content_marker
            )

            # Save debug input if enabled
            if self.debug and self.debug_dir:
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
            if self.debug and self.debug_dir:
                self._save_debug_output(chapter, page, response)

            # Handle case where parsing failed - stop pagination gracefully
            if response.output_parsed is None:
                logger.warning(
                    f"Chapter '{chapter.name}' page {page}: Got None response, stopping pagination"
                )
                break  # Stop pagination, return recipes collected so far

            batch = response.output_parsed.recipes
            has_more = response.output_parsed.has_more
            last_content_marker = response.output_parsed.last_content_marker

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

            # Find continuation point using content marker or fallback to title
            continuation_pos = -1
            if last_content_marker:
                continuation_pos = self._find_continuation_point(
                    remaining_content, last_content_marker
                )

            # Fallback to title-based search if marker not found
            if continuation_pos == -1 and batch:
                last_recipe_title = batch[-1].title
                continuation_pos = self._find_continuation_point(
                    remaining_content, last_recipe_title, is_title_fallback=True
                )
                if continuation_pos != -1:
                    logger.debug(f"Chapter '{chapter.name}': Used title fallback for continuation")

            if continuation_pos == -1:
                logger.warning(
                    f"Chapter '{chapter.name}': Could not find continuation point, stopping"
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
        self, content: str, is_continuation: bool = False, last_marker: str | None = None
    ) -> str:
        """Build extraction prompt for initial or continuation request."""
        if is_continuation and last_marker:
            return (
                "<instructions>\n"
                "Continue extracting recipes from this cookbook section.\n"
                "Start from where the previous extraction ended.\n"
                "</instructions>\n\n"
                "<previous_marker>\n"
                f"{last_marker}\n"
                "</previous_marker>\n\n"
                "<rules>\n"
                "- Extract up to 10 complete recipes in STRICT sequential order\n"
                "- Process recipes ONE BY ONE from the start - never skip ahead\n"
                "- A recipe has: title, ingredients (any format), and cooking instructions\n"
                "- Ingredients may appear as lists, blockquotes (>), or inline in prose\n"
                "- Instructions may be numbered steps OR prose paragraphs\n"
                "- Set is_standalone_recipe=true for independent recipes\n"
                "- Set is_standalone_recipe=false for components (sauces, doughs, glazes)\n"
                "- Copy titles exactly. Preserve ingredient groupings.\n"
                "- Leave time/yield blank if not stated.\n"
                "- Capture markdown image paths (e.g., '![Name](../images/pg_65.jpg)')\n"
                "- If more recipes remain, set has_more=true and provide "
                "last_content_marker (50-100 chars from the end of your last recipe)\n"
                "</rules>\n\n"
                f"<content>\n{content}\n</content>"
            )
        else:
            return (
                "<instructions>\n"
                "Extract complete recipes from this cookbook section.\n"
                "Determine if each is a standalone recipe or a component/sub-recipe.\n"
                "</instructions>\n\n"
                "<rules>\n"
                "- Extract up to 10 complete recipes in STRICT sequential order\n"
                "- Process recipes ONE BY ONE from the start - never skip ahead\n"
                "- A recipe has: title, ingredients (any format), and cooking instructions\n"
                "- Ingredients may appear as lists, blockquotes (>), or inline in prose\n"
                "- Instructions may be numbered steps OR prose paragraphs\n"
                "- Set is_standalone_recipe=true for independent recipes\n"
                "- Set is_standalone_recipe=false for components (sauces, doughs, glazes)\n"
                "- Variations under a recipe (e.g., 'For Grilled Fish') are NOT separate recipes\n"
                "- Copy titles exactly. Preserve ingredient groupings.\n"
                "- Leave time/yield blank if not stated.\n"
                "- Capture markdown image paths (e.g., '![Name](../images/pg_65.jpg)')\n"
                "- If more recipes remain after your 10, set has_more=true and provide "
                "last_content_marker (50-100 chars from the end of recipe #10)\n"
                "</rules>\n\n"
                f"<content>\n{content}\n</content>"
            )

    def _find_continuation_point(
        self, content: str, marker: str, is_title_fallback: bool = False
    ) -> int:
        """Find where to continue extraction using a content marker or title.

        Args:
            content: The content to search in
            marker: A content marker (from last_content_marker) or recipe title
            is_title_fallback: If True, search for next recipe heading after the marker

        Returns:
            Position in content after the marker, or -1 if not found.
        """
        # Try to find the first occurrence of the marker in the content
        pos = content.find(marker)

        if pos == -1:
            return -1

        # For title fallback, find the NEXT recipe heading after this one
        # to avoid including the last extracted recipe's content
        if is_title_fallback:
            # Search for next recipe heading at any level after the title
            # Try ## first (most common), then ### (some cookbooks use this)
            search_start = pos + len(marker)
            for heading_pattern in ["\n## ", "\n### "]:
                next_heading_pos = content.find(heading_pattern, search_start)
                if next_heading_pos != -1:
                    return next_heading_pos + 1  # Start at the heading
            # No more headings found - content is exhausted
            return -1

        # Move past the marker to avoid re-processing it
        return pos + len(marker)

    # ============================================================================
    # Title-Grounded Extraction (Two-Stage Approach)
    # ============================================================================

    async def enumerate_titles(self, chapter: Chapter) -> list[str]:
        """Stage 1: Enumerate all recipe titles in a chapter.

        This lightweight LLM call returns just the titles, which can then
        be verified against the content before Stage 2 extraction.

        Args:
            chapter: Chapter to scan for recipe titles

        Returns:
            List of verified recipe titles that exist in the content
        """
        prompt = self._build_enumeration_prompt(chapter.content)

        # Save debug input if enabled
        if self.debug and self.debug_dir:
            self._save_debug_input(chapter, page=0, prompt=prompt, content=chapter.content)

        try:
            response = await self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=ChapterTitles,
                reasoning={"effort": "low"},
            )

            if response.output_parsed is None:
                logger.warning(f"Chapter '{chapter.name}': Title enumeration returned None")
                return []

            # Save debug output
            if self.debug and self.debug_dir:
                self._save_enumeration_debug(chapter, response)

            raw_titles = response.output_parsed.titles
            chapter_type = response.output_parsed.chapter_type

            # Skip non-recipe chapters
            if chapter_type in ("intro", "index", "toc", "basics"):
                logger.info(f"Chapter '{chapter.name}': Identified as '{chapter_type}', skipping")
                return []

            # Verify each title exists in content
            verified_titles: list[str] = []
            for title in raw_titles:
                if self._verify_title_in_content(title, chapter.content):
                    verified_titles.append(title)
                else:
                    logger.warning(
                        f"Chapter '{chapter.name}': Title '{title}' not found in content"
                    )

            logger.info(
                f"Chapter '{chapter.name}': Found {len(verified_titles)}/{len(raw_titles)} "
                f"verified titles"
            )
            return verified_titles

        except (OpenAIError, ValueError) as e:
            logger.error(f"Chapter '{chapter.name}': Title enumeration failed: {e}")
            return []

    def _build_enumeration_prompt(self, content: str) -> str:
        """Build prompt for Stage 1 title enumeration."""
        return (
            "<instructions>\n"
            "List all recipe titles in this cookbook chapter.\n"
            "Also identify what type of chapter this is.\n"
            "</instructions>\n\n"
            "<rules>\n"
            "- A recipe has: title, ingredients with measurements, and cooking instructions\n"
            "- Copy each recipe title EXACTLY as written in the text\n"
            "- Do NOT include section headers (e.g., 'VEGETABLES', 'DESSERTS')\n"
            "- Do NOT include chapter titles or introductory headings\n"
            "- Do NOT include component names within recipes (e.g., 'For the sauce')\n"
            "- Do NOT include cross-references to other pages\n"
            "- List titles in order of appearance\n"
            "</rules>\n\n"
            f"<content>\n{content}\n</content>"
        )

    def _verify_title_in_content(self, title: str, content: str) -> bool:
        """Verify that a title exists in the chapter content.

        Performs exact match first, then progressively looser matching:
        1. Exact match
        2. Case-insensitive match
        3. Unicode-normalized match (handles curly quotes, dashes, etc.)
        4. Special character-stripped match

        Args:
            title: Recipe title to find
            content: Chapter content to search in

        Returns:
            True if title found in content (with any matching strategy)
        """
        import re
        import unicodedata

        # Exact match
        if title in content:
            return True

        # Case-insensitive match
        if title.lower() in content.lower():
            return True

        # Try without leading/trailing whitespace
        stripped = title.strip()
        if stripped in content or stripped.lower() in content.lower():
            return True

        # Unicode normalization (handles composed vs decomposed characters)
        def normalize(text: str) -> str:
            # NFC normalize unicode (canonical composition)
            text = unicodedata.normalize("NFC", text)
            # Collapse whitespace
            text = " ".join(text.split())
            return text.lower()

        norm_title = normalize(title)
        norm_content = normalize(content)

        if norm_title in norm_content:
            return True

        # Strip special characters that often vary between source and enumeration
        # (curly quotes, apostrophes, dashes, etc.)
        def clean_special_chars(text: str) -> str:
            # Replace common quote/apostrophe variants with nothing
            text = re.sub(r"[''\"'`]", "", text)
            # Replace various dashes with nothing (intentional Unicode)
            text = re.sub(r"[-–—−]", "", text)  # noqa: RUF001
            # Remove other punctuation that might differ
            text = re.sub(r"[,;:!?.]", "", text)
            return text

        clean_title = clean_special_chars(norm_title)
        clean_content = clean_special_chars(norm_content)

        if clean_title in clean_content:
            logger.debug(f"Title '{title}' matched after cleaning special chars")
            return True

        return False

    def _titles_match(self, requested: str, extracted: str) -> bool:
        """Check if extracted title matches requested title with fuzzy tolerance.

        Allows for minor variations like case differences, punctuation,
        and slight reformulations while catching completely wrong extractions.

        Args:
            requested: The title we asked to extract
            extracted: The title the LLM returned

        Returns:
            True if titles are similar enough (>85% match ratio)
        """
        from difflib import SequenceMatcher

        # Normalize for comparison
        req_norm = requested.lower().strip()
        ext_norm = extracted.lower().strip()

        # Exact match after normalization
        if req_norm == ext_norm:
            return True

        # Fuzzy match with SequenceMatcher
        ratio = SequenceMatcher(None, req_norm, ext_norm).ratio()
        return ratio > self.config.title_match_threshold

    async def extract_by_title(
        self, chapter: Chapter, title: str, title_index: int = 0
    ) -> MelaRecipe | None:
        """Stage 2: Extract a specific recipe by its title with retry logic.

        This grounded extraction is anchored to a verified title,
        ensuring we extract exactly the recipe we're looking for.
        Includes validation to catch wrong-recipe extractions and
        retry logic for transient failures.

        Args:
            chapter: Chapter containing the recipe
            title: Exact recipe title to extract
            title_index: Index of this title (for debug output naming)

        Returns:
            Extracted MelaRecipe or None if extraction failed after retries
        """
        max_retries = self.config.extraction_retry_attempts
        prompt = self._build_grounded_extraction_prompt(title, chapter.content)

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.responses.parse(
                    model=self.model,
                    input=[EasyInputMessageParam(role="user", content=prompt)],
                    text_format=MelaRecipe,
                    reasoning={"effort": "low"},
                )

                # Save debug output (only on final attempt or success)
                if self.debug and self.debug_dir:
                    self._save_grounded_debug(chapter, title, title_index, response)

                if response.output_parsed is None:
                    # Retry on None response
                    if attempt < max_retries:
                        logger.debug(
                            f"Chapter '{chapter.name}': Retrying '{title}' "
                            f"(attempt {attempt + 1}/{max_retries + 1}, got None)"
                        )
                        await asyncio.sleep(self.config.extraction_retry_delay * (attempt + 1))
                        continue
                    logger.warning(f"Chapter '{chapter.name}': Failed to extract '{title}'")
                    return None

                recipe = response.output_parsed

                # Validate: extracted title should match requested title
                if not self._titles_match(title, recipe.title):
                    logger.warning(
                        f"Chapter '{chapter.name}': Title mismatch - "
                        f"requested '{title}', got '{recipe.title}'"
                    )
                    # Still return the recipe but log the discrepancy
                    # The recipe might be correct with a slightly different title format

                logger.debug(f"Chapter '{chapter.name}': Extracted '{title}'")
                return recipe

            except (OpenAIError, ValueError) as e:
                if attempt < max_retries:
                    logger.debug(
                        f"Chapter '{chapter.name}': Retrying '{title}' "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(self.config.extraction_retry_delay * 2 * (attempt + 1))
                    continue
                logger.error(
                    f"Chapter '{chapter.name}': Failed '{title}' "
                    f"after {max_retries + 1} attempts: {e}"
                )
                return None

        return None  # Should not reach here, but for type safety

    def _build_grounded_extraction_prompt(self, title: str, content: str) -> str:
        """Build prompt for Stage 2 grounded extraction."""
        return (
            "<instructions>\n"
            f'Extract the recipe titled "{title}" from this chapter.\n'
            f'Find where "{title}" appears and extract that complete recipe.\n'
            "</instructions>\n\n"
            "<rules>\n"
            "- Extract ONLY this one recipe\n"
            "- Include all ingredients with measurements\n"
            "- Include all cooking instructions\n"
            "- Preserve ingredient groupings if present\n"
            "- Set is_standalone_recipe=false if this is a component/sub-recipe\n"
            "- Leave time/yield blank if not stated\n"
            "- Capture any associated image paths\n"
            "</rules>\n\n"
            f"<content>\n{content}\n</content>"
        )

    def _save_enumeration_debug(self, chapter: Chapter, response: Any) -> None:
        """Save debug output for title enumeration."""
        if not self.debug_dir:
            return

        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = self.debug_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)

        debug_file = chapter_dir / "enumeration.json"
        if response.output_parsed is None:
            data = {"error": "output_parsed is None"}
        else:
            data = {
                "titles": response.output_parsed.titles,
                "chapter_type": response.output_parsed.chapter_type,
                "title_count": len(response.output_parsed.titles),
            }

        debug_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_grounded_debug(self, chapter: Chapter, title: str, index: int, response: Any) -> None:
        """Save debug output for grounded extraction."""
        if not self.debug_dir:
            return

        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = self.debug_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)

        debug_file = chapter_dir / f"recipe_{index:02d}_{self._slugify(title)}.json"
        if response.output_parsed is None:
            data = {"title": title, "error": "extraction_failed"}
        else:
            data = {
                "title": title,
                "recipe": response.output_parsed.model_dump(),
            }

        debug_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _slugify(self, text: str) -> str:
        """Convert text to a safe filename slug."""
        import re

        slug = text.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        return slug[:50].strip("_")

    def _save_debug_input(self, chapter: Chapter, page: int, prompt: str, content: str) -> None:
        """Save debug information about the input sent to OpenAI."""
        if not self.debug_dir:
            return

        # Create chapter directory
        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = self.debug_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)

        # Save the prompt
        prompt_file = chapter_dir / f"page_{page:02d}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        # Save the markdown content being processed
        content_file = chapter_dir / f"page_{page:02d}_content.md"
        content_file.write_text(content, encoding="utf-8")

    def _save_debug_output(self, chapter: Chapter, page: int, response: Any) -> None:
        """Save debug information about the response from OpenAI."""
        if not self.debug_dir:
            return

        # Create chapter directory
        chapter_slug = chapter.name.replace("/", "_").replace(".", "_")
        chapter_dir = self.debug_dir / chapter_slug
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
