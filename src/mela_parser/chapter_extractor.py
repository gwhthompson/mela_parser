#!/usr/bin/env python3
"""
Production-ready async chapter-based EPUB recipe extraction framework.

This module provides a clean, efficient approach to extracting recipes from EPUB cookbooks
by leveraging the natural chapter structure of EPUB files. It eliminates duplicates through
chapter-based processing and provides validation against discovered recipe lists.

Key Features:
- Async parallel processing for optimal performance
- Chapter-based extraction using ebooklib's ITEM_DOCUMENT
- Recipe list discovery and validation
- Comprehensive error handling with custom exceptions
- Structured logging throughout
- Type-safe with full type hints
- Pydantic V2 for data validation

Example Usage:
    >>> import asyncio
    >>> from chapter_extractor import ChapterProcessor, AsyncChapterExtractor
    >>>
    >>> async def process_cookbook(epub_path: str):
    ...     processor = ChapterProcessor(epub_path)
    ...     chapters = await processor.split_into_chapters()
    ...
    ...     extractor = AsyncChapterExtractor()
    ...     results = await extractor.extract_from_chapters(chapters)
    ...     return results
    >>>
    >>> asyncio.run(process_cookbook("cookbook.epub"))
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Final, Optional

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, Field

from .parse import CookbookRecipes, MelaRecipe

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class ChapterExtractionError(Exception):
    """Base exception for chapter extraction errors."""
    pass


class EPUBConversionError(ChapterExtractionError):
    """Raised when EPUB to markdown conversion fails."""
    pass


class RecipeListDiscoveryError(ChapterExtractionError):
    """Raised when recipe list discovery encounters an error."""
    pass


class RecipeExtractionError(ChapterExtractionError):
    """Raised when recipe extraction from a chapter fails."""
    pass


# ============================================================================
# Data Models
# ============================================================================


class RecipeList(BaseModel):
    """Schema for discovered recipe titles from recipe lists."""

    titles: list[str] = Field(
        description="List of unique recipe titles found in the cookbook's table of contents or indices"
    )

    class Config:
        extra = "forbid"


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
    error: Optional[str] = None
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return self.error is None

    @property
    def recipe_count(self) -> int:
        """Get number of recipes extracted."""
        return len(self.recipes)


@dataclass
class ValidationDiff:
    """
    Detailed diff between extracted recipes and discovered recipe list.

    Attributes:
        expected_titles: Original list of recipe titles from discovery
        extracted_titles: Titles of recipes actually extracted
        exact_matches: Titles that match exactly
        missing_titles: Expected titles not found in extraction
        extra_titles: Extracted titles not in expected list
        match_rate: Percentage of expected recipes found (0.0 to 1.0)
    """

    expected_titles: set[str]
    extracted_titles: set[str]
    exact_matches: set[str] = field(init=False)
    missing_titles: set[str] = field(init=False)
    extra_titles: set[str] = field(init=False)
    match_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate diff metrics."""
        self.exact_matches = self.expected_titles & self.extracted_titles
        self.missing_titles = self.expected_titles - self.extracted_titles
        self.extra_titles = self.extracted_titles - self.expected_titles

        if self.expected_titles:
            self.match_rate = len(self.exact_matches) / len(self.expected_titles)
        else:
            self.match_rate = 0.0

    @property
    def is_perfect_match(self) -> bool:
        """Check if extraction matched all expected recipes."""
        return self.match_rate == 1.0 and not self.extra_titles


# ============================================================================
# ChapterProcessor: EPUB to Chapters
# ============================================================================


class ChapterProcessor:
    """
    Converts EPUB files into individual chapters using ebooklib.

    This class handles the conversion of EPUB documents into structured chapters,
    with each chapter converted to clean markdown using MarkItDown. It provides
    both synchronous and asynchronous processing methods.

    Attributes:
        epub_path: Path to the EPUB file
        book: Loaded EpubBook instance
        book_title: Title extracted from EPUB metadata

    Example:
        >>> processor = ChapterProcessor("cookbook.epub")
        >>> chapters = await processor.split_into_chapters()
        >>> for chapter in chapters:
        ...     print(f"{chapter.name}: {len(chapter.content)} chars")
    """

    def __init__(self, epub_path: str | Path) -> None:
        """
        Initialize the chapter processor.

        Args:
            epub_path: Path to the EPUB file to process

        Raises:
            FileNotFoundError: If EPUB file does not exist
            EPUBConversionError: If EPUB cannot be loaded
        """
        self.epub_path = Path(epub_path)

        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {self.epub_path}")

        try:
            self.book = epub.read_epub(str(self.epub_path), {"ignore_ncx": True})
            self._md = MarkItDown()

            # Extract book title from metadata
            title_metadata = self.book.get_metadata("DC", "title")
            self.book_title = title_metadata[0][0] if title_metadata else self.epub_path.stem

            logger.info(f"Loaded EPUB: {self.book_title}")

        except Exception as e:
            raise EPUBConversionError(f"Failed to load EPUB: {e}") from e

    async def split_into_chapters(self) -> list[Chapter]:
        """
        Split EPUB into chapters and convert each to markdown asynchronously.

        This method extracts all ITEM_DOCUMENT items from the EPUB, converts
        each to markdown using MarkItDown, and returns structured Chapter objects.

        Returns:
            List of Chapter objects with name and markdown content

        Raises:
            EPUBConversionError: If chapter conversion fails
        """
        logger.info("Starting chapter extraction and conversion")

        # Get all document items from EPUB
        items = list(self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        logger.info(f"Found {len(items)} chapters in EPUB")

        if not items:
            logger.warning("No chapters found in EPUB")
            return []

        # Convert chapters in parallel using asyncio
        tasks = [
            self._convert_chapter_async(item, idx)
            for idx, item in enumerate(items)
        ]

        try:
            chapters = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            valid_chapters = []
            for chapter in chapters:
                if isinstance(chapter, Exception):
                    logger.error(f"Chapter conversion failed: {chapter}")
                elif chapter is not None:
                    valid_chapters.append(chapter)

            logger.info(f"Successfully converted {len(valid_chapters)}/{len(items)} chapters")
            return valid_chapters

        except Exception as e:
            raise EPUBConversionError(f"Chapter conversion failed: {e}") from e

    async def _convert_chapter_async(
        self,
        item: epub.EpubItem,
        index: int
    ) -> Optional[Chapter]:
        """
        Convert a single EPUB item to markdown asynchronously.

        Args:
            item: EPUB item to convert
            index: Chapter index for ordering

        Returns:
            Chapter object or None if conversion fails
        """
        try:
            # Run blocking I/O in executor
            loop = asyncio.get_event_loop()
            html_content = await loop.run_in_executor(None, item.get_content)

            # Convert to markdown using MarkItDown
            result = await loop.run_in_executor(
                None,
                self._md.convert_stream,
                BytesIO(html_content),
                ".html"
            )

            markdown_content = result.text_content
            chapter_name = item.get_name()

            if not markdown_content or len(markdown_content.strip()) < 50:
                logger.warning(f"Chapter '{chapter_name}' has minimal content ({len(markdown_content)} chars)")
                return None

            logger.debug(f"Converted chapter {index}: {chapter_name} ({len(markdown_content)} chars)")

            return Chapter(
                name=chapter_name,
                content=markdown_content,
                index=index
            )

        except Exception as e:
            logger.error(f"Failed to convert chapter {index} ({item.get_name()}): {e}")
            return None


# ============================================================================
# RecipeListDiscoverer: Find Recipe Lists
# ============================================================================


class RecipeListDiscoverer:
    """
    Discovers recipe lists from cookbook chapters using pattern matching and LLM cleaning.

    This class scans all chapters for sections containing multiple recipe links
    (typically table of contents or indices), then uses an LLM to extract and clean
    the unique list of recipe titles.

    Attributes:
        client: OpenAI client for LLM interactions
        model: Model to use for recipe list cleaning (default: gpt-5-mini)
        min_links: Minimum links required to consider a section a recipe list

    Example:
        >>> discoverer = RecipeListDiscoverer()
        >>> titles = await discoverer.discover_from_chapters(chapters)
        >>> if titles:
        ...     print(f"Found {len(titles)} recipes in cookbook index")
    """

    # Constants
    DEFAULT_MODEL: Final[str] = "gpt-5-mini"
    MIN_LINKS_THRESHOLD: Final[int] = 5
    LINK_PATTERN: Final[re.Pattern] = re.compile(r'\[([^\]]+)\]\([^)]+\)')

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        min_links: int = MIN_LINKS_THRESHOLD
    ) -> None:
        """
        Initialize the recipe list discoverer.

        Args:
            model: OpenAI model to use for cleaning recipe lists
            min_links: Minimum number of links to consider section a recipe list
        """
        self.client = OpenAI()
        self.model = model
        self.min_links = min_links

        logger.info(f"Initialized RecipeListDiscoverer with model: {model}")

    async def discover_from_chapters(
        self,
        chapters: list[Chapter]
    ) -> Optional[list[str]]:
        """
        Scan all chapters for recipe lists and extract unique recipe titles.

        This method:
        1. Scans chapters for sections with many markdown links (potential TOC/index)
        2. Combines all link sections
        3. Uses LLM to extract and clean unique recipe titles

        Args:
            chapters: List of Chapter objects to scan

        Returns:
            List of unique recipe titles, or None if no recipe list found

        Raises:
            RecipeListDiscoveryError: If discovery process encounters an error
        """
        logger.info(f"Scanning {len(chapters)} chapters for recipe lists")

        try:
            # Collect all potential recipe list sections
            link_sections = await self._find_link_sections(chapters)

            if not link_sections:
                logger.info("No recipe list sections found (no sections with 5+ links)")
                return None

            logger.info(f"Found {len(link_sections)} potential recipe list sections")

            # Combine and clean with LLM
            titles = await self._extract_clean_titles(link_sections)

            if titles:
                logger.info(f"Discovered {len(titles)} unique recipe titles")
            else:
                logger.warning("Recipe list discovery returned no titles")

            return titles

        except Exception as e:
            raise RecipeListDiscoveryError(f"Failed to discover recipe list: {e}") from e

    async def _find_link_sections(self, chapters: list[Chapter]) -> list[str]:
        """
        Find sections with many markdown links (potential recipe lists).

        Args:
            chapters: Chapters to scan

        Returns:
            List of text sections containing recipe links
        """
        loop = asyncio.get_event_loop()

        async def scan_chapter(chapter: Chapter) -> list[str]:
            """Scan a single chapter for link sections."""
            # Run regex search in executor to avoid blocking
            links = await loop.run_in_executor(
                None,
                self.LINK_PATTERN.findall,
                chapter.content
            )

            if len(links) >= self.min_links:
                logger.debug(f"Chapter '{chapter.name}' has {len(links)} links - potential recipe list")
                return ["\n".join(links)]

            return []

        # Scan all chapters in parallel
        results = await asyncio.gather(*[scan_chapter(ch) for ch in chapters])

        # Flatten results
        return [section for sections in results for section in sections]

    async def _extract_clean_titles(self, link_sections: list[str]) -> Optional[list[str]]:
        """
        Use LLM to extract and clean unique recipe titles from link sections.

        Args:
            link_sections: Sections containing recipe links

        Returns:
            Cleaned list of unique recipe titles
        """
        combined = "\n\n".join(link_sections)

        prompt = f"""Extract the unique list of recipe titles from these potential recipe lists.

Remove:
- Section headers (Contents, Index, About, Introduction, etc.)
- Page numbers
- Duplicates
- Navigation elements

Keep:
- Actual recipe titles EXACTLY as written
- One entry per unique recipe
- Only complete recipe names (not partial references)

<potential_lists>
{combined}
</potential_lists>"""

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.responses.parse(
                    model=self.model,
                    input=[EasyInputMessageParam(role="user", content=prompt)],
                    text_format=RecipeList,
                )
            )

            titles = response.output_parsed.titles

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.debug(
                    f"Recipe list extraction - "
                    f"Input: {getattr(usage, 'input_tokens', 0)} tokens, "
                    f"Output: {getattr(usage, 'output_tokens', 0)} tokens"
                )

            return titles

        except Exception as e:
            logger.error(f"Failed to clean recipe list with LLM: {e}")
            return None


# ============================================================================
# AsyncChapterExtractor: Extract Recipes from Chapters
# ============================================================================


class AsyncChapterExtractor:
    """
    Asynchronously extracts recipes from individual chapters with retry logic.

    This class provides parallel recipe extraction from multiple chapters using
    async processing. It supports targeted extraction based on expected recipe
    titles and includes comprehensive retry logic with exponential backoff.

    Attributes:
        client: Async OpenAI client for parallel LLM interactions
        model: Model to use for extraction (default: gpt-5-nano)
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay for exponential backoff (seconds)

    Example:
        >>> extractor = AsyncChapterExtractor(model="gpt-5-nano")
        >>> results = await extractor.extract_from_chapters(
        ...     chapters,
        ...     expected_titles=["Chocolate Cake", "Apple Pie"]
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
        initial_retry_delay: float = INITIAL_RETRY_DELAY
    ) -> None:
        """
        Initialize the async chapter extractor.

        Args:
            model: OpenAI model to use for extraction
            max_retries: Maximum retry attempts for failed extractions
            initial_retry_delay: Initial delay in seconds for exponential backoff
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        logger.info(f"Initialized AsyncChapterExtractor with model: {model}")

    async def extract_from_chapters(
        self,
        chapters: list[Chapter],
        expected_titles: Optional[list[str]] = None,
        max_concurrent: int = 5
    ) -> list[ExtractionResult]:
        """
        Extract recipes from multiple chapters in parallel.

        Args:
            chapters: List of chapters to process
            expected_titles: Optional list of expected recipe titles for targeted extraction
            max_concurrent: Maximum number of concurrent extractions

        Returns:
            List of ExtractionResult objects for each chapter
        """
        logger.info(f"Starting extraction from {len(chapters)} chapters (max concurrent: {max_concurrent})")

        if expected_titles:
            logger.info(f"Using targeted extraction with {len(expected_titles)} expected titles")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(chapter: Chapter) -> ExtractionResult:
            """Extract with concurrency control."""
            async with semaphore:
                return await self.extract_from_chapter(chapter, expected_titles)

        # Process all chapters in parallel (with concurrency limit)
        results = await asyncio.gather(
            *[extract_with_semaphore(ch) for ch in chapters],
            return_exceptions=True
        )

        # Handle exceptions in results
        final_results = []
        for chapter, result in zip(chapters, results):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for chapter '{chapter.name}': {result}")
                final_results.append(ExtractionResult(
                    chapter_name=chapter.name,
                    recipes=[],
                    error=str(result)
                ))
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

    async def extract_from_chapter(
        self,
        chapter: Chapter,
        expected_titles: Optional[list[str]] = None
    ) -> ExtractionResult:
        """
        Extract recipes from a single chapter with retry logic.

        Args:
            chapter: Chapter to extract from
            expected_titles: Optional list of expected recipe titles

        Returns:
            ExtractionResult with extracted recipes or error information
        """
        retry_count = 0
        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                recipes = await self._extract_with_llm(chapter, expected_titles)

                logger.debug(
                    f"Chapter '{chapter.name}': Extracted {len(recipes)} recipes "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                return ExtractionResult(
                    chapter_name=chapter.name,
                    recipes=recipes,
                    retry_count=retry_count
                )

            except Exception as e:
                retry_count += 1
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Chapter '{chapter.name}' extraction failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Chapter '{chapter.name}' extraction failed after {self.max_retries} attempts: {e}"
                    )

        return ExtractionResult(
            chapter_name=chapter.name,
            recipes=[],
            error=last_error,
            retry_count=retry_count
        )

    async def _extract_with_llm(
        self,
        chapter: Chapter,
        expected_titles: Optional[list[str]] = None
    ) -> list[MelaRecipe]:
        """
        Use LLM to extract recipes from chapter content.

        Args:
            chapter: Chapter to process
            expected_titles: Optional expected recipe titles for targeted extraction

        Returns:
            List of extracted MelaRecipe objects
        """
        # Build extraction prompt
        prompt = self._build_extraction_prompt(chapter, expected_titles)

        # Call LLM API
        response = await self.client.responses.parse(
            model=self.model,
            input=[EasyInputMessageParam(role="user", content=prompt)],
            text_format=CookbookRecipes,
        )

        recipes = response.output_parsed.recipes

        # Log token usage if available
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            logger.debug(
                f"Chapter '{chapter.name}' - "
                f"Input: {getattr(usage, 'input_tokens', 0)} tokens, "
                f"Output: {getattr(usage, 'output_tokens', 0)} tokens, "
                f"Recipes: {len(recipes)}"
            )

        return recipes

    def _build_extraction_prompt(
        self,
        chapter: Chapter,
        expected_titles: Optional[list[str]] = None
    ) -> str:
        """
        Build extraction prompt for LLM based on context.

        Args:
            chapter: Chapter to extract from
            expected_titles: Optional expected recipe titles

        Returns:
            Formatted prompt string
        """
        if expected_titles:
            # Targeted extraction: find which recipes might be in this chapter
            likely_here = [
                title for title in expected_titles
                if title.lower() in chapter.content.lower()
            ]

            if not likely_here:
                # Return prompt that explicitly says no recipes expected
                return f"""This chapter likely contains no complete recipes.
Please scan carefully, but if you find no complete recipes (title + ingredients + instructions), return an empty list.

<chapter>
{chapter.content}
</chapter>"""

            expected_list = "\n".join(f"- {title}" for title in likely_here)

            return f"""Extract ONLY these specific recipes from this chapter.
Use the EXACT titles listed below.

A COMPLETE recipe MUST have:
1. A title (use EXACT title from expected list)
2. Ingredients with measurements
3. Instructions/steps

Expected recipes in this chapter:
{expected_list}

If a recipe is incomplete or you cannot find it, skip it.

<chapter>
{chapter.content}
</chapter>"""

        else:
            # General extraction - schema enforces completeness via min_length constraints
            return f"""Extract all complete recipes from this section.

The schema requires:
- At least 1 ingredient with measurements
- At least 2 instruction steps

This naturally filters out incomplete content. Extract anything that meets these requirements.

Copy titles exactly. Preserve ingredient groupings. Leave time/yield blank if not stated.

<section>
{chapter.content}
</section>"""


# ============================================================================
# ValidationEngine: Compare Expected vs Extracted
# ============================================================================


class ValidationEngine:
    """
    Validates extraction results against discovered recipe lists.

    This class provides comprehensive validation and reporting to compare
    extracted recipes against expected recipe titles discovered from the
    cookbook's table of contents or indices.

    Example:
        >>> engine = ValidationEngine()
        >>> diff = engine.create_diff(
        ...     expected_titles=["Cake", "Pie", "Cookies"],
        ...     extracted_recipes=extracted_recipes
        ... )
        >>> print(engine.generate_report(diff))
    """

    @staticmethod
    def create_diff(
        expected_titles: list[str],
        extracted_recipes: list[MelaRecipe]
    ) -> ValidationDiff:
        """
        Create detailed diff between expected and extracted recipes.

        Args:
            expected_titles: List of recipe titles from discovery
            extracted_recipes: List of extracted MelaRecipe objects

        Returns:
            ValidationDiff object with detailed comparison
        """
        expected_set = set(expected_titles)
        extracted_set = {recipe.title for recipe in extracted_recipes}

        diff = ValidationDiff(
            expected_titles=expected_set,
            extracted_titles=extracted_set
        )

        logger.info(
            f"Validation - Match rate: {diff.match_rate:.1%} "
            f"({len(diff.exact_matches)}/{len(expected_set)} recipes)"
        )

        if diff.missing_titles:
            logger.warning(f"Missing {len(diff.missing_titles)} expected recipes")

        if diff.extra_titles:
            logger.warning(f"Found {len(diff.extra_titles)} unexpected recipes")

        return diff

    @staticmethod
    def generate_report(diff: ValidationDiff, max_items: int = 20) -> str:
        """
        Generate human-readable validation report.

        Args:
            diff: ValidationDiff object to report on
            max_items: Maximum number of items to show in each section

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "VALIDATION REPORT",
            "=" * 80,
            "",
            f"Expected recipes: {len(diff.expected_titles)}",
            f"Extracted recipes: {len(diff.extracted_titles)}",
            f"Exact matches: {len(diff.exact_matches)}",
            f"Match rate: {diff.match_rate:.1%}",
            ""
        ]

        # Exact matches
        if diff.exact_matches:
            lines.append(f"✓ EXACT MATCHES ({len(diff.exact_matches)}):")
            for title in sorted(diff.exact_matches)[:max_items]:
                lines.append(f"  ✓ {title}")
            if len(diff.exact_matches) > max_items:
                lines.append(f"  ... and {len(diff.exact_matches) - max_items} more")
            lines.append("")

        # Missing titles
        if diff.missing_titles:
            lines.append(f"✗ MISSING RECIPES ({len(diff.missing_titles)}):")
            for title in sorted(diff.missing_titles)[:max_items]:
                lines.append(f"  ✗ {title}")
            if len(diff.missing_titles) > max_items:
                lines.append(f"  ... and {len(diff.missing_titles) - max_items} more")
            lines.append("")

        # Extra titles
        if diff.extra_titles:
            lines.append(f"+ UNEXPECTED RECIPES ({len(diff.extra_titles)}):")
            for title in sorted(diff.extra_titles)[:max_items]:
                lines.append(f"  + {title}")
            if len(diff.extra_titles) > max_items:
                lines.append(f"  ... and {len(diff.extra_titles) - max_items} more")
            lines.append("")

        # Perfect match indicator
        if diff.is_perfect_match:
            lines.append("🎯 PERFECT MATCH - All recipes found, no extras!")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def validate_extraction_quality(
        results: list[ExtractionResult],
        min_success_rate: float = 0.8
    ) -> tuple[bool, str]:
        """
        Validate overall extraction quality across all chapters.

        Args:
            results: List of ExtractionResult objects
            min_success_rate: Minimum required success rate (0.0 to 1.0)

        Returns:
            Tuple of (is_valid, message)
        """
        total_chapters = len(results)
        successful = sum(1 for r in results if r.is_success)
        success_rate = successful / total_chapters if total_chapters > 0 else 0.0

        total_recipes = sum(r.recipe_count for r in results)
        total_retries = sum(r.retry_count for r in results)

        if success_rate < min_success_rate:
            message = (
                f"Extraction quality below threshold: {success_rate:.1%} < {min_success_rate:.1%} "
                f"({successful}/{total_chapters} chapters successful)"
            )
            logger.error(message)
            return False, message

        message = (
            f"Extraction quality acceptable: {success_rate:.1%} success rate, "
            f"{total_recipes} recipes extracted, {total_retries} retries"
        )
        logger.info(message)
        return True, message


# ============================================================================
# Convenience Functions
# ============================================================================


async def process_epub_chapters(
    epub_path: str | Path,
    model: str = "gpt-5-nano",
    use_recipe_list: bool = True,
    max_concurrent: int = 5
) -> tuple[list[MelaRecipe], Optional[ValidationDiff]]:
    """
    High-level convenience function to process an EPUB cookbook.

    This function orchestrates the complete extraction pipeline:
    1. Split EPUB into chapters
    2. Discover recipe list (if enabled)
    3. Extract recipes from all chapters in parallel
    4. Validate results

    Args:
        epub_path: Path to EPUB file
        model: Model to use for extraction (default: gpt-5-nano)
        use_recipe_list: Whether to discover and use recipe list (default: True)
        max_concurrent: Maximum concurrent extractions

    Returns:
        Tuple of (all_recipes, validation_diff)

    Example:
        >>> recipes, diff = await process_epub_chapters("cookbook.epub")
        >>> print(f"Extracted {len(recipes)} recipes")
        >>> if diff:
        ...     print(f"Match rate: {diff.match_rate:.1%}")
    """
    # Phase 1: Split into chapters
    logger.info(f"Processing EPUB: {epub_path}")
    processor = ChapterProcessor(epub_path)
    chapters = await processor.split_into_chapters()

    if not chapters:
        logger.warning("No chapters found in EPUB")
        return [], None

    # Phase 2: Discover recipe list
    expected_titles: Optional[list[str]] = None
    if use_recipe_list:
        discoverer = RecipeListDiscoverer()
        expected_titles = await discoverer.discover_from_chapters(chapters)

    # Phase 3: Extract recipes
    extractor = AsyncChapterExtractor(model=model)
    results = await extractor.extract_from_chapters(
        chapters,
        expected_titles=expected_titles,
        max_concurrent=max_concurrent
    )

    # Collect all recipes and deduplicate
    all_recipes = []
    seen_titles = set()

    for result in results:
        for recipe in result.recipes:
            if recipe.title not in seen_titles:
                seen_titles.add(recipe.title)
                all_recipes.append(recipe)

    logger.info(f"Total unique recipes: {len(all_recipes)}")

    # Phase 4: Validate
    validation_diff: Optional[ValidationDiff] = None
    if expected_titles:
        validator = ValidationEngine()
        validation_diff = validator.create_diff(expected_titles, all_recipes)

        # Log validation report
        report = validator.generate_report(validation_diff)
        for line in report.split("\n"):
            logger.info(line)

    return all_recipes, validation_diff


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core classes
    "ChapterProcessor",
    "RecipeListDiscoverer",
    "AsyncChapterExtractor",
    "ValidationEngine",

    # Data models
    "Chapter",
    "ExtractionResult",
    "ValidationDiff",
    "RecipeList",

    # Exceptions
    "ChapterExtractionError",
    "EPUBConversionError",
    "RecipeListDiscoveryError",
    "RecipeExtractionError",

    # Convenience
    "process_epub_chapters",
]
