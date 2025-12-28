#!/usr/bin/env python3
"""Extract recipe titles from Table of Contents and Index chapters.

This module provides ground truth extraction from TOC and Index sections
of cookbooks, which can be used to verify extraction completeness.
"""

import logging
from typing import Final

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, ConfigDict, Field

from .chapter_extractor import Chapter
from .parse import CookbookTOC

# Configure module logger
logger = logging.getLogger(__name__)


class TOCExtractor:
    """Extract recipe titles from Table of Contents and Index chapters.

    This class identifies TOC and Index chapters in an EPUB and extracts
    recipe titles as ground truth for verification.

    Example:
        >>> extractor = TOCExtractor()
        >>> toc = await extractor.extract_toc(chapters)
        >>> expected_titles = toc.all_recipe_titles()
        >>> # Compare against extracted recipes
    """

    DEFAULT_MODEL: Final[str] = "gpt-5-nano"

    # Patterns to identify TOC/Index chapters
    TOC_PATTERNS: Final[tuple[str, ...]] = (
        "table of contents",
        "contents",
        "toc",
    )
    INDEX_PATTERNS: Final[tuple[str, ...]] = (
        "index",
        "recipe index",
        "alphabetical index",
    )

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the TOC extractor.

        Args:
            client: Async OpenAI client (created if not provided)
            model: OpenAI model to use for extraction
        """
        self.client = client if client is not None else AsyncOpenAI()
        self.model = model
        logger.info(f"Initialized TOCExtractor with model: {model}")

    async def extract_toc(self, chapters: list[Chapter]) -> CookbookTOC:
        """Find and parse Table of Contents chapter.

        Args:
            chapters: All chapters from the EPUB

        Returns:
            CookbookTOC with chapter-recipe mappings, or empty if no TOC found
        """
        toc_chapter = self._find_toc_chapter(chapters)

        if toc_chapter is None:
            logger.info("No Table of Contents chapter found")
            return CookbookTOC(chapters=[])

        logger.info(f"Found TOC chapter: '{toc_chapter.name}'")
        return await self._parse_toc(toc_chapter)

    async def extract_index_titles(self, chapters: list[Chapter]) -> list[str]:
        """Find and parse Index chapter for recipe titles.

        Args:
            chapters: All chapters from the EPUB

        Returns:
            List of recipe titles from index, or empty if no index found
        """
        index_chapter = self._find_index_chapter(chapters)

        if index_chapter is None:
            logger.info("No Index chapter found")
            return []

        logger.info(f"Found Index chapter: '{index_chapter.name}'")
        return await self._parse_index(index_chapter)

    def _find_toc_chapter(self, chapters: list[Chapter]) -> Chapter | None:
        """Find the Table of Contents chapter by name pattern."""
        for chapter in chapters:
            name_lower = chapter.name.lower()
            for pattern in self.TOC_PATTERNS:
                if pattern in name_lower:
                    return chapter
        return None

    def _find_index_chapter(self, chapters: list[Chapter]) -> Chapter | None:
        """Find the Index chapter by name pattern."""
        # Check from the end since index is usually at the back
        for chapter in reversed(chapters):
            name_lower = chapter.name.lower()
            for pattern in self.INDEX_PATTERNS:
                if pattern in name_lower:
                    return chapter
        return None

    async def _parse_toc(self, chapter: Chapter) -> CookbookTOC:
        """Parse TOC chapter into structured format.

        Args:
            chapter: The TOC chapter to parse

        Returns:
            CookbookTOC with chapter-recipe mappings
        """
        prompt = self._build_toc_prompt(chapter.content)

        try:
            response = await self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=CookbookTOC,
                reasoning={"effort": "low"},
            )

            if response.output_parsed is None:
                logger.warning("TOC parsing returned None")
                return CookbookTOC(chapters=[])

            toc = response.output_parsed
            total_recipes = len(toc.all_recipe_titles())
            logger.info(f"Parsed TOC: {len(toc.chapters)} chapters, {total_recipes} recipe titles")
            return toc

        except (OpenAIError, ValueError) as e:
            logger.error(f"TOC parsing failed: {e}")
            return CookbookTOC(chapters=[])

    async def _parse_index(self, chapter: Chapter) -> list[str]:
        """Parse Index chapter to extract recipe titles.

        Args:
            chapter: The Index chapter to parse

        Returns:
            List of recipe titles found in the index
        """
        prompt = self._build_index_prompt(chapter.content)

        try:
            response = await self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=IndexRecipes,
                reasoning={"effort": "low"},
            )

            if response.output_parsed is None:
                logger.warning("Index parsing returned None")
                return []

            titles = response.output_parsed.recipe_titles
            logger.info(f"Parsed Index: {len(titles)} recipe titles")
            return titles

        except (OpenAIError, ValueError) as e:
            logger.error(f"Index parsing failed: {e}")
            return []

    def _build_toc_prompt(self, content: str) -> str:
        """Build prompt for TOC parsing."""
        return (
            "<instructions>\n"
            "Parse this Table of Contents and extract all recipe titles organized by chapter.\n"
            "</instructions>\n\n"
            "<rules>\n"
            "- Identify chapter/section headings\n"
            "- List recipe titles under each chapter\n"
            "- Do NOT include page numbers\n"
            "- Do NOT include sub-section headers that aren't recipes\n"
            "- A recipe title is a specific dish name, not a category\n"
            "</rules>\n\n"
            f"<content>\n{content}\n</content>"
        )

    def _build_index_prompt(self, content: str) -> str:
        """Build prompt for Index parsing."""
        return (
            "<instructions>\n"
            "Extract all recipe titles from this cookbook index.\n"
            "</instructions>\n\n"
            "<rules>\n"
            "- Only include actual recipe names (specific dishes)\n"
            "- Do NOT include ingredients, techniques, or general terms\n"
            "- Do NOT include page numbers\n"
            "- Recipe titles are typically capitalized or in bold\n"
            "</rules>\n\n"
            f"<content>\n{content}\n</content>"
        )


# Schema for index parsing


class IndexRecipes(BaseModel):
    """Schema for extracting recipe titles from an index."""

    recipe_titles: list[str] = Field(
        default_factory=list,
        description=(
            "All recipe titles found in the index. "
            "Only include specific dish names, not ingredients or techniques."
        ),
    )

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "IndexRecipes",
    "TOCExtractor",
]
