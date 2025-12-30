#!/usr/bin/env python3
"""EPUB to Markdown converter using MarkItDown.

This module provides conversion utilities for transforming EPUB cookbooks into
clean, LLM-friendly markdown format. It handles:
- Full EPUB to markdown conversion
- Token estimation and chunking strategies
- Front matter removal (foreword, TOC, etc.)
- Smart heading-based chunking for large files

The EpubConverter class is designed to work with modern LLMs that have large
context windows (e.g., 256K tokens) but may need chunking for extremely large
cookbooks.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown

if TYPE_CHECKING:
    from .chapter_extractor import Chapter

logger = logging.getLogger(__name__)


class ChapterType(Enum):
    """Classification of EPUB chapters by content type."""

    IMAGE_ONLY = "image_only"  # Has LARGE image, <150 chars text
    TEXT_ONLY = "text_only"  # No large image, >150 chars text
    BOTH = "both"  # Has LARGE image AND >150 chars text
    MINIMAL = "minimal"  # <150 chars, no large image


# Icon patterns to filter out (dietary indicators, navigation, ratings)
ICON_PATTERNS = {"gf", "df", "ve", "vg", "logo", "brand", "star", "arrow", "num", "pub"}

# Minimum text length to consider a chapter as having meaningful content
MIN_TEXT_LENGTH = 150


def is_recipe_image(image_path: str) -> bool:
    """Check if image is likely a recipe photo (not an icon).

    Filters out common icon patterns like dietary indicators (gf, df, ve),
    chapter numbers, and navigation elements.

    Args:
        image_path: Path to the image file

    Returns:
        True if likely a recipe photo, False if likely an icon
    """
    filename = image_path.split("/")[-1].lower()
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Skip known icon patterns
    for pattern in ICON_PATTERNS:
        if pattern in base:
            return False

    # Skip tiny images by name pattern (c1-c9, 1by8, etc.)
    return not re.match(r"^[a-z]?\d{1,2}$", base)


def classify_chapter(content: str) -> ChapterType:
    """Classify chapter by recipe image presence and text content.

    Args:
        content: Markdown content of the chapter

    Returns:
        ChapterType indicating the chapter's classification
    """
    # Find all images in markdown format
    images = re.findall(r"!\[.*?\]\(([^)]+)\)", content)

    # Filter to recipe images only (exclude icons)
    recipe_images = [img for img in images if is_recipe_image(img)]
    has_recipe_image = len(recipe_images) > 0

    # Get text length excluding image markup
    text_only = re.sub(r"!\[.*?\]\(.*?\)", "", content).strip()
    text_len = len(text_only)

    if has_recipe_image and text_len > MIN_TEXT_LENGTH:
        return ChapterType.BOTH
    elif has_recipe_image:
        return ChapterType.IMAGE_ONLY
    elif text_len > MIN_TEXT_LENGTH:
        return ChapterType.TEXT_ONLY
    return ChapterType.MINIMAL


def merge_image_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Merge image-only chapters with adjacent text chapters.

    Many cookbook EPUBs have images and recipe text in separate chapters.
    This function merges them so the LLM can see both together.

    Strategy:
    1. Classify each chapter: IMAGE_ONLY, TEXT_ONLY, BOTH, MINIMAL
    2. For each IMAGE_ONLY chapter:
       - If NEXT is TEXT_ONLY: prepend image to next chapter (I→T pattern)
       - Elif PREV is TEXT_ONLY: append image to prev chapter (T→I pattern)
       - Elif NEXT is IMAGE_ONLY: collect consecutive images, merge with first TEXT
    3. Return consolidated chapters

    Args:
        chapters: List of Chapter objects from EPUB conversion

    Returns:
        List of chapters with image-only chapters merged into adjacent text
    """
    from .chapter_extractor import Chapter

    if not chapters:
        return chapters

    # Classify all chapters
    types = [classify_chapter(ch.content) for ch in chapters]

    # Track which chapters have been merged
    merged: list[Chapter] = []
    used: set[int] = set()

    for i, (ch, typ) in enumerate(zip(chapters, types, strict=True)):
        if i in used:
            continue

        if typ == ChapterType.IMAGE_ONLY:
            # Collect consecutive IMAGE_ONLY chapters
            images_content = [ch.content]
            j = i + 1
            while j < len(chapters) and types[j] == ChapterType.IMAGE_ONLY:
                images_content.append(chapters[j].content)
                used.add(j)
                j += 1

            # Find target TEXT_ONLY chapter to merge into
            if j < len(chapters) and types[j] == ChapterType.TEXT_ONLY:
                # I→T pattern: Prepend images to next text chapter
                merged_content = "\n\n".join(images_content) + "\n\n" + chapters[j].content
                merged.append(
                    Chapter(
                        name=chapters[j].name,
                        content=merged_content,
                        index=chapters[j].index,
                    )
                )
                used.add(j)
            elif merged and i > 0 and types[i - 1] == ChapterType.TEXT_ONLY:
                # T→I pattern: Append images to previous text chapter
                prev = merged[-1]
                merged[-1] = Chapter(
                    name=prev.name,
                    content=prev.content + "\n\n" + "\n\n".join(images_content),
                    index=prev.index,
                )
            else:
                # No suitable merge target, keep as-is
                merged.append(ch)
            used.add(i)
        else:
            merged.append(ch)
            used.add(i)

    merge_count = len(chapters) - len(merged)
    if merge_count > 0:
        logger.info(f"Merged {merge_count} image-only chapters into adjacent text chapters")

    return merged


def convert_epub_by_chapters(epub_path: str | Path) -> tuple[epub.EpubBook, list[Chapter]]:
    """Convert each EPUB chapter to markdown.

    Reads an EPUB file and converts each document item (chapter) to markdown
    format using MarkItDown for LLM-friendly text processing.

    This function is designed for use in the pipeline architecture without
    UI dependencies - progress tracking should be handled by the caller.

    Args:
        epub_path: Path to the EPUB file to convert.

    Returns:
        A tuple containing:
            - EpubBook object with metadata and images
            - List of Chapter objects with markdown content

    Raises:
        FileNotFoundError: If the EPUB file doesn't exist.
        Exception: If the EPUB file is corrupted or cannot be read.

    Example:
        >>> book, chapters = convert_epub_by_chapters("cookbook.epub")
        >>> print(f"Found {len(chapters)} chapters")
    """
    from .chapter_extractor import Chapter

    # ebooklib has no type stubs, suppress partial type warnings
    book = epub.read_epub(str(epub_path), {"ignore_ncx": True})  # pyright: ignore[reportUnknownMemberType]
    md = MarkItDown()

    items: list[Any] = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    chapters: list[Chapter] = []

    for i, item in enumerate(items):
        html_content: bytes = cast(bytes, item.get_content())
        result = md.convert_stream(BytesIO(html_content), file_extension=".html")
        markdown_content = result.text_content

        chapter_name: str = cast(str, item.get_name())
        chapters.append(Chapter(name=chapter_name, content=markdown_content, index=i))

    # Merge image-only chapters with adjacent text chapters
    # This ensures the LLM sees images alongside recipe text
    chapters = merge_image_chapters(chapters)

    logger.info(f"Converted {len(chapters)} chapters to markdown")
    return book, chapters


class EpubConverter:
    """Convert EPUB files to markdown using MarkItDown.

    This converter uses the MarkItDown library to transform EPUB files into
    clean markdown that's optimized for LLM processing. It includes utilities
    for estimating token counts, chunking large files, and removing non-recipe
    content like forewords and tables of contents.
    """

    def __init__(self) -> None:
        """Initialize the EPUB converter with MarkItDown."""
        self.md = MarkItDown()

    def convert_epub_to_markdown(self, epub_path: str) -> str:
        """Convert an EPUB file to markdown.

        Args:
            epub_path: Path to the EPUB file to convert

        Returns:
            Markdown content as a string

        Raises:
            OSError: If file cannot be read
            ValueError: If conversion fails
        """
        logging.info(f"Converting EPUB to Markdown: {epub_path}")

        try:
            result = self.md.convert(epub_path)
            markdown_content = result.text_content

            if not markdown_content or len(markdown_content.strip()) < 100:
                logging.warning(
                    f"Conversion produced minimal content: {len(markdown_content)} chars"
                )
            else:
                logging.info(f"Conversion successful: {len(markdown_content)} characters")

            return markdown_content

        except (OSError, ValueError, AttributeError) as e:
            logging.error(f"Error converting EPUB to Markdown: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (1 token ≈ 4 characters).

        This is a simple heuristic that works reasonably well for English text.
        For more accurate token counting, use tiktoken with the specific model's
        tokenizer.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count

        Examples:
            >>> converter = EpubConverter()
            >>> converter.estimate_tokens("Hello world!")
            3
            >>> converter.estimate_tokens("A" * 400)
            100
        """
        return len(text) // 4

    def needs_chunking(self, markdown: str, max_tokens: int = 200000) -> bool:
        """Check if markdown content exceeds token limit and needs chunking.

        Uses a conservative estimate to determine if content will fit within
        the specified token limit. Defaults to 200K tokens to leave a buffer
        for 256K context window models.

        Args:
            markdown: The markdown content to check
            max_tokens: Maximum token limit. Defaults to 200000 (leaving buffer for 256K context).

        Returns:
            True if content needs to be chunked, False otherwise
        """
        estimated_tokens = self.estimate_tokens(markdown)
        logging.info(f"Estimated tokens: {estimated_tokens}")
        return estimated_tokens > max_tokens

    def chunk_by_headings(self, markdown: str, level: int = 1) -> list[str]:
        r"""Split markdown into chunks based on heading level.

        Splits markdown at headings of the specified level, preserving the
        heading in each chunk. Useful for splitting by chapters (level 1) or
        sections (level 2).

        Args:
            markdown: The markdown content to chunk
            level: Heading level to split on (1 = #, 2 = ##, etc.). Defaults to 1.

        Returns:
            List of markdown chunks, each starting with a heading

        Examples:
            >>> content = "# Chapter 1\\nText\\n# Chapter 2\\nMore text"
            >>> converter = EpubConverter()
            >>> chunks = converter.chunk_by_headings(content, level=1)
            >>> len(chunks)
            2
        """
        heading_prefix = "#" * level + " "
        lines = markdown.split("\n")

        chunks: list[str] = []
        current_chunk: list[str] = []

        for line in lines:
            if line.startswith(heading_prefix) and current_chunk:
                # Start new chunk when we hit a heading and have content
                chunks.append("\n".join(current_chunk))
                current_chunk = []

            current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logging.info(f"Split into {len(chunks)} chunks at heading level {level}")
        return chunks

    def strip_front_matter(self, markdown: str, min_ingredient_count: int = 3) -> str:
        """Remove front matter (foreword, intro, TOC) and keep recipe content.

        Looks for where recipes actually start by detecting ingredient-like
        patterns (measurements, cooking terms) and strips everything before
        that point.

        Args:
            markdown: The full markdown content
            min_ingredient_count: Minimum ingredient-like lines before considering it
                recipe content. Defaults to 3.

        Returns:
            Markdown with front matter removed

        Notes:
            This is a heuristic approach that works well for typical cookbooks
            but may need tuning for specific book formats.
        """
        lines = markdown.split("\n")

        # Look for recipe patterns
        ingredient_patterns = [
            "tablespoon",
            "teaspoon",
            "cup",
            "g ",
            "ml ",
            "kg ",
            "serves",
            "yield",
        ]
        recipe_start_idx = 0

        for i in range(len(lines)):
            # Look ahead 20 lines and count ingredient-like lines
            lookahead = lines[i : i + 20]
            ingredient_count = sum(
                1
                for line in lookahead
                if any(pattern in line.lower() for pattern in ingredient_patterns)
            )

            if ingredient_count >= min_ingredient_count:
                recipe_start_idx = max(0, i - 10)  # Start a bit before
                logging.info(f"Found recipe content starting at line {recipe_start_idx}")
                break

        if recipe_start_idx > 0:
            stripped = "\n".join(lines[recipe_start_idx:])
            logging.info(f"Stripped {recipe_start_idx} lines of front matter")
            logging.info(f"Content reduced from {len(markdown)} to {len(stripped)} characters")
            return stripped

        return markdown

    def smart_chunk(self, markdown: str, max_tokens: int = 200000) -> list[str]:
        """Intelligently chunk markdown content if needed.

        First strips front matter to remove non-recipe content, then chunks
        by progressively deeper heading levels if the content still exceeds
        the token limit.

        Args:
            markdown: The markdown content to chunk
            max_tokens: Maximum tokens per chunk. Defaults to 200000.

        Returns:
            List of markdown chunks (single item if no chunking needed)

        Notes:
            - Tries heading levels 1-3 in sequence
            - Returns unchunked content if chunking isn't possible
            - Warns if unable to chunk within token limits
        """
        # First, strip front matter
        markdown = self.strip_front_matter(markdown)

        if not self.needs_chunking(markdown, max_tokens):
            return [markdown]

        logging.info("Content exceeds token limit, attempting to chunk...")

        # Try progressively deeper heading levels
        for level in range(1, 4):
            chunks = self.chunk_by_headings(markdown, level)

            # Check if all chunks are within limit
            all_fit = all(self.estimate_tokens(chunk) <= max_tokens for chunk in chunks)

            if all_fit:
                logging.info(f"Successfully chunked at heading level {level}")
                return chunks

        # If still too large, warn and return as-is
        logging.warning("Unable to chunk content within token limits, returning as single chunk")
        return [markdown]


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python converter.py <epub_file>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    converter = EpubConverter()
    epub_path = sys.argv[1]

    markdown = converter.convert_epub_to_markdown(epub_path)
    print(f"\n{'=' * 80}\n")
    print("Markdown Preview (first 2000 chars):\n")
    print(markdown[:2000])
    print(f"\n{'=' * 80}\n")
    print(f"Total length: {len(markdown)} characters")
    print(f"Estimated tokens: {converter.estimate_tokens(markdown)}")
    print(f"Needs chunking: {converter.needs_chunking(markdown)}")
