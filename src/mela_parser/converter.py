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
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown

if TYPE_CHECKING:
    from .chapter_extractor import Chapter

logger = logging.getLogger(__name__)


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

    book = epub.read_epub(str(epub_path), {"ignore_ncx": True})
    md = MarkItDown()

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    chapters: list[Chapter] = []

    for i, item in enumerate(items):
        html_content = item.get_content()
        result = md.convert_stream(BytesIO(html_content), file_extension=".html")
        markdown_content = result.text_content

        chapter_name = item.get_name()
        chapters.append(Chapter(name=chapter_name, content=markdown_content, index=i))

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
            Exception: If conversion fails or file cannot be read
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

        except Exception as e:
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
