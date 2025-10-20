#!/usr/bin/env python3
"""
EPUB to Markdown converter using MarkItDown.
Converts entire EPUB cookbooks to clean, LLM-friendly markdown.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from markitdown import MarkItDown


class EpubConverter:
    """Converts EPUB files to markdown using MarkItDown."""

    def __init__(self):
        self.md = MarkItDown()

    def convert_epub_to_markdown(self, epub_path: str) -> str:
        """
        Convert an EPUB file to markdown.

        Args:
            epub_path: Path to the EPUB file

        Returns:
            Markdown content as a string
        """
        logging.info(f"Converting EPUB to Markdown: {epub_path}")

        try:
            result = self.md.convert(epub_path)
            markdown_content = result.text_content

            if not markdown_content or len(markdown_content.strip()) < 100:
                logging.warning(f"Conversion produced minimal content: {len(markdown_content)} chars")
            else:
                logging.info(f"Conversion successful: {len(markdown_content)} characters")

            return markdown_content

        except Exception as e:
            logging.error(f"Error converting EPUB to Markdown: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count (1 token ≈ 4 characters).

        Args:
            text: The text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def needs_chunking(self, markdown: str, max_tokens: int = 200000) -> bool:
        """
        Check if markdown content exceeds token limit and needs chunking.

        Args:
            markdown: The markdown content
            max_tokens: Maximum token limit (default: 200K, leaving buffer for 256K context)

        Returns:
            True if content needs to be chunked
        """
        estimated_tokens = self.estimate_tokens(markdown)
        logging.info(f"Estimated tokens: {estimated_tokens}")
        return estimated_tokens > max_tokens

    def chunk_by_headings(self, markdown: str, level: int = 1) -> list[str]:
        """
        Split markdown into chunks based on heading level.

        Args:
            markdown: The markdown content
            level: Heading level to split on (1 = #, 2 = ##, etc.)

        Returns:
            List of markdown chunks
        """
        heading_prefix = "#" * level + " "
        lines = markdown.split("\n")

        chunks = []
        current_chunk = []

        for line in lines:
            if line.startswith(heading_prefix):
                # Start new chunk if we have content
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []

            current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logging.info(f"Split into {len(chunks)} chunks at heading level {level}")
        return chunks

    def strip_front_matter(self, markdown: str, min_ingredient_count: int = 3) -> str:
        """
        Remove front matter (foreword, intro, TOC) and keep recipe content.
        Looks for where recipes actually start (lines with ingredient-like patterns).

        Args:
            markdown: The full markdown content
            min_ingredient_count: Minimum ingredient-like lines before considering it recipe content

        Returns:
            Markdown with front matter removed
        """
        lines = markdown.split("\n")

        # Look for recipe patterns
        ingredient_patterns = ["tablespoon", "teaspoon", "cup", "g ", "ml ", "kg ", "serves", "yield"]
        recipe_start_idx = 0

        for i in range(len(lines)):
            # Look ahead 20 lines and count ingredient-like lines
            lookahead = lines[i:i+20]
            ingredient_count = sum(
                1 for line in lookahead
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
        """
        Intelligently chunk markdown content if needed.
        First strips front matter, then chunks if still too large.

        Args:
            markdown: The markdown content
            max_tokens: Maximum tokens per chunk

        Returns:
            List of markdown chunks (or single item if no chunking needed)
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
    print(f"\n{'='*80}\n")
    print(f"Markdown Preview (first 2000 chars):\n")
    print(markdown[:2000])
    print(f"\n{'='*80}\n")
    print(f"Total length: {len(markdown)} characters")
    print(f"Estimated tokens: {converter.estimate_tokens(markdown)}")
    print(f"Needs chunking: {converter.needs_chunking(markdown)}")
