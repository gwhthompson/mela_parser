#!/usr/bin/env python3
"""CLI for mela-parser: Extract recipes from EPUB cookbooks to Mela format.

This module provides the command-line interface for extracting recipes from EPUB
cookbooks using chapter-based extraction with parallel async processing for
maximum speed and efficiency.

The CLI supports various OpenAI models and allows customization of output
directories and image extraction options.
"""

import argparse
import asyncio
import logging
import os
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown

from .chapter_extractor import AsyncChapterExtractor, Chapter
from .recipe import RecipeProcessor


def setup_logging(log_file: str = "mela_parser.log") -> None:
    """Set up logging configuration for the application.

    Configures logging to output to both a file and the console with INFO level
    and timestamps.

    Args:
        log_file: Path to the log file. Defaults to "mela_parser.log".
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def convert_epub_by_chapters(epub_path: str) -> tuple[epub.EpubBook, list[tuple[str, str]]]:
    """Convert each EPUB chapter to markdown.

    Reads an EPUB file and converts each document item (chapter) to markdown
    format using MarkItDown for LLM-friendly text processing.

    Args:
        epub_path: Path to the EPUB file to convert.

    Returns:
        A tuple containing:
            - EpubBook object with metadata and images
            - List of tuples, each containing (chapter_name, markdown_content)

    Raises:
        FileNotFoundError: If the EPUB file doesn't exist.
        Exception: If the EPUB file is corrupted or cannot be read.
    """
    book = epub.read_epub(epub_path, {"ignore_ncx": True})
    md = MarkItDown()

    chapters: list[tuple[str, str]] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content()
        result = md.convert_stream(BytesIO(html_content), file_extension=".html")
        markdown_content = result.text_content

        chapter_name = item.get_name()
        chapters.append((chapter_name, markdown_content))

    logging.info(f"Converted {len(chapters)} chapters to markdown")
    return book, chapters


async def main_async() -> None:
    """Main async function for parallel recipe extraction processing.

    This is the core async function that orchestrates the entire extraction
    workflow:
    1. Parse command-line arguments
    2. Convert EPUB chapters to markdown
    3. Extract recipes from chapters in parallel
    4. Deduplicate recipes
    5. Write recipes to output directory
    6. Create .melarecipes archive

    The function uses high concurrency (200 parallel requests) for optimal
    performance with modern LLM APIs.

    Raises:
        FileNotFoundError: If the EPUB file doesn't exist.
        Exception: If any step in the extraction pipeline fails.
    """
    parser = argparse.ArgumentParser(
        description="Extract recipes from EPUB cookbooks to Mela format", prog="mela-parse"
    )
    parser.add_argument("epub_path", type=str, help="Path to EPUB cookbook file")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5-nano", "gpt-5-mini"],
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory (default: output)"
    )
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"File not found: {args.epub_path}")

    setup_logging()

    start_time = time.time()

    # Get metadata
    book_temp = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book_temp.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    logging.info("=" * 80)
    logging.info(f"Processing: {book_title}")
    logging.info("Method: Chapter-based extraction")
    logging.info("=" * 80)

    # PHASE 1: Convert chapters
    logging.info("\nPHASE 1: Converting chapters")
    book, chapters = convert_epub_by_chapters(args.epub_path)

    # PHASE 2: Extract from all chapters
    logging.info("\nPHASE 2: Extracting from chapters (PARALLEL)")

    # Convert to Chapter objects
    chapter_objs = [
        Chapter(name=name, content=content, index=i) for i, (name, content) in enumerate(chapters)
    ]

    # Extract recipes with high concurrency
    extractor = AsyncChapterExtractor(model=args.model)
    extraction_results = await extractor.extract_from_chapters(
        chapters=chapter_objs, expected_titles=None, max_concurrent=200
    )

    # Flatten all recipes from all chapters
    all_recipes = []
    for result in extraction_results:
        all_recipes.extend(result.recipes)
        if result.recipes:
            logging.info(f"{result.chapter_name}: ✓ {len(result.recipes)} recipes")
        else:
            logging.info(f"{result.chapter_name}")

    logging.info(f"\nTotal extracted: {len(all_recipes)} recipes")

    # PHASE 3: Deduplication
    logging.info("\nPHASE 3: Deduplication")
    seen: set[str] = set()
    unique_recipes = []

    for recipe in all_recipes:
        if recipe.title not in seen:
            seen.add(recipe.title)
            unique_recipes.append(recipe)
        else:
            logging.debug(f"  Duplicate: {recipe.title}")

    logging.info(f"Total: {len(all_recipes)} → Unique: {len(unique_recipes)}")

    # PHASE 4: Write recipes
    logging.info("\nPHASE 4: Writing recipes")
    out_dir = Path(args.output_dir) / book_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written = 0

    for recipe in unique_recipes:
        recipe_dict = processor._mela_recipe_to_object(recipe)
        recipe_dict["link"] = book_title
        recipe_dict["id"] = str(uuid.uuid4())

        # Pass through image paths from extracted recipe for processing
        if recipe.images:
            recipe_dict["images"] = recipe.images
            processor._process_images(recipe_dict)
        else:
            recipe_dict["images"] = []

        if processor.write_recipe(recipe_dict, output_dir=str(out_dir)):
            written += 1

    # Create archive
    archive_zip = shutil.make_archive(base_name=str(out_dir), format="zip", root_dir=str(out_dir))
    archive_mela = archive_zip.replace(".zip", ".melarecipes")
    os.rename(archive_zip, archive_mela)

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Chapters: {len(chapters)}")
    logging.info(f"Extracted: {len(all_recipes)}")
    logging.info(f"Unique: {len(unique_recipes)}")
    logging.info(f"Written: {written}")
    logging.info(f"Time: {time.time() - start_time:.1f}s")
    logging.info(f"Output: {archive_mela}")


def main() -> None:
    """Entry point for mela-parse CLI command.

    This function serves as the synchronous entry point that launches the async
    main function. It's called when running 'mela-parse' from the command line.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
