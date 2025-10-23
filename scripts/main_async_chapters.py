#!/usr/bin/env python3
"""
Production-ready async chapter-based EPUB cookbook extraction.

This script uses the new chapter_extractor module to process EPUB cookbooks
efficiently using async parallel processing and chapter-based extraction.

Features:
- Async parallel chapter processing for maximum speed
- Recipe list discovery and validation
- Comprehensive error handling and logging
- Integration with existing RecipeProcessor for output
- Detailed progress reporting and metrics

Usage:
    python main_async_chapters.py cookbook.epub
    python main_async_chapters.py cookbook.epub --model gpt-5-mini
    python main_async_chapters.py cookbook.epub --no-recipe-list
    python main_async_chapters.py cookbook.epub --max-concurrent 10
"""

import argparse
import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from ebooklib import epub

from mela_parser.chapter_extractor import (
    ChapterProcessor,
    EPUBConversionError,
    RecipeListDiscoverer,
    AsyncChapterExtractor,
    ValidationEngine,
)
from mela_parser.recipe import RecipeProcessor


def setup_logging(log_file: str = "async_chapters.log", verbose: bool = False) -> None:
    """
    Configure logging for the extraction process.

    Args:
        log_file: Path to log file
        verbose: Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ],
    )


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    """
    Main async processing function.

    Args:
        args: Parsed command-line arguments
    """
    start_time = time.time()

    # Get book metadata
    book_temp = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title_metadata = book_temp.get_metadata("DC", "title")
    book_title = book_title_metadata[0][0] if book_title_metadata else Path(args.epub_path).stem
    book_slug = RecipeProcessor.slugify(book_title)

    print_section_header(f"Processing: {book_title}")
    logging.info(f"EPUB Path: {args.epub_path}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Max Concurrent: {args.max_concurrent}")
    logging.info(f"Recipe List Discovery: {'Enabled' if not args.no_recipe_list else 'Disabled'}")

    # ========================================================================
    # PHASE 1: Split EPUB into Chapters
    # ========================================================================

    print_section_header("PHASE 1: Converting EPUB chapters to markdown")

    try:
        processor = ChapterProcessor(args.epub_path)
        chapters = await processor.split_into_chapters()

        if not chapters:
            logging.error("No chapters found in EPUB")
            return

        logging.info(f"Successfully converted {len(chapters)} chapters")

        # Show chapter summary
        total_chars = sum(len(ch.content) for ch in chapters)
        avg_chars = total_chars / len(chapters) if chapters else 0

        logging.info(f"Total content: {total_chars:,} characters")
        logging.info(f"Average per chapter: {avg_chars:,.0f} characters")

    except EPUBConversionError as e:
        logging.error(f"Failed to convert EPUB: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error during chapter conversion: {e}")
        return

    # ========================================================================
    # PHASE 2: Discover Recipe List
    # ========================================================================

    print_section_header("PHASE 2: Discovering recipe list")

    expected_titles = None

    if args.no_recipe_list:
        logging.info("Recipe list discovery disabled (--no-recipe-list)")
    else:
        try:
            discoverer = RecipeListDiscoverer()
            expected_titles = await discoverer.discover_from_chapters(chapters)

            if expected_titles:
                logging.info(f"✓ Discovered {len(expected_titles)} recipes in cookbook index")

                # Show sample titles
                sample_count = min(10, len(expected_titles))
                logging.info(f"Sample titles (first {sample_count}):")
                for i, title in enumerate(expected_titles[:sample_count], 1):
                    logging.info(f"  {i}. {title}")

                if len(expected_titles) > sample_count:
                    logging.info(f"  ... and {len(expected_titles) - sample_count} more")
            else:
                logging.info("No recipe list found in cookbook")

        except Exception as e:
            logging.warning(f"Recipe list discovery failed (continuing without it): {e}")
            expected_titles = None

    # ========================================================================
    # PHASE 3: Extract Recipes from Chapters (Async Parallel)
    # ========================================================================

    print_section_header("PHASE 3: Extracting recipes from chapters (async)")

    try:
        extractor = AsyncChapterExtractor(
            model=args.model,
            max_retries=args.max_retries,
        )

        extraction_start = time.time()

        results = await extractor.extract_from_chapters(
            chapters,
            expected_titles=expected_titles,
            max_concurrent=args.max_concurrent
        )

        extraction_time = time.time() - extraction_start

        # Log results summary
        total_recipes = sum(r.recipe_count for r in results)
        successful_chapters = sum(1 for r in results if r.is_success)
        failed_chapters = len(results) - successful_chapters

        logging.info(f"Extraction completed in {extraction_time:.1f}s")
        logging.info(f"Chapters processed: {len(results)}")
        logging.info(f"  Successful: {successful_chapters}")
        logging.info(f"  Failed: {failed_chapters}")
        logging.info(f"Total recipes extracted: {total_recipes}")

        # Show per-chapter breakdown
        logging.info("\nPer-chapter results:")
        for i, result in enumerate(results, 1):
            if result.is_success and result.recipe_count > 0:
                logging.info(f"  [{i:2d}] {result.chapter_name}: {result.recipe_count} recipes")
                for recipe in result.recipes:
                    logging.info(f"       ✓ {recipe.title}")
            elif result.is_success:
                logging.debug(f"  [{i:2d}] {result.chapter_name}: No recipes")
            else:
                logging.error(f"  [{i:2d}] {result.chapter_name}: ERROR - {result.error}")

        # Collect all recipes
        all_recipes = []
        for result in results:
            all_recipes.extend(result.recipes)

    except Exception as e:
        logging.error(f"Recipe extraction failed: {e}")
        return

    # ========================================================================
    # PHASE 4: Deduplication
    # ========================================================================

    print_section_header("PHASE 4: Deduplication")

    seen_titles = set()
    unique_recipes = []

    for recipe in all_recipes:
        if recipe.title not in seen_titles:
            seen_titles.add(recipe.title)
            unique_recipes.append(recipe)
        else:
            logging.debug(f"Skipping duplicate: {recipe.title}")

    logging.info(f"Total extracted: {len(all_recipes)}")
    logging.info(f"Unique recipes: {len(unique_recipes)}")
    if len(all_recipes) > len(unique_recipes):
        duplicates = len(all_recipes) - len(unique_recipes)
        logging.info(f"Duplicates removed: {duplicates}")

    # ========================================================================
    # PHASE 5: Validation
    # ========================================================================

    if expected_titles:
        print_section_header("PHASE 5: Validation")

        validator = ValidationEngine()
        diff = validator.create_diff(expected_titles, unique_recipes)

        # Generate and log report
        report = validator.generate_report(diff, max_items=20)
        for line in report.split("\n"):
            logging.info(line)

        # Quality check
        is_valid, message = validator.validate_extraction_quality(
            results,
            min_success_rate=0.8
        )

        if is_valid:
            logging.info(f"✓ Quality check: {message}")
        else:
            logging.warning(f"⚠ Quality check: {message}")

    # ========================================================================
    # PHASE 6: Write Recipes
    # ========================================================================

    print_section_header("PHASE 6: Writing recipes to disk")

    # Create output directory
    out_dir = Path(args.output_dir) / f"{book_slug}-async-chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Output directory: {out_dir}")

    # Load EPUB book for image processing
    book = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    recipe_processor = RecipeProcessor(args.epub_path, book)

    written = 0
    skipped = 0

    for recipe in unique_recipes:
        try:
            # Convert to RecipeDict format
            recipe_dict = recipe_processor._mela_recipe_to_object(recipe)
            recipe_dict["link"] = book_title
            recipe_dict["id"] = str(uuid.uuid4())

            # Note: Image extraction is disabled for performance
            # To enable images, remove this line and let RecipeProcessor handle it
            recipe_dict["images"] = []

            # Write recipe
            if recipe_processor.write_recipe(recipe_dict, output_dir=str(out_dir)):
                written += 1
                logging.debug(f"Wrote: {recipe.title}")
            else:
                skipped += 1
                logging.debug(f"Skipped: {recipe.title}")

        except Exception as e:
            logging.error(f"Failed to write recipe '{recipe.title}': {e}")
            skipped += 1

    logging.info(f"Recipes written: {written}")
    if skipped > 0:
        logging.info(f"Recipes skipped: {skipped}")

    # Create .melarecipes archive
    try:
        logging.info("Creating .melarecipes archive...")
        archive_zip = shutil.make_archive(
            base_name=str(out_dir),
            format="zip",
            root_dir=str(out_dir)
        )
        archive_mela = archive_zip.replace(".zip", ".melarecipes")
        os.rename(archive_zip, archive_mela)
        logging.info(f"✓ Archive created: {archive_mela}")
    except Exception as e:
        logging.error(f"Failed to create archive: {e}")
        archive_mela = None

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    total_time = time.time() - start_time

    print_section_header("EXTRACTION COMPLETE")

    print(f"Book Title:           {book_title}")
    print(f"Chapters:             {len(chapters)}")
    print(f"Total Extracted:      {len(all_recipes)}")
    print(f"Unique Recipes:       {len(unique_recipes)}")
    print(f"Written to Disk:      {written}")

    if expected_titles:
        print(f"\nValidation:")
        print(f"Expected Recipes:     {len(expected_titles)}")
        print(f"Exact Matches:        {len(diff.exact_matches)}")
        print(f"Missing:              {len(diff.missing_titles)}")
        print(f"Extra:                {len(diff.extra_titles)}")
        print(f"Match Rate:           {diff.match_rate:.1%}")

    print(f"\nPerformance:")
    print(f"Total Time:           {total_time:.1f}s")
    print(f"Extraction Time:      {extraction_time:.1f}s")
    print(f"Recipes/Second:       {len(unique_recipes) / total_time:.2f}")

    if archive_mela:
        print(f"\n✓ Output: {archive_mela}")

    print("\n" + "=" * 80 + "\n")


def main() -> None:
    """Parse arguments and run async main."""
    parser = argparse.ArgumentParser(
        description="Async chapter-based EPUB cookbook extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python main_async_chapters.py cookbook.epub

  # Use larger model for better accuracy
  python main_async_chapters.py cookbook.epub --model gpt-5-mini

  # Increase parallelism for faster processing
  python main_async_chapters.py cookbook.epub --max-concurrent 10

  # Skip recipe list discovery
  python main_async_chapters.py cookbook.epub --no-recipe-list

  # Custom output directory
  python main_async_chapters.py cookbook.epub --output-dir my_recipes

  # Verbose logging for debugging
  python main_async_chapters.py cookbook.epub --verbose
        """
    )

    parser.add_argument(
        "epub_path",
        type=str,
        help="Path to EPUB cookbook file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        choices=["gpt-5-nano", "gpt-5-mini"],
        help="OpenAI model to use for extraction (default: gpt-5-nano)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for recipes (default: output)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent chapter extractions (default: 5)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per chapter (default: 3)"
    )

    parser.add_argument(
        "--no-recipe-list",
        action="store_true",
        help="Skip recipe list discovery phase"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    args = parser.parse_args()

    # Validate EPUB file exists
    if not os.path.exists(args.epub_path):
        parser.error(f"EPUB file not found: {args.epub_path}")

    # Setup logging
    log_file = f"async_chapters_{int(time.time())}.log"
    setup_logging(log_file, args.verbose)

    logging.info("Starting async chapter-based extraction")
    logging.info(f"Log file: {log_file}")

    # Run async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.warning("\nExtraction interrupted by user")
        print("\n⚠ Extraction interrupted by user\n")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}\n")
        raise


if __name__ == "__main__":
    main()
