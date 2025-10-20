#!/usr/bin/env python3
"""
Simplified EPUB cookbook parser using MarkItDown + single-pass LLM extraction.
Supports both GPT-5-nano and GPT-5-mini for comparison.
"""
import argparse
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from ebooklib import epub

from converter import EpubConverter
from parse import CookbookParser
from recipe import RecipeProcessor


def setup_logging(log_file: str = "process_v2.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process an EPUB cookbook using MarkItDown + single-pass extraction."
    )
    parser.add_argument(
        "epub_path", type=str, help="Path to the EPUB file containing the recipes"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-5-nano", "gpt-5-mini"],
        default="gpt-5-nano",
        help="Model to use for extraction (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both models and compare results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory (default: output)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"The file at {args.epub_path} does not exist.")

    setup_logging()

    # Get book title for organization
    book = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    logging.info(f"Processing: {book_title}")
    logging.info(f"EPUB path: {args.epub_path}")

    # Convert EPUB to Markdown
    logging.info("=" * 80)
    logging.info("PHASE 1: Converting EPUB to Markdown")
    logging.info("=" * 80)

    converter = EpubConverter()
    markdown = converter.convert_epub_to_markdown(args.epub_path)

    logging.info(f"Markdown length: {len(markdown)} characters")
    logging.info(f"Estimated tokens: {converter.estimate_tokens(markdown)}")

    # Check if chunking is needed
    chunks = converter.smart_chunk(markdown)
    logging.info(f"Processing in {len(chunks)} chunk(s)")

    # Determine which models to test
    models_to_test = ["gpt-5-nano", "gpt-5-mini"] if args.compare else [args.model]

    results = {}

    for model in models_to_test:
        logging.info("=" * 80)
        logging.info(f"PHASE 2: Extracting recipes with {model}")
        logging.info("=" * 80)

        start_time = time.time()

        # Extract recipes
        cookbook_parser = CookbookParser(model=model)
        all_recipes = []

        for i, chunk in enumerate(chunks, 1):
            logging.info(f"Processing chunk {i}/{len(chunks)}")
            cookbook_recipes = cookbook_parser.parse_cookbook(chunk, book_title)
            all_recipes.extend(cookbook_recipes.recipes)

        extraction_time = time.time() - start_time

        logging.info(f"Total recipes extracted: {len(all_recipes)}")
        logging.info(f"Extraction time: {extraction_time:.2f}s")

        # Write recipes to disk
        logging.info("=" * 80)
        logging.info("PHASE 3: Writing recipes to disk")
        logging.info("=" * 80)

        output_suffix = f"-{model}" if args.compare else ""
        out_dir = Path(args.output_dir) / f"{book_slug}{output_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        processor = RecipeProcessor(args.epub_path, book)
        written_count = 0
        skipped_count = 0

        for mela_recipe in all_recipes:
            try:
                # Convert MelaRecipe to dict format expected by write_recipe
                recipe_dict = processor._mela_recipe_to_object(mela_recipe)
                recipe_dict["link"] = book_title
                recipe_dict["id"] = str(uuid.uuid4())

                # Process images (use existing logic from RecipeProcessor)
                # Note: For now, we skip image extraction in v2 since we don't have
                # the HTML source. This could be added back by keeping the EPUB
                # and extracting images separately.
                recipe_dict["images"] = []

                # Write recipe
                output_path = processor.write_recipe(recipe_dict, output_dir=str(out_dir))

                if output_path:
                    written_count += 1
                    logging.info(f"✓ {recipe_dict['title']}")
                else:
                    skipped_count += 1
                    logging.info(f"⊘ {recipe_dict.get('title', 'UNKNOWN')} (incomplete)")

            except Exception as e:
                skipped_count += 1
                logging.error(f"✗ Error writing recipe: {e}")

        logging.info(f"Written: {written_count}, Skipped: {skipped_count}")

        # Create .melarecipes archive
        logging.info("=" * 80)
        logging.info("PHASE 4: Creating archive")
        logging.info("=" * 80)

        archive_zip = shutil.make_archive(
            base_name=str(out_dir), format="zip", root_dir=str(out_dir)
        )
        archive_mela = archive_zip.replace(".zip", ".melarecipes")
        os.rename(archive_zip, archive_mela)

        logging.info(f"Archive created: {archive_mela}")

        # Store results for comparison
        results[model] = {
            "recipes_extracted": len(all_recipes),
            "recipes_written": written_count,
            "recipes_skipped": skipped_count,
            "extraction_time": extraction_time,
            "archive_path": archive_mela,
        }

    # Print comparison summary if testing both models
    if args.compare:
        logging.info("=" * 80)
        logging.info("MODEL COMPARISON SUMMARY")
        logging.info("=" * 80)

        for model, stats in results.items():
            logging.info(f"\n{model.upper()}:")
            logging.info(f"  Recipes extracted: {stats['recipes_extracted']}")
            logging.info(f"  Recipes written: {stats['recipes_written']}")
            logging.info(f"  Recipes skipped: {stats['recipes_skipped']}")
            logging.info(f"  Extraction time: {stats['extraction_time']:.2f}s")
            logging.info(f"  Archive: {stats['archive_path']}")

        # Calculate differences
        nano_stats = results["gpt-5-nano"]
        mini_stats = results["gpt-5-mini"]

        recipe_diff = nano_stats["recipes_extracted"] - mini_stats["recipes_extracted"]
        time_diff = nano_stats["extraction_time"] - mini_stats["extraction_time"]

        logging.info("\nDIFFERENCES (nano - mini):")
        logging.info(f"  Recipes: {recipe_diff:+d}")
        logging.info(f"  Time: {time_diff:+.2f}s")

    logging.info("\n" + "=" * 80)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
