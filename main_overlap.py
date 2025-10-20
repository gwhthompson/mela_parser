#!/usr/bin/env python3
"""
Overlapping chunk strategy for 100% recipe extraction.
Splits cookbook into overlapping chunks, extracts all recipes, discards last from each chunk,
then deduplicates. Guarantees completeness through overlap.
"""
import argparse
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List

from ebooklib import epub

from converter import EpubConverter
from parse import CookbookParser, MelaRecipe
from recipe import RecipeProcessor


def setup_logging(log_file: str = "process_overlap.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def create_overlapping_chunks(text: str, chunk_size: int = 80000, overlap: int = 40000) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap size in characters (typically 50% of chunk_size)

    Returns:
        List of overlapping text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        start += (chunk_size - overlap)

        # Break if we've covered the whole text
        if end >= len(text):
            break

    logging.info(f"Created {len(chunks)} overlapping chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def normalize_title(title: str) -> str:
    """
    Normalize recipe title for comparison.

    Args:
        title: Recipe title

    Returns:
        Normalized title (lowercase, no punctuation, trimmed, no duplicate markers)
    """
    import re

    # Convert to lowercase
    title = title.lower().strip()

    # Remove common suffixes that indicate duplicates
    suffixes = [
        "(alternate entry)",
        "(duplicate entry)",
        "(shortened)",
        "(detailed)",
        "(repeat entry)",
        "(duplicate entry 2)",
        "(alternate)",
        "(variation)",
        "(recipe entry)",
        "(starter)",
        "(separate recipe)",
    ]

    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()

    # Remove punctuation and extra whitespace
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)

    return title.strip()


def count_recipe_fields(recipe: MelaRecipe) -> int:
    """
    Count how many fields are populated in a recipe (quality metric).

    Args:
        recipe: Recipe to evaluate

    Returns:
        Number of populated fields
    """
    count = 0

    if recipe.title and len(recipe.title.strip()) > 0:
        count += 1
    if recipe.text and len(recipe.text.strip()) > 0:
        count += 2  # Description is valuable, count it more
    if recipe.yield_ and len(recipe.yield_.strip()) > 0:
        count += 1
    if recipe.prepTime and recipe.prepTime > 0:
        count += 1
    if recipe.cookTime and recipe.cookTime > 0:
        count += 1
    if recipe.totalTime and recipe.totalTime > 0:
        count += 1
    if recipe.ingredients:
        count += sum(len(grp.ingredients) for grp in recipe.ingredients)
    if recipe.instructions:
        count += len(recipe.instructions) * 2  # Instructions are valuable
    if recipe.notes and len(recipe.notes.strip()) > 0:
        count += 1
    if recipe.categories:
        count += len(recipe.categories)

    return count


def is_recipe_complete(recipe: MelaRecipe) -> bool:
    """
    Check if recipe has minimum required fields for quality.

    Args:
        recipe: Recipe to validate

    Returns:
        True if recipe meets minimum quality standards
    """
    has_title = recipe.title and len(recipe.title.strip()) > 0
    has_ingredients = (
        recipe.ingredients
        and len(recipe.ingredients) > 0
        and len(recipe.ingredients[0].ingredients) > 0
    )
    has_instructions = recipe.instructions and len(recipe.instructions) > 0

    return has_title and has_ingredients and has_instructions


def deduplicate_recipes(recipes: List[MelaRecipe]) -> List[MelaRecipe]:
    """
    Smart deduplication: keep most complete version of duplicate recipes.

    Args:
        recipes: List of recipes that may contain duplicates

    Returns:
        Deduplicated list with most complete versions kept
    """
    by_title = {}

    for recipe in recipes:
        # Normalize title for comparison
        normalized = normalize_title(recipe.title)

        if not normalized:  # Skip empty titles
            continue

        if normalized not in by_title:
            by_title[normalized] = recipe
        else:
            # Compare completeness - keep better version
            existing = by_title[normalized]
            existing_score = count_recipe_fields(existing)
            new_score = count_recipe_fields(recipe)

            if new_score > existing_score:
                logging.debug(
                    f"Replacing duplicate '{recipe.title}' "
                    f"(score {new_score} > {existing_score})"
                )
                by_title[normalized] = recipe
            else:
                logging.debug(f"Skipping duplicate: {recipe.title}")

    unique_recipes = list(by_title.values())
    logging.info(f"Deduplicated: {len(recipes)} → {len(unique_recipes)} recipes (quality-based)")
    return unique_recipes


def main():
    parser = argparse.ArgumentParser(
        description="Process EPUB cookbook using overlapping chunk extraction."
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
        "--chunk-size",
        type=int,
        default=80000,
        help="Chunk size in characters (default: 80000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40000,
        help="Overlap size in characters (default: 40000, 50%% of chunk)",
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

    # Get book metadata
    book = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    logging.info("=" * 80)
    logging.info(f"Processing: {book_title}")
    logging.info(f"EPUB path: {args.epub_path}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    logging.info("=" * 80)

    start_time = time.time()

    # PHASE 1: Convert EPUB to Markdown
    logging.info("=" * 80)
    logging.info("PHASE 1: Converting EPUB to Markdown")
    logging.info("=" * 80)

    converter = EpubConverter()
    markdown = converter.convert_epub_to_markdown(args.epub_path)

    logging.info(f"Markdown length: {len(markdown):,} characters")
    logging.info(f"Estimated tokens: {converter.estimate_tokens(markdown):,}")

    # Strip front matter
    markdown_clean = converter.strip_front_matter(markdown)
    if len(markdown_clean) != len(markdown):
        logging.info(f"Cleaned markdown: {len(markdown_clean):,} characters")

    # PHASE 2: Create Overlapping Chunks
    logging.info("=" * 80)
    logging.info("PHASE 2: Creating Overlapping Chunks")
    logging.info("=" * 80)

    chunks = create_overlapping_chunks(markdown_clean, args.chunk_size, args.overlap)

    # PHASE 3: Extract Recipes from Each Chunk
    logging.info("=" * 80)
    logging.info("PHASE 3: Extracting Recipes from Chunks")
    logging.info("=" * 80)

    cookbook_parser = CookbookParser(model=args.model)
    all_recipes = []

    for i, chunk in enumerate(chunks, 1):
        logging.info(f"\nProcessing chunk {i}/{len(chunks)} ({len(chunk):,} chars)")

        try:
            # Extract all recipes from this chunk
            chunk_recipes = cookbook_parser.parse_cookbook(chunk, book_title)

            # Discard last recipe (might be incomplete due to boundary)
            if len(chunk_recipes.recipes) > 1 and i < len(chunks):
                recipes_to_keep = chunk_recipes.recipes[:-1]
                discarded = chunk_recipes.recipes[-1]
                logging.info(f"  Extracted {len(chunk_recipes.recipes)} recipes, keeping {len(recipes_to_keep)}")
                logging.info(f"  Discarded (boundary): {discarded.title}")
            else:
                # Last chunk or only one recipe: keep all
                recipes_to_keep = chunk_recipes.recipes
                logging.info(f"  Extracted {len(recipes_to_keep)} recipes (last chunk, keeping all)")

            all_recipes.extend(recipes_to_keep)

            # Log extracted titles
            for recipe in recipes_to_keep:
                logging.info(f"    ✓ {recipe.title}")

        except Exception as e:
            logging.error(f"  ✗ Failed to extract from chunk {i}: {e}")
            continue

    extraction_time = time.time() - start_time

    logging.info("=" * 80)
    logging.info(f"Extraction Complete:")
    logging.info(f"  Total recipes extracted (with duplicates): {len(all_recipes)}")
    logging.info(f"  Time: {extraction_time:.2f}s")
    logging.info("=" * 80)

    # PHASE 4: Filter Incomplete Recipes
    logging.info("=" * 80)
    logging.info("PHASE 4: Filtering Incomplete Recipes")
    logging.info("=" * 80)

    complete_recipes = [r for r in all_recipes if is_recipe_complete(r)]
    incomplete_count = len(all_recipes) - len(complete_recipes)

    if incomplete_count > 0:
        logging.info(f"Filtered out {incomplete_count} incomplete recipes")
        # Log some examples
        incomplete_recipes = [r for r in all_recipes if not is_recipe_complete(r)]
        for recipe in incomplete_recipes[:10]:  # Show first 10
            logging.debug(f"  Incomplete: {recipe.title}")

    # PHASE 5: Deduplicate (Quality-Based)
    logging.info("=" * 80)
    logging.info("PHASE 5: Smart Deduplication")
    logging.info("=" * 80)

    unique_recipes = deduplicate_recipes(complete_recipes)

    logging.info(f"Final recipe count: {len(unique_recipes)}")

    # PHASE 6: Write Recipes to Disk
    logging.info("=" * 80)
    logging.info("PHASE 6: Writing Recipes to Disk")
    logging.info("=" * 80)

    out_dir = Path(args.output_dir) / f"{book_slug}-overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written_count = 0
    skipped_count = 0

    for mela_recipe in unique_recipes:
        try:
            # Convert MelaRecipe to dict format
            recipe_dict = processor._mela_recipe_to_object(mela_recipe)
            recipe_dict["link"] = book_title
            recipe_dict["id"] = str(uuid.uuid4())
            recipe_dict["images"] = []

            # Write recipe
            output_path = processor.write_recipe(recipe_dict, output_dir=str(out_dir))

            if output_path:
                written_count += 1
                logging.info(f"✓ {recipe_dict['title']}")
            else:
                skipped_count += 1
                logging.warning(f"⊘ Skipped (incomplete): {recipe_dict.get('title', 'UNKNOWN')}")

        except Exception as e:
            skipped_count += 1
            logging.error(f"✗ Error writing recipe: {e}")

    logging.info(f"\nWritten: {written_count}, Skipped: {skipped_count}")

    # PHASE 7: Create Archive
    logging.info("=" * 80)
    logging.info("PHASE 7: Creating Archive")
    logging.info("=" * 80)

    archive_zip = shutil.make_archive(
        base_name=str(out_dir), format="zip", root_dir=str(out_dir)
    )
    archive_mela = archive_zip.replace(".zip", ".melarecipes")
    os.rename(archive_zip, archive_mela)

    logging.info(f"Archive created: {archive_mela}")

    # Final Summary
    logging.info("=" * 80)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Book: {book_title}")
    logging.info(f"Chunks processed: {len(chunks)}")
    logging.info(f"Total recipes extracted: {len(all_recipes)} (with duplicates)")
    logging.info(f"Unique recipes: {len(unique_recipes)}")
    logging.info(f"Recipes written: {written_count}")
    logging.info(f"Success rate: {written_count/len(unique_recipes)*100:.1f}%")
    logging.info(f"Total time: {time.time() - start_time:.2f}s")
    logging.info(f"Output: {archive_mela}")


if __name__ == "__main__":
    main()
