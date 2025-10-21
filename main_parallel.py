#!/usr/bin/env python3
"""
Parallel overlapping chunk strategy for maximum speed.
Processes all chunks concurrently for 5-10x speedup.
"""
import argparse
import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Tuple

from ebooklib import epub

from converter import EpubConverter
from parse import CookbookParser, MelaRecipe
from recipe import RecipeProcessor
from main_overlap import (
    create_overlapping_chunks,
    normalize_title,
    count_recipe_fields,
    is_recipe_complete,
    deduplicate_recipes,
)


def setup_logging(log_file: str = "process_parallel.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


async def process_chunk_async(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    book_title: str,
    model: str,
) -> Tuple[List[MelaRecipe], int]:
    """
    Process a single chunk asynchronously.

    Args:
        chunk: Markdown chunk to process
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks
        book_title: Book title for context
        model: Model to use

    Returns:
        Tuple of (recipes_to_keep, chunk_index)
    """
    chunk_num = chunk_index + 1
    logging.info(f"[Chunk {chunk_num}/{total_chunks}] Starting extraction ({len(chunk):,} chars)")

    try:
        parser = CookbookParser(model=model)
        # Note: OpenAI SDK responses.parse is not async, runs in executor
        loop = asyncio.get_event_loop()
        chunk_recipes = await loop.run_in_executor(
            None, lambda: parser.parse_cookbook(chunk, book_title)
        )

        # Discard last recipe (might be incomplete due to boundary)
        if len(chunk_recipes.recipes) > 1 and chunk_index < total_chunks - 1:
            recipes_to_keep = chunk_recipes.recipes[:-1]
            discarded = chunk_recipes.recipes[-1]
            logging.info(
                f"[Chunk {chunk_num}/{total_chunks}] Extracted {len(chunk_recipes.recipes)}, "
                f"keeping {len(recipes_to_keep)}, discarded: {discarded.title}"
            )
        else:
            recipes_to_keep = chunk_recipes.recipes
            logging.info(
                f"[Chunk {chunk_num}/{total_chunks}] Extracted {len(recipes_to_keep)} "
                f"(last chunk, keeping all)"
            )

        # Log titles
        for recipe in recipes_to_keep:
            logging.info(f"[Chunk {chunk_num}/{total_chunks}]   ✓ {recipe.title}")

        return (recipes_to_keep, chunk_index)

    except Exception as e:
        logging.error(f"[Chunk {chunk_num}/{total_chunks}] Failed: {e}")
        return ([], chunk_index)


async def process_all_chunks_parallel(
    chunks: List[str], book_title: str, model: str
) -> List[Tuple[MelaRecipe, int]]:
    """
    Process all chunks in parallel.

    Args:
        chunks: List of markdown chunks
        book_title: Book title
        model: Model to use

    Returns:
        List of (recipe, chunk_index) tuples
    """
    logging.info(f"Processing {len(chunks)} chunks in PARALLEL...")

    # Create tasks for all chunks
    tasks = [
        process_chunk_async(chunk, i, len(chunks), book_title, model)
        for i, chunk in enumerate(chunks)
    ]

    # Process all chunks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_recipes = []
    for recipes, chunk_index in results:
        for recipe in recipes:
            all_recipes.append((recipe, chunk_index))

    return all_recipes


async def async_main():
    """Async main function for parallel processing."""
    parser = argparse.ArgumentParser(
        description="Process EPUB cookbook using PARALLEL overlapping chunk extraction."
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
        default=60000,
        help="Overlap size in characters (default: 60000, 75%% of chunk)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory (default: output)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction entirely (faster)",
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
    logging.info(f"Mode: PARALLEL (async chunk processing)")
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

    # PHASE 3: Extract Recipes from All Chunks IN PARALLEL
    logging.info("=" * 80)
    logging.info("PHASE 3: Extracting Recipes (PARALLEL)")
    logging.info("=" * 80)

    all_recipes = await process_all_chunks_parallel(chunks, book_title, args.model)

    extraction_time = time.time() - start_time

    logging.info("=" * 80)
    logging.info(f"Parallel Extraction Complete:")
    logging.info(f"  Total recipes extracted (with duplicates): {len(all_recipes)}")
    logging.info(f"  Time: {extraction_time:.2f}s")
    logging.info(f"  Avg time per chunk: {extraction_time/len(chunks):.2f}s")
    logging.info(f"  Speedup: ~{len(chunks)}x vs sequential")
    logging.info("=" * 80)

    # PHASE 4: Filter Incomplete Recipes
    logging.info("=" * 80)
    logging.info("PHASE 4: Filtering Incomplete Recipes")
    logging.info("=" * 80)

    complete_tuples = [(r, idx) for r, idx in all_recipes if is_recipe_complete(r)]
    incomplete_count = len(all_recipes) - len(complete_tuples)

    if incomplete_count > 0:
        logging.info(f"Filtered out {incomplete_count} incomplete recipes")

    # PHASE 5: Deduplicate (Quality-Based)
    logging.info("=" * 80)
    logging.info("PHASE 5: Smart Deduplication")
    logging.info("=" * 80)

    complete_recipes_only = [r for r, _ in complete_tuples]
    unique_recipes_only = deduplicate_recipes(complete_recipes_only)

    # Rebuild with chunk indices
    title_to_chunk = {r.title: idx for r, idx in complete_tuples}
    unique_recipes = [(r, title_to_chunk[r.title]) for r in unique_recipes_only]

    logging.info(f"Final recipe count: {len(unique_recipes)}")

    # PHASE 6: Write Recipes to Disk
    logging.info("=" * 80)
    logging.info("PHASE 6: Writing Recipes to Disk")
    logging.info("=" * 80)

    out_dir = Path(args.output_dir) / f"{book_slug}-parallel"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written_count = 0
    skipped_count = 0

    for mela_recipe, chunk_index in unique_recipes:
        try:
            # Convert MelaRecipe to dict format
            recipe_dict = processor._mela_recipe_to_object(mela_recipe)
            recipe_dict["link"] = book_title
            recipe_dict["id"] = str(uuid.uuid4())

            # Skip images for speed (can be added back with async image processing)
            recipe_dict["images"] = []

            # Write recipe
            output_path = processor.write_recipe(recipe_dict, output_dir=str(out_dir))

            if output_path:
                written_count += 1
                logging.info(f"✓ {recipe_dict['title']}")
            else:
                skipped_count += 1

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
    total_time = time.time() - start_time

    logging.info("=" * 80)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Book: {book_title}")
    logging.info(f"Mode: PARALLEL processing")
    logging.info(f"Chunks processed: {len(chunks)} (concurrently)")
    logging.info(f"Total recipes extracted: {len(all_recipes)} (with duplicates)")
    logging.info(f"Unique recipes: {len(unique_recipes)}")
    logging.info(f"Recipes written: {written_count}")
    logging.info(f"Success rate: {written_count/len(unique_recipes)*100:.1f}%")
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    logging.info(f"Speedup: ~{len(chunks)}x vs sequential")
    logging.info(f"Output: {archive_mela}")


def main():
    """Entry point that runs async main."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
