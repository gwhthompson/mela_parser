#!/usr/bin/env python3
"""
Delimiter-based EPUB cookbook parser.
Uses marker insertion to identify recipe boundaries, then extracts each recipe individually.
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
from parse import RecipeMarkerInserter, RecipeParser
from recipe import RecipeProcessor


def setup_logging(log_file: str = "process_delimiter.log"):
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
        description="Process EPUB cookbook using delimiter-based extraction."
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
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory (default: output)",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        choices=["gpt-5-nano", "gpt-5-mini"],
        default="gpt-5-mini",
        help="Fallback model for failed extractions (default: gpt-5-mini)",
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

    # PHASE 2: Chunk for Processing
    logging.info("=" * 80)
    logging.info("PHASE 2: Chunking Content")
    logging.info("=" * 80)

    # Chunk into smaller pieces (max ~25K tokens = ~100K chars per chunk)
    # This allows model to output full chunk with markers
    def chunk_by_size(text: str, max_chars: int = 100000) -> List[str]:
        """Chunk text into pieces small enough for model output."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_chars and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    chunks = chunk_by_size(markdown, max_chars=80000)  # Leave buffer for output
    logging.info(f"Split into {len(chunks)} chunks for processing")

    # PHASE 3: Insert Recipe Markers in Each Chunk
    logging.info("=" * 80)
    logging.info("PHASE 3: Inserting Recipe Markers")
    logging.info("=" * 80)

    marker_inserter = RecipeMarkerInserter(model=args.model)
    marked_chunks = []

    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk):,} chars)")
        try:
            marked_chunk = marker_inserter.insert_markers(chunk, book_title)
            marked_chunks.append(marked_chunk)
        except Exception as e:
            logging.error(f"Failed to process chunk {i}: {e}")
            # Use original chunk if marking fails
            marked_chunks.append(chunk)

    # Reassemble
    marked_text = '\n\n'.join(marked_chunks)

    # PHASE 4: Split into Recipe Sections
    logging.info("=" * 80)
    logging.info("PHASE 4: Splitting into Recipe Sections")
    logging.info("=" * 80)

    recipe_sections = marker_inserter.split_by_markers(marked_text)
    logging.info(f"Found {len(recipe_sections)} recipe sections")

    # PHASE 5: Extract Each Recipe
    logging.info("=" * 80)
    logging.info("PHASE 5: Extracting Recipes")
    logging.info("=" * 80)

    extracted_recipes = []
    failed_extractions = []

    for i, section in enumerate(recipe_sections, 1):
        try:
            # Try with primary model
            parser = RecipeParser(section, model=args.model)
            mela_recipe = parser.parse()
            extracted_recipes.append(mela_recipe)
            logging.info(f"✓ Recipe {i}/{len(recipe_sections)}: {mela_recipe.title}")

        except Exception as e:
            logging.warning(f"✗ Recipe {i}/{len(recipe_sections)} failed with {args.model}: {e}")

            # Try fallback model
            try:
                parser = RecipeParser(section, model=args.fallback_model)
                mela_recipe = parser.parse()
                extracted_recipes.append(mela_recipe)
                logging.info(
                    f"✓ Recipe {i}/{len(recipe_sections)}: {mela_recipe.title} (fallback successful)"
                )
            except Exception as e2:
                logging.error(f"✗ Recipe {i}/{len(recipe_sections)} failed even with fallback: {e2}")
                failed_extractions.append({"index": i, "section_preview": section[:500]})

    extraction_time = time.time() - start_time

    logging.info("=" * 80)
    logging.info(f"Extraction Summary:")
    logging.info(f"  Total sections: {len(recipe_sections)}")
    logging.info(f"  Successfully extracted: {len(extracted_recipes)}")
    logging.info(f"  Failed: {len(failed_extractions)}")
    logging.info(f"  Success rate: {len(extracted_recipes)/len(recipe_sections)*100:.1f}%")
    logging.info(f"  Time: {extraction_time:.2f}s")
    logging.info("=" * 80)

    # PHASE 6: Write Recipes to Disk
    logging.info("=" * 80)
    logging.info("PHASE 6: Writing Recipes to Disk")
    logging.info("=" * 80)

    out_dir = Path(args.output_dir) / f"{book_slug}-delimiter"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written_count = 0
    skipped_count = 0

    for mela_recipe in extracted_recipes:
        try:
            # Convert MelaRecipe to dict format
            recipe_dict = processor._mela_recipe_to_object(mela_recipe)
            recipe_dict["link"] = book_title
            recipe_dict["id"] = str(uuid.uuid4())

            # Note: Image extraction skipped in delimiter approach
            # Could be added back by keeping EPUB reference
            recipe_dict["images"] = []

            # Write recipe
            output_path = processor.write_recipe(recipe_dict, output_dir=str(out_dir))

            if output_path:
                written_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            skipped_count += 1
            logging.error(f"✗ Error writing recipe: {e}")

    logging.info(f"Written: {written_count}, Skipped: {skipped_count}")

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
    logging.info(f"Recipe sections found: {len(recipe_sections)}")
    logging.info(f"Recipes extracted: {len(extracted_recipes)}")
    logging.info(f"Success rate: {len(extracted_recipes)/len(recipe_sections)*100:.1f}%")
    logging.info(f"Recipes written: {written_count}")
    logging.info(f"Total time: {time.time() - start_time:.2f}s")
    logging.info(f"Output: {archive_mela}")

    if failed_extractions:
        logging.warning(f"\nFailed extractions: {len(failed_extractions)}")
        for failure in failed_extractions[:5]:  # Show first 5
            logging.warning(f"  Section {failure['index']}: {failure['section_preview'][:200]}...")


if __name__ == "__main__":
    main()
