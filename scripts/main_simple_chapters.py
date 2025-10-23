#!/usr/bin/env python3
"""
Simple chapter-based extraction - clean implementation using your exact pattern.
Processes each EPUB chapter, extracts recipes, filters incomplete ones.
PARALLEL ASYNC PROCESSING FOR MAXIMUM SPEED.
"""
import argparse
import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel

from mela_parser.parse import CookbookRecipes, MelaRecipe
from mela_parser.recipe import RecipeProcessor
from mela_parser.chapter_extractor import AsyncChapterExtractor, Chapter


def setup_logging(log_file: str = "simple_chapters.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


class RecipeList(BaseModel):
    """Schema for recipe list discovery."""
    titles: list[str]


def convert_epub_by_chapters(epub_path: str) -> tuple[epub.EpubBook, list[tuple[str, str]]]:
    """Convert each EPUB chapter to markdown using your exact pattern."""
    book = epub.read_epub(epub_path, {"ignore_ncx": True})
    md = MarkItDown()

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content()
        result = md.convert_stream(BytesIO(html_content), file_extension=".html")
        markdown_content = result.text_content

        chapter_name = item.get_name()
        chapters.append((chapter_name, markdown_content))

    logging.info(f"Converted {len(chapters)} chapters to markdown")
    return book, chapters


def discover_recipe_list(chapters: list[tuple[str, str]]) -> Optional[list[str]]:
    """Discover recipe list by finding link patterns."""
    client = OpenAI()

    all_link_sections = []
    for chapter_name, chapter_md in chapters:
        link_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links = re.findall(link_pattern, chapter_md)

        if len(links) > 5:
            all_link_sections.append("\n".join(links))

    if not all_link_sections:
        logging.info("No recipe list sections found")
        return None

    combined = "\n\n".join(all_link_sections)

    prompt = f"""Extract the unique list of recipe titles from these potential recipe lists.

Remove:
- Section headers (Contents, Index, About, etc.)
- Page numbers
- Duplicates

Keep:
- Actual recipe titles EXACTLY as written
- One entry per unique recipe

<potential_lists>
{combined}
</potential_lists>"""

    try:
        response = client.responses.parse(
            model="gpt-5-mini",
            input=[EasyInputMessageParam(role="user", content=prompt)],
            text_format=RecipeList,
        )

        titles = response.output_parsed.titles
        logging.info(f"Discovered {len(titles)} recipes in book's recipe list")
        return titles
    except Exception as e:
        logging.warning(f"Failed to extract recipe list: {e}")
        return None


# Removed duplicate extraction logic - using AsyncChapterExtractor instead
# Schema validation ensures all returned recipes are complete (min_items constraints)


async def main_async():
    """Main async function for parallel processing."""
    parser = argparse.ArgumentParser(description="Simple chapter-based EPUB extraction (PARALLEL)")
    parser.add_argument("epub_path", type=str, help="Path to EPUB file")
    parser.add_argument("--model", type=str, default="gpt-5-nano", choices=["gpt-5-nano", "gpt-5-mini"])
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--skip-recipe-list", action="store_true")
    parser.add_argument("--no-images", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"File not found: {args.epub_path}")

    setup_logging()

    start_time = time.time()

    # Get metadata
    book_temp = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book_temp.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    logging.info("="*80)
    logging.info(f"Processing: {book_title}")
    logging.info(f"Method: Simple chapter-based extraction")
    logging.info("="*80)

    # PHASE 1: Convert chapters
    logging.info("\nPHASE 1: Converting chapters")
    book, chapters = convert_epub_by_chapters(args.epub_path)

    # PHASE 2: Discover recipe list
    logging.info("\nPHASE 2: Discovering recipe list")
    if args.skip_recipe_list:
        expected_titles = None
        logging.info("Skipping recipe list discovery")
    else:
        expected_titles = discover_recipe_list(chapters)

    # PHASE 3: Extract from all chapters using AsyncChapterExtractor
    logging.info("\nPHASE 3: Extracting from chapters (PARALLEL)")

    # Convert to Chapter objects
    chapter_objs = [
        Chapter(name=name, content=content, index=i)
        for i, (name, content) in enumerate(chapters)
    ]

    # Use the actual chapter extractor with high concurrency
    # With 10,000 RPM rate limit, we can process many chapters in parallel
    extractor = AsyncChapterExtractor(model=args.model)
    extraction_results = await extractor.extract_from_chapters(
        chapters=chapter_objs,
        expected_titles=expected_titles,
        max_concurrent=200  # High concurrency for 10k RPM rate limit
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

    # PHASE 4: Deduplication (schema validation already ensures completeness)
    logging.info("\nPHASE 4: Deduplication")
    seen = set()
    unique_recipes = []

    for recipe in all_recipes:
        if recipe.title not in seen:
            seen.add(recipe.title)
            unique_recipes.append(recipe)
        else:
            logging.debug(f"  Duplicate: {recipe.title}")

    logging.info(f"Total: {len(all_recipes)} → Unique: {len(unique_recipes)}")

    # Validate against expected
    if expected_titles:
        missing = set(expected_titles) - seen
        extra = seen - set(expected_titles)

        if missing:
            logging.warning(f"Missing {len(missing)} recipes")
            for m in list(missing)[:10]:
                logging.warning(f"  - {m}")
        if extra:
            logging.warning(f"Extra {len(extra)} recipes")
            for e in list(extra)[:10]:
                logging.warning(f"  + {e}")

        match_count = len(seen & set(expected_titles))
        logging.info(f"Match rate: {match_count}/{len(expected_titles)} ({match_count/len(expected_titles)*100:.1f}%)")

    # PHASE 5: Write recipes
    logging.info("\nPHASE 5: Writing recipes")
    out_dir = Path(args.output_dir) / f"{book_slug}-simple-chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written = 0

    for recipe in unique_recipes:
        recipe_dict = processor._mela_recipe_to_object(recipe)
        recipe_dict["link"] = book_title
        recipe_dict["id"] = str(uuid.uuid4())
        recipe_dict["images"] = []

        if processor.write_recipe(recipe_dict, output_dir=str(out_dir)):
            written += 1

    # Create archive
    archive_zip = shutil.make_archive(base_name=str(out_dir), format="zip", root_dir=str(out_dir))
    archive_mela = archive_zip.replace(".zip", ".melarecipes")
    os.rename(archive_zip, archive_mela)

    # Summary
    logging.info("\n" + "="*80)
    logging.info("COMPLETE")
    logging.info("="*80)
    logging.info(f"Chapters: {len(chapters)}")
    logging.info(f"Extracted: {len(all_recipes)}")
    logging.info(f"Unique: {len(unique_recipes)}")
    logging.info(f"Written: {written}")
    if expected_titles:
        logging.info(f"Expected: {len(expected_titles)}")
        logging.info(f"Match: {match_count}/{len(expected_titles)}")
    logging.info(f"Time: {time.time()-start_time:.1f}s")
    logging.info(f"Output: {archive_mela}")


def main():
    """Entry point - runs async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
