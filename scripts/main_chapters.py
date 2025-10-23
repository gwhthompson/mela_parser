#!/usr/bin/env python3
"""
Chapter-based extraction - The proper solution.
Uses EPUB's natural chapter structure + recipe list discovery.
No overlap, no duplicates, exact matching.
"""
import argparse
import logging
import os
import re
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel

from mela_parser.parse import CookbookParser, MelaRecipe
from mela_parser.recipe import RecipeProcessor


def setup_logging(log_file: str = "process_chapters.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


class RecipeList(BaseModel):
    titles: List[str]


def convert_epub_by_chapters(epub_path: str) -> Tuple[epub.EpubBook, List[Tuple[str, str]]]:
    """Convert each EPUB chapter to markdown using MarkItDown."""
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


def discover_recipe_list(chapters: List[Tuple[str, str]]) -> Optional[List[str]]:
    """Discover recipe list by finding link patterns anywhere in the book."""
    client = OpenAI()

    # Collect all potential recipe list sections
    all_link_sections = []

    for chapter_name, chapter_md in chapters:
        # Find sections with many markdown links
        link_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links = re.findall(link_pattern, chapter_md)

        if len(links) > 5:  # Looks like a list
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


def extract_from_chapter(
    chapter_md: str,
    chapter_name: str,
    expected_titles: Optional[List[str]] = None,
    model: str = "gpt-5-nano"
) -> List[MelaRecipe]:
    """Extract recipes from a single chapter."""
    client = OpenAI()

    # Build targeted prompt
    if expected_titles:
        # Find which recipes might be in this chapter
        likely_here = [t for t in expected_titles if t.lower() in chapter_md.lower()]

        if not likely_here:
            return []

        expected_list = "\n".join(f"- {title}" for title in likely_here)
        prompt = f"""Extract ONLY these specific recipes from this chapter.
Use the EXACT titles listed below.

Expected recipes:
{expected_list}

<chapter>
{chapter_md}
</chapter>"""
    else:
        prompt = f"""Extract ALL complete recipes from this chapter.
Copy titles EXACTLY as they appear in the text.
Do NOT add commentary or labels.

<chapter>
{chapter_md}
</chapter>"""

    try:
        parser = CookbookParser(model=model)
        # Override to add temperature
        response = client.responses.parse(
            model=model,
            input=[EasyInputMessageParam(role="user", content=prompt)],
            text_format=CookbookParser.__annotations__['parse_cookbook'].__args__[0],
            temperature=0,
        )

        from parse import CookbookRecipes
        response = client.responses.parse(
            model=model,
            input=[EasyInputMessageParam(role="user", content=prompt)],
            text_format=CookbookRecipes,
        )

        return response.output_parsed.recipes
    except Exception as e:
        logging.error(f"Failed to extract from chapter {chapter_name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Chapter-based EPUB cookbook extraction")
    parser.add_argument("epub_path", type=str, help="Path to EPUB file")
    parser.add_argument("--model", type=str, default="gpt-5-nano", choices=["gpt-5-nano", "gpt-5-mini"])
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--skip-recipe-list", action="store_true", help="Skip recipe list discovery")

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"File not found: {args.epub_path}")

    setup_logging()

    start_time = time.time()

    # Get book metadata
    book_temp = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book_temp.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    logging.info("="*80)
    logging.info(f"Processing: {book_title}")
    logging.info(f"Method: Chapter-based extraction")
    logging.info("="*80)

    # PHASE 1: Convert chapters
    logging.info("\nPHASE 1: Converting EPUB chapters to markdown")
    logging.info("="*80)

    book, chapters = convert_epub_by_chapters(args.epub_path)

    # PHASE 2: Discover recipe list
    logging.info("\nPHASE 2: Discovering recipe list")
    logging.info("="*80)

    if args.skip_recipe_list:
        expected_titles = None
        logging.info("Skipping recipe list discovery (--skip-recipe-list)")
    else:
        expected_titles = discover_recipe_list(chapters)

    # PHASE 3: Extract from each chapter
    logging.info("\nPHASE 3: Extracting recipes from chapters")
    logging.info("="*80)

    all_recipes = []

    for i, (chapter_name, chapter_md) in enumerate(chapters, 1):
        logging.info(f"\n[Chapter {i}/{len(chapters)}] {chapter_name}")

        recipes = extract_from_chapter(chapter_md, chapter_name, expected_titles, args.model)

        if recipes:
            logging.info(f"  Extracted {len(recipes)} recipes")
            for r in recipes:
                logging.info(f"    ✓ {r.title}")
            all_recipes.extend(recipes)
        else:
            logging.info(f"  No recipes found")

    extraction_time = time.time() - start_time

    # PHASE 4: Simple deduplication
    logging.info("\nPHASE 4: Deduplication")
    logging.info("="*80)

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
            logging.warning(f"Missing {len(missing)} recipes: {list(missing)[:10]}")
        if extra:
            logging.warning(f"Extra {len(extra)} recipes: {list(extra)[:10]}")

        logging.info(f"Match rate: {len(seen & set(expected_titles))}/{len(expected_titles)} ({len(seen & set(expected_titles))/len(expected_titles)*100:.1f}%)")

    # PHASE 5: Write recipes
    logging.info("\nPHASE 5: Writing recipes")
    logging.info("="*80)

    out_dir = Path(args.output_dir) / f"{book_slug}-chapters"
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
        logging.info(f"Match: {len(seen & set(expected_titles))}/{len(expected_titles)}")
    logging.info(f"Time: {time.time()-start_time:.1f}s")
    logging.info(f"Output: {archive_mela}")


if __name__ == "__main__":
    main()
