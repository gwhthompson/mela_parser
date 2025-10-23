#!/usr/bin/env python3
"""Debug script to inspect the 6 recipes that failed to save."""

import asyncio
import logging
from io import BytesIO

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam

from mela_parser.parse import CookbookRecipes

logging.basicConfig(level=logging.INFO)

# The 6 failed recipe titles
FAILED_TITLES = [
    "Seeded Granola and Chai-spiced Poached Plums",
    "Potato, Celeriac, Onion Seed and Thyme Rostis with HP Gravy",
    "Sweet Potato Fritters with Avocado and Onion Jam",
    "Apple, Fennel and Fig Bircher Muesli",
    "Caramelised Banana French Toast, Maple and Smoked Sea Salt",
    "Mushroom, Spinach and Truffle Toast",
]


async def extract_from_chapter(chapter_md: str, chapter_name: str):
    """Extract recipes from a single chapter."""
    client = AsyncOpenAI()

    prompt = f"""Extract all complete recipes from this section.

The schema requires:
- At least 1 ingredient with measurements
- At least 2 instruction steps

This naturally filters out incomplete content. Extract anything that meets these requirements.

Copy titles exactly. Preserve ingredient groupings. Leave time/yield blank if not stated.

<section>
{chapter_md}
</section>"""

    response = await client.responses.parse(
        model="gpt-5-nano",
        input=[EasyInputMessageParam(role="user", content=prompt)],
        text_format=CookbookRecipes,
    )
    return response.output_parsed.recipes


async def main():
    """Find and inspect the chapters containing failed recipes."""
    epub_path = "examples/input/planted.epub"

    book = epub.read_epub(epub_path, {"ignore_ncx": True})
    md = MarkItDown()

    print("Searching for chapters containing failed recipes...\n")

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content()
        result = md.convert_stream(BytesIO(html_content), file_extension=".html")
        markdown = result.text_content

        # Check if any failed title appears in this chapter
        found_titles = [title for title in FAILED_TITLES if title in markdown]

        if found_titles:
            print(f"=" * 80)
            print(f"Chapter: {item.get_name()}")
            print(f"Found: {', '.join(found_titles)}")
            print("=" * 80)

            # Extract recipes
            recipes = await extract_from_chapter(markdown, item.get_name())

            for recipe in recipes:
                if recipe.title in FAILED_TITLES:
                    print(f"\n📋 Recipe: {recipe.title}")
                    print(f"   Ingredient groups: {len(recipe.ingredients)}")
                    for i, grp in enumerate(recipe.ingredients):
                        print(f"   Group {i+1}: '{grp.title}' - {len(grp.ingredients)} items")
                        if not grp.ingredients:
                            print(f"   ⚠️  EMPTY INGREDIENT GROUP!")
                        else:
                            print(f"      First item: {grp.ingredients[0][:50]}")

                    print(f"   Instructions: {len(recipe.instructions)} steps")
                    if not recipe.instructions:
                        print(f"   ⚠️  NO INSTRUCTIONS!")
                    else:
                        print(f"      First step: {recipe.instructions[0][:80]}")
                    print()


if __name__ == "__main__":
    asyncio.run(main())
