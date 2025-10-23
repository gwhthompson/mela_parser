import os
import shutil
import argparse
import logging
from ebooklib import epub

from epub import RecipesList
from mela_parser.recipe import RecipeProcessor

logging.basicConfig(
    filename="process.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def canonical_link(link: str) -> str:
    # Split the link at '#' (if present) and normalize the parts
    parts = link.split("#", 1)
    path = parts[0].strip().lower()
    fragment = parts[1].strip().lower() if len(parts) > 1 else ""
    return f"{path}#{fragment}" if fragment else path


def main():
    parser = argparse.ArgumentParser(
        description="Process an EPUB file containing recipes."
    )
    parser.add_argument(
        "epub_path", type=str, help="Path to the EPUB file containing the recipes"
    )
    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"The file at {args.epub_path} does not exist.")

    book = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    recipes_list = RecipesList(args.epub_path, book)
    parsed_recipes = recipes_list.parse()  # Only call once.
    processor = RecipeProcessor(args.epub_path, book)

    # Extract the full list of TOC links from the book (using the helper from RecipesList)
    def traverse_toc(entries, links):
        for entry in entries:
            if isinstance(entry, dict) and "href" in entry:
                links.append(entry["href"])
            if isinstance(entry, dict) and "children" in entry:
                traverse_toc(entry["children"], links)

    toc_entries = RecipesList.get_toc_entries(book.toc)
    all_links = []
    traverse_toc(toc_entries, all_links)
    logging.info(f"Full TOC links: {all_links}")

    book_slug = processor.slugify(processor.book_title)
    out_dir = os.path.join("output", book_slug)
    os.makedirs(out_dir, exist_ok=True)

    # Build a deduplicated dictionary keyed by a canonical link
    unique_recipes = {}
    for recipe in parsed_recipes.recipes:
        key = canonical_link(recipe.link)
        if key not in unique_recipes:
            unique_recipes[key] = recipe

    for recipe in unique_recipes.values():
        key = canonical_link(recipe.link)
        try:
            # Determine next link from full TOC links
            try:
                idx = next(
                    i for i, link in enumerate(all_links) if canonical_link(link) == key
                )
            except StopIteration:
                idx = None
            next_link = (
                all_links[idx + 1]
                if idx is not None and idx + 1 < len(all_links)
                else None
            )

            # Build filename using only the slugified recipe title.
            title_part = processor.slugify(recipe.title)
            filename = f"{title_part}.melarecipe"
            filepath = os.path.join(out_dir, filename)

            if os.path.exists(filepath):
                logging.info(f"{key}: skipped (file exists)")
                continue

            recipe_dict = processor.extract_recipe(recipe.link, next_link)
            # Override extracted title with one from list if needed.
            recipe_dict["title"] = recipe.title

            output_path = processor.write_recipe(recipe_dict, output_dir=out_dir)
            if output_path:
                logging.info(f"{key}: success")
            else:
                logging.info(f"{key}: fail")
        except Exception as e:
            logging.error(f"Error while extracting recipe for {key}: {e}")
            continue

    archive_zip = shutil.make_archive(base_name=out_dir, format="zip", root_dir=out_dir)
    archive_mela = archive_zip.replace(".zip", ".melarecipes")
    os.rename(archive_zip, archive_mela)
    print(f"Book recipes archive created: {archive_mela}")


if __name__ == "__main__":
    main()
