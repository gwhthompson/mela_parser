from typing import List
import logging

from ebooklib import epub
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel


class Recipe(BaseModel):
    title: str
    link: str


class Recipes(BaseModel):
    recipes: List[Recipe]


class RecipesList:
    def __init__(self, epub_path: str, book=None):
        self.epub_path = epub_path
        self.book = (
            book
            if book is not None
            else epub.read_epub(epub_path, {"ignore_ncx": True})
        )
        self.client = OpenAI()
        # Use the standardized slugify from RecipeProcessor
        from recipe import RecipeProcessor

        self.slugify = RecipeProcessor.slugify

    @staticmethod
    def get_toc_entries(toc):
        entries = []
        for entry in toc:
            if hasattr(entry, "title"):
                entries.append({"title": entry.title, "href": entry.href})
            elif isinstance(entry, tuple) and len(entry) == 2:
                link, children = entry
                entries.append(
                    {
                        "title": link.title,
                        "href": link.href,
                        "children": RecipesList.get_toc_entries(children),
                    }
                )
            elif isinstance(entry, list):
                entries.extend(RecipesList.get_toc_entries(entry))
            else:
                entries.append({"entry": str(entry)})
        return entries

    def parse(self):
        navigation = self.get_toc_entries(self.book.toc)
        prompt = (
            'Extract recipe items from this navigation list (preserving order) by filtering out non-recipe entries like "Contents", "About the Book", etc. '
            "Return each unique dish only once based on its title."
            "<navigation_items>\n{{NAVIGATION_ITEMS}}\n</navigation_items>\n\n"
        ).replace("{{NAVIGATION_ITEMS}}", str(navigation))
        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    EasyInputMessageParam(role="user", content=prompt),
                ],
                text_format=Recipes,
            )
        except Exception as e:
            logging.error(f"Error during OpenAI parsing: {e}")
            return Recipes(recipes=[])

        if not hasattr(response, "output_parsed"):
            logging.error("Response missing 'output_parsed'; skipping recipe parsing.")
            return Recipes(recipes=[])

        # Import canonical_link from main for consistent deduplication
        from main import canonical_link

        # Enhanced deduplication: use canonical link as key
        unique = {}
        for recipe in response.output_parsed.recipes:
            key = canonical_link(recipe.link)
            if key not in unique:
                unique[key] = recipe
        return Recipes(recipes=list(unique.values()))
