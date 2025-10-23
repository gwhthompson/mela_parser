#!/usr/bin/env python3
import base64
import html.parser
import json
import logging
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
from bs4 import BeautifulSoup

import html2text
from ebooklib import epub
from PIL import Image, UnidentifiedImageError

from .parse import MelaRecipe, RecipeParser

# logging.basicConfig(level=logging.INFO)


class RecipeDict(TypedDict, total=False):
    title: str
    recipeYield: str
    prepTime: str
    cookTime: str
    totalTime: str
    ingredients: str
    instructions: str
    categories: List[str]
    link: str
    id: str
    images: List[str]


class RecipeExtractor(html.parser.HTMLParser):
    def __init__(self, fragment_id: str) -> None:
        super().__init__()
        self.fragment_id: str = fragment_id
        self.recording: bool = False
        self.found: bool = False
        self.parts: List[str] = []

    def get_recipe(self) -> Optional[str]:
        return "".join(self.parts) if self.found else None


class RecipeProcessor:
    def __init__(self, epub_path: str, book: Optional[epub.EpubBook] = None) -> None:
        self.epub_path: str = epub_path
        self.book: epub.EpubBook = book or epub.read_epub(
            epub_path, {"ignore_ncx": True}
        )
        self.book_title: str = self.book.get_metadata("DC", "title")[0][0]

    @staticmethod
    def slugify(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"[\s_-]+", "-", s)
        s = re.sub(r"^-+|-+$", "", s)
        return s

    @staticmethod
    def normalize_path(path: str) -> str:
        return path[3:] if path.startswith("../") else path

    def extract_segment(self, html, current_id, next_id=None):
        soup = BeautifulSoup(html, "html.parser")
        start = soup.find(id=current_id)
        if not start:
            return ""
        output = [str(start)]
        for elem in start.find_all_next():
            if next_id and elem.get("id") == next_id:
                break
            output.append(str(elem))
        return "".join(output)

    def get_content_for_navigation_item(
        self, current_navigation_item: str, next_navigation_item: Optional[str] = None
    ) -> str:
        file_path, *curr_frag = current_navigation_item.split("#")
        current_fragment = curr_frag[0] if curr_frag else None

        next_fragment = None
        if next_navigation_item:
            _, *next_frag = next_navigation_item.split("#")
            next_fragment = next_frag[0] if next_frag else None

        item = self.book.get_item_with_href(file_path)
        if not item:
            logging.info(f"No item found for {file_path}")
            return ""
        content = item.get_body_content().decode("utf-8")
        section = self.extract_segment(content, current_fragment, next_fragment)
        logging.info(f"Extracted section for {current_navigation_item}")
        return section

    def extract_recipe(
        self, current_navigation_item: str, next_navigation_item: Optional[str] = None
    ) -> RecipeDict:
        html_content = self.get_content_for_navigation_item(
            current_navigation_item, next_navigation_item
        )
        markdown = (
            html2text.html2text(html_content) if "<" in html_content else html_content
        )
        mela_recipe: MelaRecipe = RecipeParser(markdown).parse()
        recipe_dict: RecipeDict = self._mela_recipe_to_object(mela_recipe)
        recipe_dict["link"] = self.book_title

        image_pattern = r'(?i)(?:src=["\']?)([^"\'<>]+\.(?:png|jpg|jpeg|gif))'
        recipe_dict["images"] = re.findall(image_pattern, html_content)
        self._process_images(recipe_dict)
        return recipe_dict

    def _mela_recipe_to_object(self, mela_recipe: MelaRecipe) -> RecipeDict:
        raw: Dict[str, Any] = mela_recipe.model_dump(by_alias=True)
        res: RecipeDict = {k: (v if v is not None else "") for k, v in raw.items()}  # type: ignore

        def format_minutes(minutes: Union[int, str]) -> str:
            try:
                minutes = int(minutes)
            except (ValueError, TypeError):
                minutes = 0
            if minutes <= 0:
                return ""
            hours, mins = divmod(minutes, 60)
            if hours > 0:
                return f"{hours} hr {mins} min" if mins > 0 else f"{hours} hr"
            return f"{mins} min"

        res["prepTime"] = format_minutes(res.get("prepTime", 0))
        res["cookTime"] = format_minutes(res.get("cookTime", 0))
        res["totalTime"] = format_minutes(res.get("totalTime", 0))

        if len(mela_recipe.ingredients) == 1:
            ingredients_str = "\n".join(mela_recipe.ingredients[0].ingredients)
        else:
            ingredients_str = "\n".join(
                f"# {grp.title}\n" + "\n".join(grp.ingredients)
                for grp in mela_recipe.ingredients
            )
        res["ingredients"] = ingredients_str
        res["instructions"] = "\n".join(mela_recipe.instructions)
        res["categories"] = (
            [c.value for c in mela_recipe.categories] if mela_recipe.categories else []
        )
        # Link will be set to navigation_item in extract_recipe
        res["id"] = str(uuid.uuid4())
        return res

    def _process_images(self, recipe_dict: RecipeDict) -> None:
        images: List[str] = recipe_dict.get("images", [])
        best_image: Optional[bytes] = None
        best_area: int = 0
        best_size: int = 0

        for img_path in images:
            normalized = self.normalize_path(img_path)
            item = self.book.get_item_with_href(normalized)
            if not item:
                continue
            try:
                content: bytes = item.get_content()
                with Image.open(BytesIO(content)) as img:
                    area = img.width * img.height
            except (OSError, UnidentifiedImageError):
                continue
            if area >= 300000 and (
                area > best_area or (area == best_area and len(content) > best_size)
            ):
                best_image = content
                best_area = area
                best_size = len(content)
        if best_image:
            recipe_dict["images"] = [base64.b64encode(best_image).decode("utf-8")]
        else:
            recipe_dict["images"] = []

    def write_recipe(
        self, recipe_dict: RecipeDict, output_dir: Optional[str] = None
    ) -> str:
        if (
            not recipe_dict.get("title", "").strip()
            or not recipe_dict.get("ingredients", "").strip()
            or not recipe_dict.get("instructions", "").strip()
        ):
            print(
                f"Skipping save; incomplete recipe: {recipe_dict.get('title', 'UNKNOWN')}"
            )
            return ""
        # Extract a unique ID from the link to ensure filename uniqueness
        filename: str = f"{self.slugify(recipe_dict.get('title', 'recipe'))}.melarecipe"
        out_dir: Path = Path(output_dir) if output_dir else Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath: Path = out_dir / filename
        # logging.debug(f"Target file path for recipe: {filepath}")
        if filepath.exists():
            logging.info(f"File exists: {filepath}. Skipping.")
            return ""
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(recipe_dict, f, ensure_ascii=False)
            f.flush()  # Ensure data is written to disk
        logging.info(f"Recipe written to: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    proc = RecipeProcessor("example.epub")
    recipe = proc.extract_recipe("chapter1.html#fragment")
    print(proc.write_recipe(recipe))
