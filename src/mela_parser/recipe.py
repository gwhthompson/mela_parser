#!/usr/bin/env python3
"""Recipe extraction and processing for Mela format.

This module handles the extraction of recipe data from EPUB cookbooks and
converts it to the Mela recipe manager format. It includes functionality for:
- Extracting recipes from HTML/markdown content
- Processing and optimizing recipe images
- Converting recipe data to Mela-compatible JSON
- Writing recipe files to disk

The RecipeProcessor class is the main interface for working with recipes from
EPUB files, handling both metadata extraction and image processing.
"""

import base64
import html.parser
import json
import logging
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import html2text
from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image, UnidentifiedImageError

from .parse import MelaRecipe, RecipeParser


class RecipeDict(TypedDict, total=False):
    """TypedDict for Mela recipe JSON format.

    This TypedDict defines the structure of a recipe as it's stored in
    Mela's .melarecipe JSON files. All fields are optional to support
    partial recipe data.
    """

    title: str
    recipeYield: str
    prepTime: str
    cookTime: str
    totalTime: str
    ingredients: str
    instructions: str
    categories: list[str]
    link: str
    id: str
    images: list[str]


class RecipeExtractor(html.parser.HTMLParser):
    """HTML parser for extracting recipe fragments from EPUB content.

    This parser can extract specific HTML fragments identified by their
    fragment ID, which is useful for extracting individual recipes from
    multi-recipe HTML documents.
    """

    def __init__(self, fragment_id: str) -> None:
        """Initialize the recipe extractor.

        Args:
            fragment_id: HTML fragment ID to extract (e.g., "recipe-1")
        """
        super().__init__()
        self.fragment_id: str = fragment_id
        self.recording: bool = False
        self.found: bool = False
        self.parts: list[str] = []

    def get_recipe(self) -> str | None:
        """Get the extracted recipe HTML.

        Returns:
            Concatenated HTML string if recipe was found, None otherwise
        """
        return "".join(self.parts) if self.found else None


class RecipeProcessor:
    """Process and extract recipes from EPUB cookbooks to Mela format.

    This class handles the complete workflow of recipe extraction including:
    - Reading EPUB files and extracting content
    - Parsing HTML/markdown recipe data
    - Processing and optimizing images
    - Converting to Mela JSON format
    - Writing recipe files to disk

    The processor maintains a reference to the EPUB book for accessing images
    and metadata throughout the extraction process.
    """

    def __init__(self, epub_path: str, book: epub.EpubBook | None = None) -> None:
        """Initialize the recipe processor.

        Args:
            epub_path: Path to the EPUB file to process
            book: Optional pre-loaded EpubBook object. If None, will load from epub_path.
        """
        self.epub_path: str = epub_path
        self.book: epub.EpubBook = book or epub.read_epub(epub_path, {"ignore_ncx": True})
        self.book_title: str = self.book.get_metadata("DC", "title")[0][0]

    @staticmethod
    def slugify(s: str) -> str:
        """Convert a string to a URL-friendly slug.

        Converts to lowercase, removes non-alphanumeric characters (except
        spaces and hyphens), and normalizes whitespace to single hyphens.

        Args:
            s: String to slugify

        Returns:
            Slugified string suitable for filenames or URLs

        Examples:
            >>> RecipeProcessor.slugify("Hello World!")
            'hello-world'
            >>> RecipeProcessor.slugify("  Foo & Bar  ")
            'foo-bar'
        """
        s = s.lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"[\s_-]+", "-", s)
        s = re.sub(r"^-+|-+$", "", s)
        return s

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize an image path by removing relative path prefixes.

        Args:
            path: Image path, potentially with ../ prefix

        Returns:
            Normalized path without ../ prefix

        Examples:
            >>> RecipeProcessor.normalize_path("../images/foo.jpg")
            'images/foo.jpg'
            >>> RecipeProcessor.normalize_path("images/foo.jpg")
            'images/foo.jpg'
        """
        return path[3:] if path.startswith("../") else path

    def extract_segment(self, html: str, current_id: str | None, next_id: str | None = None) -> str:
        """Extract a segment of HTML between two fragment IDs.

        Useful for extracting content between two navigation points in an
        EPUB, such as extracting a single recipe that spans from one ID to
        another.

        Args:
            html: HTML content to extract from
            current_id: Starting fragment ID (None means extract from beginning)
            next_id: Optional ending fragment ID. If None, extracts to end.

        Returns:
            Extracted HTML segment as a string, or empty string if current_id not found
        """
        soup = BeautifulSoup(html, "html.parser")

        # If current_id is None, return entire html
        if current_id is None:
            return html

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
        self, current_navigation_item: str, next_navigation_item: str | None = None
    ) -> str:
        """Get content for a navigation item (e.g., chapter or section).

        Extracts content between two navigation points in the EPUB, handling
        both file-level and fragment-level navigation.

        Args:
            current_navigation_item: Navigation href (e.g., "chapter1.html#recipe-1")
            next_navigation_item: Optional next navigation href for boundary detection

        Returns:
            Extracted HTML content as a string

        Raises:
            None, but logs warning if navigation item not found
        """
        file_path, *curr_frag = current_navigation_item.split("#")
        current_fragment = curr_frag[0] if curr_frag else None

        next_fragment: str | None = None
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
        self, current_navigation_item: str, next_navigation_item: str | None = None
    ) -> RecipeDict:
        """Extract a recipe from a navigation item in the EPUB.

        Extracts the HTML content for the navigation item, converts to markdown,
        parses the recipe using the RecipeParser, and processes any associated
        images.

        Args:
            current_navigation_item: Navigation href for the recipe
            next_navigation_item: Optional next navigation href for boundary detection

        Returns:
            RecipeDict with extracted recipe data including processed images
        """
        html_content = self.get_content_for_navigation_item(
            current_navigation_item, next_navigation_item
        )
        markdown = html2text.html2text(html_content) if "<" in html_content else html_content
        mela_recipe: MelaRecipe = RecipeParser(markdown).parse()
        recipe_dict: RecipeDict = self._mela_recipe_to_object(mela_recipe)
        recipe_dict["link"] = self.book_title

        image_pattern = r'(?i)(?:src=["\']?)([^"\'<>]+\.(?:png|jpg|jpeg|gif))'
        recipe_dict["images"] = re.findall(image_pattern, html_content)
        self._process_images(recipe_dict)
        return recipe_dict

    def _mela_recipe_to_object(self, mela_recipe: MelaRecipe) -> RecipeDict:
        """Convert a MelaRecipe Pydantic model to a Mela-format dictionary.

        Transforms the structured MelaRecipe object into the dictionary format
        expected by Mela, including:
        - Converting time integers to formatted strings (e.g., 60 -> "1 hr")
        - Flattening ingredient groups into newline-separated string
        - Joining instruction steps with newlines
        - Extracting category values from enum

        Args:
            mela_recipe: Parsed MelaRecipe object from the LLM

        Returns:
            RecipeDict formatted for Mela JSON output
        """
        raw: dict[str, Any] = mela_recipe.model_dump(by_alias=True)
        res: RecipeDict = {k: (v if v is not None else "") for k, v in raw.items()}  # type: ignore

        def format_minutes(minutes: int | str) -> str:
            """Format minutes as human-readable time string.

            Args:
                minutes: Time in minutes (int or str)

            Returns:
                Formatted time string (e.g., "1 hr 30 min") or empty string if invalid
            """
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
                f"# {grp.title}\n" + "\n".join(grp.ingredients) for grp in mela_recipe.ingredients
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
        """Process and select the best image for a recipe.

        Examines all candidate images for a recipe, selects the largest one
        that meets minimum size requirements (300,000 pixels), optimizes it
        for mobile viewing (max 600px wide), and encodes it as base64.

        Modifies the recipe_dict in-place to set the 'images' field to either
        a single-element list with the base64-encoded image, or an empty list.

        Args:
            recipe_dict: Recipe dictionary to process (modified in-place)
        """
        images: list[str] = recipe_dict.get("images", [])
        best_image: Image.Image | None = None
        best_area: int = 0

        for img_path in images:
            normalized = self.normalize_path(img_path)
            item = self.book.get_item_with_href(normalized)
            if not item:
                continue
            try:
                content: bytes = item.get_content()
                img = Image.open(BytesIO(content))
                area = img.width * img.height
            except (OSError, UnidentifiedImageError):
                continue
            if area >= 300000 and area > best_area:
                if best_image:
                    best_image.close()
                best_image = img
                best_area = area
            else:
                img.close()

        if best_image:
            # Optimize image: max 600px wide, 144 DPI
            optimized = self._optimize_image(best_image)
            best_image.close()

            # Encode optimized image to base64
            buffer = BytesIO()
            optimized.save(buffer, format="JPEG", quality=85, dpi=(144, 144))
            optimized.close()
            recipe_dict["images"] = [base64.b64encode(buffer.getvalue()).decode("utf-8")]
        else:
            recipe_dict["images"] = []

    def _optimize_image(self, img: Image.Image, max_width: int = 600) -> Image.Image:
        """Optimize image by resizing to max width and converting to RGB.

        Prepares images for mobile viewing by:
        - Converting to RGB (handles RGBA, P, etc.)
        - Resizing to max width while preserving aspect ratio
        - Using high-quality LANCZOS resampling

        Args:
            img: PIL Image to optimize
            max_width: Maximum width in pixels. Defaults to 600.

        Returns:
            Optimized PIL Image ready for JPEG encoding
        """
        # Convert to RGB if needed (handles RGBA, P, etc.)
        if img.mode != "RGB":
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            else:
                rgb_img.paste(img)
            img = rgb_img

        # Resize if wider than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        return img

    def write_recipe(self, recipe_dict: RecipeDict, output_dir: str | None = None) -> str:
        """Write a recipe to disk as a .melarecipe JSON file.

        Validates the recipe has required fields (title, ingredients, instructions)
        before writing. Skips writing if a file with the same name already exists.

        Args:
            recipe_dict: Recipe data to write
            output_dir: Optional output directory. Defaults to "output".

        Returns:
            Path to the written file as a string, or empty string if skipped

        Notes:
            - Skips recipes missing required fields
            - Skips if file already exists
            - Creates output directory if it doesn't exist
        """
        if (
            not recipe_dict.get("title", "").strip()
            or not recipe_dict.get("ingredients", "").strip()
            or not recipe_dict.get("instructions", "").strip()
        ):
            print(f"Skipping save; incomplete recipe: {recipe_dict.get('title', 'UNKNOWN')}")
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
