"""
List-based recipe extraction pipeline.

Uses StructuredListExtractor to get recipe list, then extracts each recipe
individually through MarkItDown → OpenAI pipeline with async parallel processing.
"""
import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from ebooklib import epub
from markitdown import MarkItDown
from openai import AsyncOpenAI

from .structured_list import StructuredListExtractor, RecipeLink
from ..parse import RecipeParser, MelaRecipe
from ..recipe import RecipeProcessor

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extracting a single recipe"""
    recipe_link: RecipeLink
    success: bool
    recipe_dict: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Results from processing a batch of recipes"""
    total: int
    successful: int
    failed: int
    results: List[ExtractionResult]


class ListBasedRecipeExtractor:
    """
    Extract recipes using list-based approach.

    Pipeline:
    1. Use StructuredListExtractor to get validated recipe list
    2. For each recipe (identified by href+fragment):
       - Extract HTML content from EPUB
       - Convert HTML → Markdown via MarkItDown
       - Parse Markdown → structured data via OpenAI
       - Extract and process images
    3. Process recipes in async parallel batches
    4. Save as .melarecipe files
    """

    def __init__(self, epub_path: str):
        """
        Initialize extractor.

        Args:
            epub_path: Path to EPUB file
        """
        self.epub_path = epub_path
        self.book = epub.read_epub(epub_path)
        self.list_extractor = StructuredListExtractor()
        self.recipe_processor = RecipeProcessor(epub_path, self.book)
        self.markdown_converter = MarkItDown()
        self.openai_client = AsyncOpenAI()

    def get_recipe_list(self) -> List[RecipeLink]:
        """
        Get validated recipe list using StructuredListExtractor.

        Returns:
            List of RecipeLink objects with href+fragment identifiers
        """
        # Find recipe list pages
        list_pages = self.list_extractor.find_recipe_list_pages(self.book)

        if not list_pages:
            logger.warning("No high-link-density pages found, checking nav/toc...")
            import ebooklib
            for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                if any(kw in item.file_name.lower() for kw in ['nav', 'toc', 'contents']):
                    list_pages = [item]
                    break

        if not list_pages:
            logger.error("No recipe list pages found in EPUB")
            return []

        # Extract links
        all_links = []
        for page in list_pages:
            all_links.extend(self.list_extractor.extract_links_from_page(page))

        # Apply filters and LLM validation (pass book for proximity dedup)
        filtered = self.list_extractor.apply_structural_filters(all_links, self.book)
        validated = self.list_extractor.validate_with_llm(filtered.candidates)

        logger.info(f"Found {len(validated.recipes)} recipes in TOC")
        return validated.recipes

    def get_page_content(self, href: str) -> str:
        """
        Extract HTML content for a given href.

        Handles relative paths by trying common prefixes.

        Args:
            href: File path within EPUB (may be relative)

        Returns:
            HTML content as string
        """
        # Try direct lookup first
        item = self.book.get_item_with_href(href)

        # If not found, try with common path prefixes
        if not item:
            for prefix in ['pages/', 'text/', 'xhtml/', 'OEBPS/', 'content/']:
                item = self.book.get_item_with_href(f'{prefix}{href}')
                if item:
                    break

        if not item:
            raise ValueError(f"No item found for href: {href} (tried with common prefixes)")

        return item.get_body_content().decode('utf-8', errors='ignore')

    def extract_recipe_html(self, recipe_link: RecipeLink, next_recipe: Optional[RecipeLink] = None) -> str:
        """
        Extract HTML content for a recipe using boundary detection.

        Uses fragment IDs as boundaries if available, otherwise extracts full page.

        Args:
            recipe_link: Current recipe to extract
            next_recipe: Next recipe in list (for boundary detection)

        Returns:
            HTML content for the recipe
        """
        # If recipe has fragment, use existing segment extraction
        if recipe_link.fragment:
            # Build navigation item string
            nav_item = f"{recipe_link.href}#{recipe_link.fragment}"

            # If next recipe is on same page with fragment, use it as boundary
            next_nav = None
            if next_recipe and next_recipe.href == recipe_link.href and next_recipe.fragment:
                next_nav = f"{next_recipe.href}#{next_recipe.fragment}"

            return self.recipe_processor.get_content_for_navigation_item(nav_item, next_nav)
        else:
            # No fragment - extract full page
            return self.get_page_content(recipe_link.href)

    def html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML to Markdown using MarkItDown.

        Args:
            html_content: Raw HTML string

        Returns:
            Markdown content
        """
        # MarkItDown needs a file-like object or path, not raw string
        # Convert HTML string to BytesIO
        html_bytes = html_content.encode('utf-8')
        html_stream = BytesIO(html_bytes)

        # Use stream_info to tell MarkItDown this is HTML
        from markitdown._stream_info import StreamInfo
        stream_info = StreamInfo(extension=".html", mimetype="text/html")

        result = self.markdown_converter.convert(html_stream, stream_info=stream_info)
        return result.text_content

    async def extract_single_recipe(
        self,
        recipe_link: RecipeLink,
        next_recipe: Optional[RecipeLink] = None
    ) -> ExtractionResult:
        """
        Extract a single recipe through the full pipeline.

        Args:
            recipe_link: Recipe to extract
            next_recipe: Next recipe (for boundary detection)

        Returns:
            ExtractionResult with success status and data
        """
        try:
            # Step 1: Extract HTML
            html_content = self.extract_recipe_html(recipe_link, next_recipe)

            # Step 2: Convert to Markdown
            markdown = self.html_to_markdown(html_content)

            # Step 3: Parse with OpenAI
            mela_recipe: MelaRecipe = RecipeParser(markdown).parse()

            if not mela_recipe:
                raise ValueError("RecipeParser returned None - OpenAI parsing failed")

            # Step 4: Convert to dict and extract images
            recipe_dict = self.recipe_processor._mela_recipe_to_object(mela_recipe)
            recipe_dict["link"] = recipe_link.href

            # Extract images from HTML
            import re
            image_pattern = r'(?i)(?:src=["\']?)([^"\'<>]+\.(?:png|jpg|jpeg|gif))'
            recipe_dict["images"] = re.findall(image_pattern, html_content)
            self.recipe_processor._process_images(recipe_dict)

            logger.info(f"Successfully extracted: {recipe_link.title}")

            return ExtractionResult(
                recipe_link=recipe_link,
                success=True,
                recipe_dict=recipe_dict
            )

        except Exception as e:
            logger.error(f"Failed to extract {recipe_link.title}: {e}")
            return ExtractionResult(
                recipe_link=recipe_link,
                success=False,
                error=str(e)
            )

    async def process_batch(
        self,
        recipes: List[RecipeLink],
        batch_size: int = 10
    ) -> BatchResult:
        """
        Process recipes in async parallel batches.

        Args:
            recipes: List of recipes to process
            batch_size: Number of recipes to process concurrently

        Returns:
            BatchResult with statistics and results
        """
        all_results = []

        # Process in batches
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: recipes {i+1}-{min(i+batch_size, len(recipes))}")

            # Create tasks for this batch
            tasks = []
            for j, recipe in enumerate(batch):
                next_recipe = recipes[i+j+1] if i+j+1 < len(recipes) else None
                tasks.append(self.extract_single_recipe(recipe, next_recipe))

            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            all_results.extend(batch_results)

        # Calculate statistics
        successful = sum(1 for r in all_results if r.success)
        failed = len(all_results) - successful

        return BatchResult(
            total=len(all_results),
            successful=successful,
            failed=failed,
            results=all_results
        )

    async def extract_all_recipes(
        self,
        output_dir: Optional[str] = None,
        batch_size: int = 10
    ) -> BatchResult:
        """
        Extract all recipes from the cookbook.

        Args:
            output_dir: Directory to save .melarecipe files
            batch_size: Number of recipes to process concurrently

        Returns:
            BatchResult with extraction statistics
        """
        # Step 1: Get recipe list
        logger.info("Step 1: Extracting recipe list from TOC...")
        recipe_list = self.get_recipe_list()
        logger.info(f"Found {len(recipe_list)} recipes in TOC")

        if not recipe_list:
            logger.error("No recipes found in TOC")
            return BatchResult(total=0, successful=0, failed=0, results=[])

        # Step 2: Process recipes in parallel
        logger.info(f"Step 2: Extracting {len(recipe_list)} recipes (batch_size={batch_size})...")
        results = await self.process_batch(recipe_list, batch_size)

        # Step 3: Save successful recipes
        if output_dir:
            logger.info(f"Step 3: Saving {results.successful} recipes to {output_dir}...")
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            for result in results.results:
                if result.success and result.recipe_dict:
                    filepath = self.recipe_processor.write_recipe(result.recipe_dict, output_dir)
                    if filepath:
                        logger.debug(f"Saved: {filepath}")

        logger.info(f"Extraction complete: {results.successful}/{results.total} successful")
        return results


async def extract_cookbook(epub_path: str, output_dir: str = "output", batch_size: int = 10) -> BatchResult:
    """
    Convenience function to extract all recipes from a cookbook.

    Args:
        epub_path: Path to EPUB file
        output_dir: Where to save .melarecipe files
        batch_size: Number of recipes to process concurrently

    Returns:
        BatchResult with statistics
    """
    extractor = ListBasedRecipeExtractor(epub_path)
    return await extractor.extract_all_recipes(output_dir, batch_size)
