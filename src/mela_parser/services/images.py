"""Unified image processing service.

This module provides a unified image processing service that consolidates
all image handling logic with a strategy pattern for selection.

Strategies:
- LARGEST: Select image with largest area (heuristic)
- AI_VERIFIED: Use AI vision to verify image matches recipe

Example:
    >>> from mela_parser.services.images import ImageService, ImageConfig
    >>> config = ImageConfig(min_area=300000, max_width=600)
    >>> service = ImageService(book=epub_book, config=config)
    >>> image_data = service.select_best_image(recipe)
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from openai import OpenAIError
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from ebooklib.epub import EpubBook
    from openai import AsyncOpenAI

    from ..parse import MelaRecipe

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Image selection strategy."""

    LARGEST = "largest"  # Select by size (heuristic)
    AI_VERIFIED = "ai_verified"  # Use AI vision verification


@dataclass
class ImageConfig:
    """Configuration for image processing.

    Attributes:
        min_area: Minimum image area in pixels (filter small images)
        max_width: Maximum width for optimized images
        quality: JPEG quality for optimized images (1-100)
        strategy: Selection strategy to use
        ai_threshold: Confidence threshold for AI verification (0-1)
    """

    min_area: int = 100_000  # ~316x316 pixels (allows typical cookbook images)
    max_width: int = 600
    quality: int = 85
    strategy: SelectionStrategy = SelectionStrategy.LARGEST
    ai_threshold: float = 0.6


@dataclass
class ImageCandidate:
    """A candidate image for selection."""

    path: str
    data: bytes
    width: int
    height: int
    area: int
    ai_confidence: float = 0.0
    ai_matches: bool = False


class ImageService:
    """Unified image processing with strategy pattern.

    This service handles all image-related operations:
    - Extracting image paths from markdown
    - Loading images from EPUB files
    - Selecting the best image using configured strategy
    - Optimizing images for output

    Attributes:
        book: EPUB book containing images
        config: Image processing configuration
        client: Optional OpenAI client for AI verification

    Example:
        >>> service = ImageService(book=epub_book, config=ImageConfig())
        >>> image_data = service.select_best_image(recipe)
    """

    def __init__(
        self,
        book: EpubBook,
        config: ImageConfig,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the image service.

        Args:
            book: EPUB book containing images
            config: Image processing configuration
            client: Optional OpenAI client for AI verification
        """
        self.book = book
        self.config = config
        self.client = client

    async def select_best_image(self, recipe: MelaRecipe) -> str | None:
        """Select the best image for a recipe.

        Uses the configured strategy to select from candidate images
        found in the recipe's markdown content.

        Args:
            recipe: Recipe to find image for

        Returns:
            Base64-encoded image data, or None if no suitable image found
        """
        # Extract image paths from recipe content
        image_paths = self._extract_image_paths(recipe)
        if not image_paths:
            logger.debug(f"No image paths found for '{recipe.title}'")
            return None

        # Load candidate images
        candidates = self._load_candidates(image_paths)
        if not candidates:
            logger.debug(f"No valid images found for '{recipe.title}'")
            return None

        # Select based on strategy
        if self.config.strategy == SelectionStrategy.AI_VERIFIED and self.client is not None:
            return await self._select_with_ai(candidates, recipe)
        return self._select_by_size(candidates, recipe)

    def _extract_image_paths(self, recipe: MelaRecipe) -> list[str]:
        """Extract image paths from recipe.

        Checks recipe.images first (set by LLM), then falls back to
        searching instructions for markdown image syntax.

        Args:
            recipe: Recipe containing image references

        Returns:
            List of unique image paths
        """
        paths: list[str] = []

        # First, check if recipe already has image paths from LLM
        if recipe.images:
            for img_ref in recipe.images:
                # Handle EPUB internal references like "c02_split_007.xhtml#image_page_54_1"
                # Extract image ID from anchor
                if "#" in img_ref and "image_" in img_ref:
                    # This is an EPUB reference - we'll resolve it via the EPUB's image items
                    paths.append(img_ref)
                # Handle direct image paths like "../Images/photo.jpg"
                elif img_ref.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    paths.append(img_ref)

        # Fall back to searching instructions if no images found
        if not paths:
            markdown = "\n".join(recipe.instructions)

            # MarkItDown creates: ![alt](../images/filename.jpg)
            pattern = r"!\[.*?\]\(([^)]+\.(?:jpg|jpeg|png|gif))\)"
            paths = re.findall(pattern, markdown, re.IGNORECASE)

            # Also try simpler pattern for src attributes
            simple_pattern = r'(?:src=["\']?)([^"\'<>]+\.(?:png|jpg|jpeg|gif))'
            paths.extend(re.findall(simple_pattern, markdown, re.IGNORECASE))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_paths: list[str] = []
        for path in paths:
            normalized = path.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(path)

        return unique_paths

    def _load_candidates(self, paths: list[str]) -> list[ImageCandidate]:
        """Load candidate images from EPUB.

        Args:
            paths: List of image paths to load

        Returns:
            List of valid image candidates meeting minimum size
        """
        candidates: list[ImageCandidate] = []

        for path in paths:
            content = self._resolve_image_path(path)
            if content is None:
                continue

            try:
                with Image.open(BytesIO(content)) as img:
                    width, height = img.size
                    area = width * height

                    # Filter by minimum area
                    if area < self.config.min_area:
                        logger.debug(f"Image too small: {path} ({area} < {self.config.min_area})")
                        continue

                    candidates.append(
                        ImageCandidate(
                            path=path,
                            data=content,
                            width=width,
                            height=height,
                            area=area,
                        )
                    )

            except (OSError, UnidentifiedImageError) as e:
                logger.debug(f"Failed to load image {path}: {e}")
                continue

        return candidates

    def _resolve_image_path(self, path: str) -> bytes | None:
        """Resolve an image path to its binary content.

        Handles both direct image paths and EPUB internal references.

        Args:
            path: Image path or EPUB reference

        Returns:
            Image bytes, or None if not found
        """
        # Handle EPUB internal references like "c02_split_007.xhtml#image_page_54_1"
        if "#" in path and ("image_" in path or "page_" in path):
            return self._resolve_epub_reference(path)

        # Handle direct image paths - try with various prefixes
        normalized = path[3:] if path.startswith("../") else path

        # Try different prefixes (EPUB items often have paths like "OEBPS/Images/...")
        prefixes_to_try = ["", "OEBPS/", "OEBPS/Images/"]
        for prefix in prefixes_to_try:
            test_path = prefix + normalized if not normalized.startswith(prefix) else normalized
            # ebooklib has no type stubs
            item: Any = self.book.get_item_with_href(test_path)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if item:
                return cast(bytes, item.get_content())  # pyright: ignore[reportUnknownMemberType]

        logger.debug(f"Image not found in EPUB: {normalized}")
        return None

    def _resolve_epub_reference(self, ref: str) -> bytes | None:
        """Resolve an EPUB internal reference to image content.

        Parses the XHTML document to find the image element by ID.

        Args:
            ref: EPUB reference like "c02_split_007.xhtml#image_page_54_1"

        Returns:
            Image bytes, or None if not found
        """
        # Split into document and anchor
        if "#" not in ref:
            return None

        doc_href, anchor = ref.split("#", 1)

        # Try to find the document with various path prefixes
        # EPUB items often have paths like "OEBPS/Text/c02_split_007.xhtml"
        doc_item: Any = None
        found_prefix = ""
        prefixes_to_try = ["", "OEBPS/Text/", "OEBPS/", "Text/"]
        for prefix in prefixes_to_try:
            # ebooklib has no type stubs
            doc_item = self.book.get_item_with_href(prefix + doc_href)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if doc_item:
                found_prefix = prefix
                break

        if not doc_item:
            logger.debug(f"EPUB document not found: {doc_href}")
            return None

        try:
            # ebooklib has no type stubs
            html_content: str = doc_item.get_content().decode("utf-8")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

            # Find image element with matching id
            # Pattern matches: <img ... id="image_page_54_1" ... src="path" ...>
            # or: <image ... id="..." ... xlink:href="path" ...>
            escaped_anchor = re.escape(anchor)
            img_pattern = (
                rf'<(?:img|image)[^>]*id=["\']?{escaped_anchor}["\']?'
                rf'[^>]*(?:src|xlink:href)=["\']([^"\']+)["\']'
            )
            match = re.search(img_pattern, html_content, re.IGNORECASE)  # pyright: ignore[reportUnknownArgumentType]

            if not match:
                # Try reverse order (src before id)
                img_pattern = (
                    rf'<(?:img|image)[^>]*(?:src|xlink:href)=["\']([^"\']+)["\']'
                    rf'[^>]*id=["\']?{escaped_anchor}["\']?'
                )
                match = re.search(img_pattern, html_content, re.IGNORECASE)  # pyright: ignore[reportUnknownArgumentType]

            if not match:
                logger.debug(f"Image element not found for anchor: {anchor}")
                return None

            img_src = match.group(1)

            # Resolve relative path from document location
            # Use full path with prefix for correct relative resolution
            import posixpath

            full_doc_path = found_prefix + doc_href
            doc_dir = posixpath.dirname(full_doc_path)
            if img_src.startswith("../"):
                img_path = posixpath.normpath(posixpath.join(doc_dir, img_src))
            else:
                img_path = img_src

            # Get the actual image
            # ebooklib has no type stubs
            img_item: Any = self.book.get_item_with_href(img_path)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if not img_item:
                logger.debug(f"Image file not found: {img_path}")
                return None

            return cast(bytes, img_item.get_content())  # pyright: ignore[reportUnknownMemberType]

        except (UnicodeDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse EPUB reference {ref}: {e}")
            return None

    def _select_by_size(self, candidates: list[ImageCandidate], recipe: MelaRecipe) -> str | None:
        """Select image by largest area (heuristic).

        Args:
            candidates: List of candidate images
            recipe: Recipe for logging

        Returns:
            Base64-encoded image data
        """
        # Sort by area (largest first)
        candidates.sort(key=lambda c: (c.area, len(c.data)), reverse=True)
        best = candidates[0]

        logger.info(
            f"Selected image for '{recipe.title}': {best.path} (area={best.area}, heuristic)"
        )

        return base64.b64encode(best.data).decode("utf-8")

    async def _select_with_ai(
        self, candidates: list[ImageCandidate], recipe: MelaRecipe
    ) -> str | None:
        """Select image using AI verification.

        Args:
            candidates: List of candidate images
            recipe: Recipe to match against

        Returns:
            Base64-encoded image data
        """
        # Verify each candidate with AI
        for candidate in candidates:
            matches, confidence = await self._verify_image(candidate.data, recipe)
            candidate.ai_confidence = confidence
            candidate.ai_matches = matches

        # Sort by AI confidence, then by area
        candidates.sort(
            key=lambda c: (c.ai_confidence, c.area, len(c.data)),
            reverse=True,
        )

        best = candidates[0]

        if not best.ai_matches:
            logger.warning(
                f"No confident image match for '{recipe.title}' "
                f"(best confidence: {best.ai_confidence:.2f})"
            )

        logger.info(
            f"Selected image for '{recipe.title}': {best.path} "
            f"(confidence={best.ai_confidence:.2f}, area={best.area})"
        )

        return base64.b64encode(best.data).decode("utf-8")

    async def _verify_image(self, image_data: bytes, recipe: MelaRecipe) -> tuple[bool, float]:
        """Verify image matches recipe using AI vision.

        Args:
            image_data: Image bytes to verify
            recipe: Recipe to match against

        Returns:
            Tuple of (matches, confidence)
        """
        if self.client is None:
            return (True, 0.5)

        # Build context for better matching
        ingredients_list: list[str] = []
        for group in recipe.ingredients:
            ingredients_list.extend(group.ingredients[:5])
        ingredients_preview = ", ".join(ingredients_list[:5])
        if len(ingredients_list) > 5:
            ingredients_preview += "..."

        b64_image = base64.b64encode(image_data).decode("utf-8")

        prompt = f"""Does this image show the dish "{recipe.title}"?

Key ingredients to look for: {ingredients_preview}

Answer with ONE of:
- YES (confident match)
- MAYBE (possible match but uncertain)
- NO (clearly different dish)

Be specific about what you see in the image."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content
            if content is None:
                logger.error("Image verification returned empty response")
                return (True, 0.5)

            answer = content.upper()

            # Parse confidence
            if "YES" in answer:
                confidence = 0.9
            elif "MAYBE" in answer:
                confidence = 0.5
            else:
                confidence = 0.1

            matches = confidence >= self.config.ai_threshold
            logger.debug(
                f"Image verification for '{recipe.title}': {answer} (confidence={confidence})"
            )

            return (matches, confidence)

        except (KeyError, IndexError, TypeError) as e:
            # Response parsing errors
            logger.error(f"Image verification response parsing failed: {e}")
            return (True, 0.5)
        except OpenAIError as e:
            # OpenAI API errors
            logger.error(f"Image verification API call failed: {e}")
            return (True, 0.5)

    def optimize(
        self,
        image_data: bytes,
        max_width: int | None = None,
        quality: int | None = None,
    ) -> bytes:
        """Optimize an image for storage.

        Resizes if wider than max_width and compresses to target quality.

        Args:
            image_data: Raw image bytes
            max_width: Maximum width (uses config default if None)
            quality: JPEG quality (uses config default if None)

        Returns:
            Optimized image bytes
        """
        max_width = max_width or self.config.max_width
        quality = quality or self.config.quality

        try:
            with Image.open(BytesIO(image_data)) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Resize if too wide
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                # Save optimized
                output = BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                return output.getvalue()

        except UnidentifiedImageError as e:
            logger.warning(f"Unrecognized image format: {e}, returning original")
            return image_data
        except OSError as e:
            logger.warning(f"Image I/O error during optimization: {e}, returning original")
            return image_data
