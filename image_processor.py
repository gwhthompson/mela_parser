#!/usr/bin/env python3
"""
Image extraction and AI verification for recipes.
Uses GPT-5-nano vision to match images with recipes.
"""
import base64
import logging
import re
from io import BytesIO
from typing import List, Optional, Tuple

from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from parse import MelaRecipe


class ImageProcessor:
    """Process and verify recipe images using AI vision."""

    def __init__(self, book, model: str = "gpt-5-nano"):
        """
        Initialize image processor.

        Args:
            book: EpubBook object for loading images
            model: Vision model to use for verification
        """
        self.book = book
        self.model = model
        self.client = OpenAI()

    @staticmethod
    def extract_image_paths_from_markdown(markdown: str) -> List[str]:
        """
        Extract all image paths from markdown content.

        Args:
            markdown: Markdown text containing image references

        Returns:
            List of image paths (e.g., ["../images/00011.jpeg", ...])
        """
        # MarkItDown creates: ![alt](../images/filename.jpg)
        pattern = r'!\[.*?\]\(([^)]+\.(?:jpg|jpeg|png|gif))\)'
        paths = re.findall(pattern, markdown, re.IGNORECASE)

        # Also try simpler pattern
        simple_pattern = r'(?:src=["\']?)([^"\'<>]+\.(?:png|jpg|jpeg|gif))'
        paths.extend(re.findall(simple_pattern, markdown, re.IGNORECASE))

        # Deduplicate while preserving order
        seen = set()
        unique_paths = []
        for path in paths:
            normalized = path.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(path)

        return unique_paths

    @staticmethod
    def normalize_image_path(path: str) -> str:
        """Normalize image path for EPUB lookup."""
        # Remove ../ prefix
        return path[3:] if path.startswith("../") else path

    def load_image_from_epub(self, image_path: str) -> Optional[Tuple[bytes, int, int]]:
        """
        Load image from EPUB and get dimensions.

        Args:
            image_path: Path to image in EPUB

        Returns:
            Tuple of (image_bytes, width, height) or None if failed
        """
        normalized = self.normalize_image_path(image_path)
        item = self.book.get_item_with_href(normalized)

        if not item:
            logging.debug(f"Image not found in EPUB: {normalized}")
            return None

        try:
            content = item.get_content()
            with Image.open(BytesIO(content)) as img:
                return (content, img.width, img.height)
        except (OSError, UnidentifiedImageError) as e:
            logging.debug(f"Failed to load image {normalized}: {e}")
            return None

    def verify_image_matches_recipe(
        self, image_data: bytes, recipe: MelaRecipe, threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Use AI vision to verify if image matches recipe.

        Args:
            image_data: Image bytes
            recipe: Recipe to match against
            threshold: Confidence threshold (0-1)

        Returns:
            Tuple of (matches: bool, confidence: float)
        """
        # Convert to base64
        b64_image = base64.b64encode(image_data).decode("utf-8")

        # Build context for better matching
        ingredients_preview = ", ".join(recipe.ingredients[0].ingredients[:5])
        if len(recipe.ingredients[0].ingredients) > 5:
            ingredients_preview += "..."

        prompt = f"""Does this image show the dish "{recipe.title}"?

Key ingredients to look for: {ingredients_preview}

Answer with ONE of:
- YES (confident match)
- MAYBE (possible match but uncertain)
- NO (clearly different dish)

Be specific about what you see in the image."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",  # Faster, cheaper
                                },
                            },
                        ],
                    }
                ],
            )

            answer = response.choices[0].message.content.upper()

            # Parse confidence
            if "YES" in answer:
                confidence = 0.9
            elif "MAYBE" in answer:
                confidence = 0.5
            else:
                confidence = 0.1

            matches = confidence >= threshold
            logging.debug(
                f"Image verification for '{recipe.title}': {answer} (confidence={confidence})"
            )

            return (matches, confidence)

        except Exception as e:
            logging.error(f"Image verification failed: {e}")
            # On error, fall back to heuristic
            return (True, 0.5)

    def select_best_image_for_recipe(
        self,
        image_paths: List[str],
        recipe: MelaRecipe,
        use_ai_verification: bool = True,
        min_area: int = 300000,
    ) -> Optional[str]:
        """
        Select the best image for a recipe.

        Args:
            image_paths: List of candidate image paths
            recipe: Recipe to match
            use_ai_verification: Whether to use AI vision verification
            min_area: Minimum image area (width × height)

        Returns:
            Base64 encoded image or None
        """
        if not image_paths:
            return None

        candidates = []

        # Load all candidate images
        for path in image_paths:
            result = self.load_image_from_epub(path)
            if not result:
                continue

            image_data, width, height = result
            area = width * height

            # Filter by minimum area
            if area < min_area:
                logging.debug(f"Image too small: {path} ({area} pixels)")
                continue

            candidates.append(
                {
                    "path": path,
                    "data": image_data,
                    "width": width,
                    "height": height,
                    "area": area,
                }
            )

        if not candidates:
            logging.debug(f"No suitable images found for '{recipe.title}'")
            return None

        # If AI verification enabled, verify each candidate
        if use_ai_verification:
            for candidate in candidates:
                matches, confidence = self.verify_image_matches_recipe(
                    candidate["data"], recipe
                )
                candidate["ai_confidence"] = confidence
                candidate["ai_matches"] = matches

            # Sort by AI confidence, then by area, then by file size
            candidates.sort(
                key=lambda c: (
                    c.get("ai_confidence", 0),
                    c["area"],
                    len(c["data"]),
                ),
                reverse=True,
            )

            # Pick best match
            best = candidates[0]

            if not best.get("ai_matches", False):
                logging.warning(
                    f"No confident image match for '{recipe.title}' "
                    f"(best confidence: {best.get('ai_confidence', 0):.2f})"
                )
                # Use it anyway if it's the only option
                return base64.b64encode(best["data"]).decode("utf-8")

            logging.info(
                f"Selected image for '{recipe.title}': {best['path']} "
                f"(confidence={best['ai_confidence']:.2f}, area={best['area']})"
            )
            return base64.b64encode(best["data"]).decode("utf-8")

        else:
            # Heuristic only: pick largest image
            candidates.sort(key=lambda c: (c["area"], len(c["data"])), reverse=True)
            best = candidates[0]

            logging.info(
                f"Selected image for '{recipe.title}': {best['path']} "
                f"(area={best['area']}, heuristic)"
            )
            return base64.b64encode(best["data"]).decode("utf-8")


def extract_images_for_recipe(
    markdown_chunks: List[str],
    chunk_index: int,
    recipe: MelaRecipe,
    book,
    use_ai_verification: bool = True,
) -> List[str]:
    """
    Extract and verify images for a recipe from chunks.

    Args:
        markdown_chunks: All markdown chunks
        chunk_index: Index of chunk containing this recipe
        recipe: Recipe to find images for
        book: EpubBook for loading images
        use_ai_verification: Whether to use AI vision

    Returns:
        List containing base64 encoded image (or empty if no image found)
    """
    image_processor = ImageProcessor(book, model="gpt-5-nano")

    # Collect image paths from current + adjacent chunks
    all_image_paths = []

    # Previous chunk (images before recipe)
    if chunk_index > 0:
        prev_chunk = markdown_chunks[chunk_index - 1]
        all_image_paths.extend(image_processor.extract_image_paths_from_markdown(prev_chunk))

    # Current chunk
    current_chunk = markdown_chunks[chunk_index]
    all_image_paths.extend(image_processor.extract_image_paths_from_markdown(current_chunk))

    # Next chunk (images after recipe)
    if chunk_index < len(markdown_chunks) - 1:
        next_chunk = markdown_chunks[chunk_index + 1]
        all_image_paths.extend(image_processor.extract_image_paths_from_markdown(next_chunk))

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for path in all_image_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    logging.debug(f"Found {len(unique_paths)} candidate images for '{recipe.title}'")

    # Select best image
    best_image = image_processor.select_best_image_for_recipe(
        unique_paths, recipe, use_ai_verification=use_ai_verification
    )

    if best_image:
        return [best_image]
    else:
        return []
