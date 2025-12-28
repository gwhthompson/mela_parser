"""Recipe repository for persistence operations.

This module provides the repository layer for saving and managing recipes.
It abstracts file system operations and provides deduplication logic.

Example:
    >>> from mela_parser.repository import FileRecipeRepository
    >>> repository = FileRecipeRepository()
    >>> path = repository.save(recipe, Path("output"))
    >>> unique = repository.deduplicate(recipes)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .parse import MelaRecipe
    from .validator import RecipeValidator

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Useful for generating filenames from recipe titles.

    Args:
        text: Text to slugify

    Returns:
        Lowercase, hyphenated slug

    Example:
        >>> slugify("Roasted Chicken & Vegetables!")
        'roasted-chicken-vegetables'
    """
    # Remove special characters
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace whitespace with hyphens
    slug = re.sub(r"[\s_-]+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    return slug or "recipe"


class RecipeDict(dict[str, Any]):
    """Type alias for recipe dictionaries in Mela format."""

    pass


class FileRecipeRepository:
    """Repository for saving recipes to the filesystem.

    Handles all file I/O operations for recipes, including:
    - Conversion from MelaRecipe to Mela JSON format
    - File writing with deduplication
    - Recipe validation before save

    Attributes:
        validator: Optional validator for recipe quality checks
        written_titles: Set of titles already written (for deduplication)

    Example:
        >>> repo = FileRecipeRepository()
        >>> path = repo.save(recipe, Path("output"))
        >>> print(f"Saved to: {path}")
    """

    def __init__(self, validator: RecipeValidator | None = None) -> None:
        """Initialize the repository.

        Args:
            validator: Optional validator for quality checks
        """
        self.validator = validator
        self._written_titles: set[str] = set()

    def save(self, recipe: MelaRecipe, output_dir: Path) -> Path | None:
        """Save a recipe to the filesystem.

        Converts the recipe to Mela JSON format and writes to disk.
        Skips recipes that are incomplete or already saved.

        Args:
            recipe: Recipe to save
            output_dir: Directory to save to

        Returns:
            Path to saved file, or None if skipped

        Example:
            >>> path = repo.save(recipe, Path("output"))
        """
        # Convert to dict format
        recipe_dict = self._to_dict(recipe)

        # Validate required fields
        if not self._is_valid(recipe_dict):
            logger.info(f"Skipping incomplete recipe: {recipe.title}")
            return None

        # Check for duplicates
        title_key = recipe.title.lower().strip()
        if title_key in self._written_titles:
            logger.debug(f"Skipping duplicate: {recipe.title}")
            return None

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{self._slugify(recipe.title)}.melarecipe"
        filepath = output_dir / filename

        # Skip if exists
        if filepath.exists():
            logger.debug(f"File exists: {filepath}")
            return None

        # Write to disk
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(recipe_dict, f, ensure_ascii=False, indent=2)

        self._written_titles.add(title_key)
        logger.info(f"Saved recipe: {filepath}")
        return filepath

    def deduplicate(self, recipes: list[MelaRecipe]) -> list[MelaRecipe]:
        """Remove duplicate recipes from a list.

        Deduplicates by title (case-insensitive, trimmed).

        Args:
            recipes: List of recipes to deduplicate

        Returns:
            List of unique recipes

        Example:
            >>> unique = repo.deduplicate(recipes)
            >>> print(f"Removed {len(recipes) - len(unique)} duplicates")
        """
        seen: set[str] = set()
        unique: list[MelaRecipe] = []

        for recipe in recipes:
            key = recipe.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(recipe)

        return unique

    def _to_dict(self, recipe: MelaRecipe) -> dict[str, Any]:
        """Convert MelaRecipe to Mela JSON format.

        Args:
            recipe: Recipe to convert

        Returns:
            Dictionary in Mela JSON format
        """

        # Format time values
        def format_minutes(minutes: int | str | None) -> str:
            if minutes is None:
                return ""
            try:
                mins = int(minutes)
            except (ValueError, TypeError):
                return str(minutes) if minutes else ""
            if mins <= 0:
                return ""
            hours, remainder = divmod(mins, 60)
            if hours > 0:
                return f"{hours} hr {remainder} min" if remainder > 0 else f"{hours} hr"
            return f"{remainder} min"

        # Build ingredients string
        if len(recipe.ingredients) == 1:
            ingredients_str = "\n".join(recipe.ingredients[0].ingredients)
        else:
            ingredients_str = "\n".join(
                f"# {grp.title}\n" + "\n".join(grp.ingredients) for grp in recipe.ingredients
            )

        # Build instructions string
        instructions_str = "\n".join(recipe.instructions)

        # Extract category values
        categories = [c.value for c in recipe.categories] if recipe.categories else []

        return {
            "id": str(uuid.uuid4()),
            "title": recipe.title,
            "ingredients": ingredients_str,
            "instructions": instructions_str,
            "recipeYield": recipe.recipeYield or "",
            "prepTime": format_minutes(recipe.prepTime),
            "cookTime": format_minutes(recipe.cookTime),
            "totalTime": format_minutes(recipe.totalTime),
            "categories": categories,
            "images": recipe.images,
            "link": "",  # Will be set by caller if needed
        }

    def _is_valid(self, recipe_dict: dict[str, Any]) -> bool:
        """Check if recipe has required fields.

        Args:
            recipe_dict: Recipe dictionary to validate

        Returns:
            True if recipe has all required fields
        """
        return bool(
            recipe_dict.get("title", "").strip()
            and recipe_dict.get("ingredients", "").strip()
            and recipe_dict.get("instructions", "").strip()
        )

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-safe slug.

        Args:
            text: Text to slugify

        Returns:
            Lowercase, hyphenated slug
        """
        return slugify(text)
