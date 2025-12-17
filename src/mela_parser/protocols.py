"""Protocol definitions for mela_parser.

This module defines abstract interfaces (Protocols) for the key components
of the recipe extraction pipeline. Using Protocols enables:

1. **Testability**: Components can be mocked without complex inheritance
2. **Type Safety**: Type checkers validate interface compliance
3. **Flexibility**: Implementations can be swapped without code changes
4. **Documentation**: Clear contracts between components

Example:
    >>> class MockExtractor:
    ...     async def extract(self, chapters: list[Chapter]) -> list[ExtractionResult]:
    ...         return []  # Mock implementation
    ...
    >>> def pipeline(extractor: RecipeExtractor) -> None:
    ...     # Type checker verifies MockExtractor satisfies RecipeExtractor
    ...     pass
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ebooklib.epub import EpubBook

    from .chapter_extractor import Chapter, ExtractionResult
    from .parse import MelaRecipe
    from .validator import RecipeQualityScore


@runtime_checkable
class ContentConverter(Protocol):
    """Protocol for converting EPUB content to markdown chapters.

    Implementations handle the conversion of EPUB files into markdown
    chapters suitable for recipe extraction.

    Example:
        >>> class EpubConverter:
        ...     def convert(self, epub_path: Path) -> tuple[EpubBook, list[Chapter]]:
        ...         # Implementation
        ...         pass
    """

    def convert(self, epub_path: Path) -> tuple[EpubBook, list[Chapter]]:
        """Convert EPUB file to markdown chapters.

        Args:
            epub_path: Path to the EPUB file

        Returns:
            Tuple of (EpubBook, list of Chapter objects)

        Raises:
            ConversionError: If conversion fails
        """
        ...


@runtime_checkable
class RecipeExtractor(Protocol):
    """Protocol for extracting recipes from markdown content.

    Implementations use LLMs or other methods to parse markdown
    and extract structured recipe data.

    Example:
        >>> class OpenAIExtractor:
        ...     async def extract(
        ...         self, chapters: list[Chapter]
        ...     ) -> list[ExtractionResult]:
        ...         # Implementation using OpenAI
        ...         pass
    """

    async def extract(self, chapters: list[Chapter]) -> list[ExtractionResult]:
        """Extract recipes from markdown chapters.

        Args:
            chapters: List of markdown chapters to process

        Returns:
            List of extraction results containing recipes

        Raises:
            ExtractionError: If extraction fails
        """
        ...


@runtime_checkable
class RecipeValidator(Protocol):
    """Protocol for validating recipe quality.

    Implementations score recipes based on completeness,
    formatting, and other quality metrics.

    Example:
        >>> class QualityValidator:
        ...     def validate(self, recipe: MelaRecipe) -> RecipeQualityScore:
        ...         # Implementation
        ...         pass
    """

    def validate(self, recipe: MelaRecipe) -> RecipeQualityScore:
        """Validate a recipe and return quality score.

        Args:
            recipe: Recipe to validate

        Returns:
            Quality score with detailed metrics
        """
        ...


@runtime_checkable
class RecipeRepository(Protocol):
    """Protocol for recipe persistence operations.

    Implementations handle saving, loading, and deduplicating
    recipes from various storage backends.

    Example:
        >>> class FileRepository:
        ...     def save(self, recipe: MelaRecipe, output_dir: Path) -> Path:
        ...         # Save to filesystem
        ...         pass
    """

    def save(self, recipe: MelaRecipe, output_dir: Path) -> Path:
        """Save a recipe to storage.

        Args:
            recipe: Recipe to save
            output_dir: Directory for output files

        Returns:
            Path to the saved recipe file
        """
        ...

    def deduplicate(self, recipes: list[MelaRecipe]) -> list[MelaRecipe]:
        """Remove duplicate recipes from a list.

        Args:
            recipes: List of recipes to deduplicate

        Returns:
            List of unique recipes
        """
        ...


@runtime_checkable
class ImageSelector(Protocol):
    """Protocol for selecting the best image for a recipe.

    Implementations use heuristics or AI to select the most
    relevant image from available options.

    Example:
        >>> class AIImageSelector:
        ...     def select(
        ...         self, recipe: MelaRecipe, book: EpubBook
        ...     ) -> str | None:
        ...         # Use AI to verify images
        ...         pass
    """

    def select(self, recipe: MelaRecipe, book: EpubBook) -> str | None:
        """Select the best image for a recipe.

        Args:
            recipe: Recipe to find image for
            book: EPUB book containing images

        Returns:
            Base64-encoded image data, or None if no suitable image
        """
        ...


@runtime_checkable
class ImageOptimizer(Protocol):
    """Protocol for optimizing images.

    Implementations handle resizing, compression, and format
    conversion for recipe images.

    Example:
        >>> class PILOptimizer:
        ...     def optimize(self, image_data: bytes) -> bytes:
        ...         # Resize and compress
        ...         pass
    """

    def optimize(
        self,
        image_data: bytes,
        max_width: int = 600,
        quality: int = 85,
    ) -> bytes:
        """Optimize an image for storage.

        Args:
            image_data: Raw image bytes
            max_width: Maximum width in pixels
            quality: JPEG quality (1-100)

        Returns:
            Optimized image bytes
        """
        ...
