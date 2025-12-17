"""Service factory for centralized dependency injection.

This module provides the ServiceFactory class which acts as a dependency
injection container, creating and managing service instances with proper
dependency sharing.

Benefits:
- Single OpenAI client instance (connection pooling)
- Centralized configuration management
- Easy testing via mock injection
- Clear dependency graph

Example:
    >>> from mela_parser.config import ExtractionConfig
    >>> config = ExtractionConfig.load()
    >>> factory = ServiceFactory(config)
    >>> extractor = factory.create_extractor()
    >>> validator = factory.create_validator()
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from ebooklib.epub import EpubBook

    from ..chapter_extractor import AsyncChapterExtractor
    from ..config import ExtractionConfig
    from ..repository import FileRecipeRepository
    from ..validator import RecipeValidator
    from .images import ImageService


@dataclass
class ServiceFactory:
    """Factory for creating service instances with shared dependencies.

    The ServiceFactory centralizes service creation and ensures that
    expensive resources (like OpenAI clients) are shared across services.

    Attributes:
        config: Extraction configuration for all services

    Example:
        >>> factory = ServiceFactory(ExtractionConfig())
        >>> extractor = factory.create_extractor()
        >>> validator = factory.create_validator()

    Note:
        The OpenAI client is lazily created and cached. This ensures
        connection pooling and efficient resource usage.
    """

    config: ExtractionConfig

    @cached_property
    def client(self) -> AsyncOpenAI:
        """Get the shared async OpenAI client.

        Returns:
            Cached AsyncOpenAI client instance

        Note:
            The client is lazily created on first access and then cached
            for subsequent calls. This ensures only one client exists
            per ServiceFactory instance.
        """
        return AsyncOpenAI()

    def create_extractor(self) -> AsyncChapterExtractor:
        """Create a chapter extractor with injected dependencies.

        Returns:
            Configured AsyncChapterExtractor instance

        Example:
            >>> extractor = factory.create_extractor()
            >>> results = await extractor.extract_from_chapters(chapters)
        """
        from ..chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=self.client,
            model=self.config.model,
            max_retries=self.config.retry_attempts,
            initial_retry_delay=self.config.initial_retry_delay,
            debug=self.config.debug_mode,
        )

    def create_validator(self) -> RecipeValidator:
        """Create a recipe validator with configured thresholds.

        Returns:
            Configured RecipeValidator instance

        Example:
            >>> validator = factory.create_validator()
            >>> score = validator.score_recipe(recipe)
        """
        from ..validator import RecipeValidator

        return RecipeValidator(
            min_ingredients=self.config.min_ingredients,
            min_instructions=self.config.min_instructions,
        )

    def create_image_service(self, book: EpubBook) -> ImageService:
        """Create an image service for a specific EPUB book.

        Args:
            book: EPUB book to extract images from

        Returns:
            Configured ImageService instance

        Example:
            >>> image_service = factory.create_image_service(book)
            >>> image_data = image_service.select_best_image(recipe)
        """
        from .images import ImageConfig, ImageService

        image_config = ImageConfig(
            min_area=self.config.min_image_area,
            max_width=self.config.max_image_width,
        )

        return ImageService(
            book=book,
            config=image_config,
            client=self.client if self.config.use_ai_verification else None,
        )

    def create_repository(self) -> FileRecipeRepository:
        """Create a recipe repository for persistence.

        Returns:
            Configured FileRecipeRepository instance

        Example:
            >>> repository = factory.create_repository()
            >>> path = repository.save(recipe, output_dir)
        """
        from ..repository import FileRecipeRepository

        return FileRecipeRepository(
            validator=self.create_validator(),
        )
