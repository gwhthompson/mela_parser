"""Pipeline architecture for recipe extraction.

This module implements a stage-based pipeline pattern for extracting recipes
from EPUB cookbooks. The pipeline provides:

1. **Explicit Stages**: Clear separation of conversion, extraction, deduplication
2. **Shared Context**: Type-safe context passing between stages
3. **Extensibility**: Add new stages without modifying existing code
4. **Testability**: Each stage can be tested in isolation

Example:
    >>> from mela_parser.pipeline import (
    ...     PipelineContext, ExtractionPipeline, create_default_pipeline
    ... )
    >>> context = PipelineContext(
    ...     epub_path=Path("cookbook.epub"),
    ...     output_dir=Path("output"),
    ...     config=config,
    ... )
    >>> pipeline = create_default_pipeline()
    >>> await pipeline.run(context, factory)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebooklib.epub import EpubBook

    from .chapter_extractor import Chapter, ExtractionResult
    from .config import ExtractionConfig
    from .parse import MelaRecipe
    from .services.factory import ServiceFactory

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state passed through pipeline stages.

    The context holds all data accumulated during pipeline execution.
    Stages read inputs and write outputs to this context.

    Attributes:
        epub_path: Path to the input EPUB file
        output_dir: Directory for output files
        config: Extraction configuration

        book: EPUB book object (populated by ConversionStage)
        chapters: List of markdown chapters (populated by ConversionStage)
        extraction_results: Raw extraction results (populated by ExtractionStage)
        recipes: Flat list of all recipes (populated by ExtractionStage)
        unique_recipes: Deduplicated recipes (populated by DeduplicationStage)

    Example:
        >>> context = PipelineContext(
        ...     epub_path=Path("cookbook.epub"),
        ...     output_dir=Path("output"),
        ...     config=ExtractionConfig(),
        ... )
    """

    # Required inputs
    epub_path: Path
    output_dir: Path
    config: ExtractionConfig

    # Populated by stages
    book: EpubBook | None = None
    chapters: list[Chapter] = field(default_factory=list)
    extraction_results: list[ExtractionResult] = field(default_factory=list)
    recipes: list[MelaRecipe] = field(default_factory=list)
    unique_recipes: list[MelaRecipe] = field(default_factory=list)

    # Progress tracking
    progress_callback: Callable[[str, int, int], None] | None = None

    def report_progress(self, stage: str, current: int, total: int) -> None:
        """Report progress to callback if set.

        Args:
            stage: Name of the current stage
            current: Current item number
            total: Total number of items
        """
        if self.progress_callback:
            self.progress_callback(stage, current, total)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    Each stage performs a specific transformation on the pipeline context.
    Stages should be stateless - all state is stored in the context.

    Example:
        >>> class CustomStage(PipelineStage):
        ...     @property
        ...     def name(self) -> str:
        ...         return "Custom"
        ...
        ...     async def execute(
        ...         self, ctx: PipelineContext, factory: ServiceFactory
        ...     ) -> None:
        ...         # Implementation
        ...         pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this stage."""
        ...

    @abstractmethod
    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Execute this pipeline stage.

        Args:
            ctx: Pipeline context with shared state
            factory: Service factory for creating dependencies

        Raises:
            Exception: If stage execution fails
        """
        ...


class ConversionStage(PipelineStage):
    """Stage 1: Convert EPUB to markdown chapters.

    Reads the EPUB file and converts each chapter to clean markdown
    suitable for recipe extraction.

    Populates:
        - ctx.book: The EPUB book object
        - ctx.chapters: List of Chapter objects with markdown content
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "Conversion"

    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Convert EPUB to markdown chapters."""
        from .converter import convert_epub_by_chapters

        logger.info(f"Converting EPUB: {ctx.epub_path}")
        ctx.book, ctx.chapters = convert_epub_by_chapters(str(ctx.epub_path))
        logger.info(f"Converted {len(ctx.chapters)} chapters")


class ExtractionStage(PipelineStage):
    """Stage 2: Extract recipes from chapters.

    Uses AI to extract structured recipe data from markdown chapters.
    Processes chapters in parallel for efficiency.

    Populates:
        - ctx.extraction_results: Raw results from extraction
        - ctx.recipes: Flat list of all extracted recipes
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "Extraction"

    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Extract recipes from all chapters."""
        extractor = factory.create_extractor()

        logger.info(f"Extracting recipes from {len(ctx.chapters)} chapters")
        ctx.extraction_results = await extractor.extract_from_chapters(
            ctx.chapters,
            max_concurrent=ctx.config.max_concurrent,
        )

        # Flatten recipes from all results
        ctx.recipes = [recipe for result in ctx.extraction_results for recipe in result.recipes]
        logger.info(f"Extracted {len(ctx.recipes)} total recipes")


class DeduplicationStage(PipelineStage):
    """Stage 3: Deduplicate recipes.

    Removes duplicate recipes based on title similarity.

    Populates:
        - ctx.unique_recipes: Deduplicated list of recipes
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "Deduplication"

    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Deduplicate recipes by title."""
        repository = factory.create_repository()

        logger.info(f"Deduplicating {len(ctx.recipes)} recipes")
        ctx.unique_recipes = repository.deduplicate(ctx.recipes)

        duplicates_removed = len(ctx.recipes) - len(ctx.unique_recipes)
        logger.info(
            f"Removed {duplicates_removed} duplicates, {len(ctx.unique_recipes)} unique recipes"
        )


class ImageStage(PipelineStage):
    """Stage 4: Process images for recipes.

    Extracts and processes images from the EPUB, selecting the best
    image for each recipe based on configured strategy.

    Updates:
        - recipe.images for each recipe in ctx.unique_recipes
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "Images"

    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Process images for all recipes."""
        if not ctx.config.extract_images:
            logger.info("Image extraction disabled")
            return

        if ctx.book is None:
            logger.warning("No EPUB book available for image extraction")
            return

        image_service = factory.create_image_service(ctx.book)

        logger.info(f"Processing images for {len(ctx.unique_recipes)} recipes")
        for i, recipe in enumerate(ctx.unique_recipes):
            ctx.report_progress("Images", i + 1, len(ctx.unique_recipes))
            image_data = await image_service.select_best_image(recipe)
            if image_data:
                recipe.images = [image_data]


class PersistenceStage(PipelineStage):
    """Stage 5: Save recipes to disk.

    Writes all unique recipes to the output directory in Mela format.

    Side effects:
        - Creates .melarecipe files in ctx.output_dir
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "Persistence"

    async def execute(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Save all recipes to disk."""
        repository = factory.create_repository()

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(ctx.unique_recipes)} recipes to {ctx.output_dir}")

        for i, recipe in enumerate(ctx.unique_recipes):
            ctx.report_progress("Saving", i + 1, len(ctx.unique_recipes))
            repository.save(recipe, ctx.output_dir)


class ExtractionPipeline:
    """Orchestrates recipe extraction through a series of stages.

    The pipeline executes stages in order, passing shared context
    between them. Stages can be customized by providing a different
    list of stages.

    Attributes:
        stages: Ordered list of pipeline stages to execute

    Example:
        >>> pipeline = ExtractionPipeline([
        ...     ConversionStage(),
        ...     ExtractionStage(),
        ...     DeduplicationStage(),
        ...     PersistenceStage(),
        ... ])
        >>> await pipeline.run(context, factory)
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Initialize the pipeline with stages.

        Args:
            stages: Ordered list of stages to execute
        """
        self.stages = stages

    async def run(self, ctx: PipelineContext, factory: ServiceFactory) -> None:
        """Execute all pipeline stages in order.

        Args:
            ctx: Pipeline context with inputs and shared state
            factory: Service factory for creating dependencies

        Raises:
            Exception: If any stage fails
        """
        for stage in self.stages:
            logger.info(f"Starting stage: {stage.name}")
            await stage.execute(ctx, factory)
            logger.info(f"Completed stage: {stage.name}")


def create_default_pipeline() -> ExtractionPipeline:
    """Create the default extraction pipeline.

    Returns:
        Pipeline with all standard stages:
        1. ConversionStage - EPUB to markdown
        2. ExtractionStage - AI recipe extraction
        3. DeduplicationStage - Remove duplicates
        4. ImageStage - Process images
        5. PersistenceStage - Save to disk

    Example:
        >>> pipeline = create_default_pipeline()
        >>> await pipeline.run(context, factory)
    """
    return ExtractionPipeline(
        [
            ConversionStage(),
            ExtractionStage(),
            DeduplicationStage(),
            ImageStage(),
            PersistenceStage(),
        ]
    )
