"""Unit tests for mela_parser.pipeline module.

Tests pipeline stages, context, and orchestration.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mela_parser.config import ExtractionConfig
from mela_parser.pipeline import (
    ConversionStage,
    DeduplicationStage,
    ExtractionPipeline,
    ExtractionStage,
    ImageStage,
    PersistenceStage,
    PipelineContext,
    create_default_pipeline,
)


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_create_with_required_fields(self, tmp_path: Path) -> None:
        """PipelineContext can be created with required fields."""
        config = ExtractionConfig()
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=config,
        )
        assert ctx.epub_path == Path("book.epub")
        assert ctx.output_dir == tmp_path
        assert ctx.config is config

    def test_default_field_values(self, tmp_path: Path) -> None:
        """PipelineContext has sensible defaults."""
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        assert ctx.book is None
        assert ctx.chapters == []
        assert ctx.extraction_results == []
        assert ctx.recipes == []
        assert ctx.unique_recipes == []
        assert ctx.progress_callback is None

    def test_report_progress_with_callback(self, tmp_path: Path) -> None:
        """report_progress calls callback when set."""
        mock_callback = MagicMock()
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
            progress_callback=mock_callback,
        )

        ctx.report_progress("Testing", 5, 10)

        mock_callback.assert_called_once_with("Testing", 5, 10)

    def test_report_progress_without_callback(self, tmp_path: Path) -> None:
        """report_progress does nothing when callback is None."""
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        # Should not raise
        ctx.report_progress("Testing", 5, 10)


class TestConversionStage:
    """Tests for ConversionStage."""

    def test_name(self) -> None:
        """ConversionStage has correct name."""
        stage = ConversionStage()
        assert stage.name == "Conversion"

    @pytest.mark.asyncio
    async def test_execute_converts_epub(self, tmp_path: Path) -> None:
        """ConversionStage converts EPUB and populates context."""
        from mela_parser.chapter_extractor import Chapter

        # Create mock book and chapters
        mock_book = MagicMock()
        mock_chapters = [
            Chapter(name="chapter1.xhtml", content="# Chapter 1", index=0),
            Chapter(name="chapter2.xhtml", content="# Chapter 2", index=1),
        ]

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        mock_factory = MagicMock()

        # Patch at the module where it's imported, not where defined
        with patch(
            "mela_parser.converter.convert_epub_by_chapters",
            return_value=(mock_book, mock_chapters),
        ):
            stage = ConversionStage()
            await stage.execute(ctx, mock_factory)

        assert ctx.book is mock_book
        assert len(ctx.chapters) == 2
        assert ctx.chapters[0].name == "chapter1.xhtml"


class TestExtractionStage:
    """Tests for ExtractionStage."""

    def test_name(self) -> None:
        """ExtractionStage has correct name."""
        stage = ExtractionStage()
        assert stage.name == "Extraction"

    @pytest.mark.asyncio
    async def test_execute_extracts_recipes(self, tmp_path: Path) -> None:
        """ExtractionStage extracts recipes and populates context."""
        from mela_parser.chapter_extractor import Chapter, ExtractionResult
        from mela_parser.parse import IngredientGroup, MelaRecipe

        # Set up context with chapters
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        ctx.chapters = [
            Chapter(name="ch1.xhtml", content="Recipe content", index=0),
        ]

        # Create mock extractor
        mock_recipe = MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup flour"])],
            instructions=["Mix well", "Bake until golden"],
        )
        mock_result = ExtractionResult(
            chapter_name="ch1.xhtml",
            recipes=[mock_recipe],
            error=None,
        )

        mock_extractor = AsyncMock()
        mock_extractor.extract_from_chapters.return_value = [mock_result]

        mock_factory = MagicMock()
        mock_factory.create_extractor.return_value = mock_extractor

        stage = ExtractionStage()
        await stage.execute(ctx, mock_factory)

        assert len(ctx.extraction_results) == 1
        assert len(ctx.recipes) == 1
        assert ctx.recipes[0].title == "Test Recipe"


class TestDeduplicationStage:
    """Tests for DeduplicationStage."""

    def test_name(self) -> None:
        """DeduplicationStage has correct name."""
        stage = DeduplicationStage()
        assert stage.name == "Deduplication"

    @pytest.mark.asyncio
    async def test_execute_deduplicates_recipes(self, tmp_path: Path) -> None:
        """DeduplicationStage deduplicates recipes."""
        from mela_parser.parse import IngredientGroup, MelaRecipe

        # Set up context with recipes (including duplicate)
        recipe1 = MelaRecipe(
            title="Recipe A",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )
        recipe2 = MelaRecipe(
            title="Recipe B",
            ingredients=[IngredientGroup(title="", ingredients=["2 cups"])],
            instructions=["Step 1", "Step 2"],
        )

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        ctx.recipes = [recipe1, recipe2, recipe1]  # recipe1 is duplicated

        # Mock repository
        mock_repository = MagicMock()
        mock_repository.deduplicate.return_value = [recipe1, recipe2]  # Deduplicated

        mock_factory = MagicMock()
        mock_factory.create_repository.return_value = mock_repository

        stage = DeduplicationStage()
        await stage.execute(ctx, mock_factory)

        assert len(ctx.unique_recipes) == 2


class TestImageStage:
    """Tests for ImageStage."""

    def test_name(self) -> None:
        """ImageStage has correct name."""
        stage = ImageStage()
        assert stage.name == "Images"

    @pytest.mark.asyncio
    async def test_execute_skips_when_disabled(self, tmp_path: Path) -> None:
        """ImageStage skips when extract_images is False."""
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(extract_images=False),
        )
        mock_factory = MagicMock()

        stage = ImageStage()
        await stage.execute(ctx, mock_factory)

        # Should not call create_image_service
        mock_factory.create_image_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_skips_when_no_book(self, tmp_path: Path) -> None:
        """ImageStage skips when no EPUB book available."""
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(extract_images=True),
        )
        ctx.book = None  # No book
        mock_factory = MagicMock()

        stage = ImageStage()
        await stage.execute(ctx, mock_factory)

        mock_factory.create_image_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_processes_images(self, tmp_path: Path) -> None:
        """ImageStage processes images for recipes."""
        from mela_parser.parse import IngredientGroup, MelaRecipe

        recipe = MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(extract_images=True),
        )
        ctx.book = MagicMock()
        ctx.unique_recipes = [recipe]

        # Mock image service
        mock_image_service = AsyncMock()
        mock_image_service.select_best_image.return_value = b"fake_image_data"

        mock_factory = MagicMock()
        mock_factory.create_image_service.return_value = mock_image_service

        stage = ImageStage()
        await stage.execute(ctx, mock_factory)

        mock_image_service.select_best_image.assert_called_once_with(recipe)
        assert recipe.images == [b"fake_image_data"]


class TestPersistenceStage:
    """Tests for PersistenceStage."""

    def test_name(self) -> None:
        """PersistenceStage has correct name."""
        stage = PersistenceStage()
        assert stage.name == "Persistence"

    @pytest.mark.asyncio
    async def test_execute_creates_output_dir(self, tmp_path: Path) -> None:
        """PersistenceStage creates output directory."""
        output_dir = tmp_path / "nested" / "output"
        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=output_dir,
            config=ExtractionConfig(),
        )
        ctx.unique_recipes = []

        mock_factory = MagicMock()
        mock_factory.create_repository.return_value = MagicMock()

        stage = PersistenceStage()
        await stage.execute(ctx, mock_factory)

        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_execute_saves_recipes(self, tmp_path: Path) -> None:
        """PersistenceStage saves each recipe."""
        from mela_parser.parse import IngredientGroup, MelaRecipe

        recipes = [
            MelaRecipe(
                title=f"Recipe {i}",
                ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
                instructions=["Step 1", "Step 2"],
            )
            for i in range(3)
        ]

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        ctx.unique_recipes = recipes

        mock_repository = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create_repository.return_value = mock_repository

        stage = PersistenceStage()
        await stage.execute(ctx, mock_factory)

        assert mock_repository.save.call_count == 3


class TestExtractionPipeline:
    """Tests for ExtractionPipeline orchestrator."""

    def test_init_stores_stages(self) -> None:
        """ExtractionPipeline stores provided stages."""
        stage1 = MagicMock()
        stage2 = MagicMock()

        pipeline = ExtractionPipeline([stage1, stage2])

        assert pipeline.stages == [stage1, stage2]

    @pytest.mark.asyncio
    async def test_run_executes_all_stages(self, tmp_path: Path) -> None:
        """Pipeline.run executes all stages in order."""
        stage1 = MagicMock()
        stage1.name = "Stage1"
        stage1.execute = AsyncMock()

        stage2 = MagicMock()
        stage2.name = "Stage2"
        stage2.execute = AsyncMock()

        pipeline = ExtractionPipeline([stage1, stage2])

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )
        mock_factory = MagicMock()

        await pipeline.run(ctx, mock_factory)

        stage1.execute.assert_called_once_with(ctx, mock_factory)
        stage2.execute.assert_called_once_with(ctx, mock_factory)

    @pytest.mark.asyncio
    async def test_run_executes_stages_in_order(self, tmp_path: Path) -> None:
        """Pipeline.run executes stages in the correct order."""
        call_order = []

        async def make_execute(name: str):
            async def execute(ctx, factory):
                call_order.append(name)

            return execute

        stage1 = MagicMock()
        stage1.name = "First"
        stage1.execute = AsyncMock(side_effect=lambda ctx, f: call_order.append("First"))

        stage2 = MagicMock()
        stage2.name = "Second"
        stage2.execute = AsyncMock(side_effect=lambda ctx, f: call_order.append("Second"))

        stage3 = MagicMock()
        stage3.name = "Third"
        stage3.execute = AsyncMock(side_effect=lambda ctx, f: call_order.append("Third"))

        pipeline = ExtractionPipeline([stage1, stage2, stage3])

        ctx = PipelineContext(
            epub_path=Path("book.epub"),
            output_dir=tmp_path,
            config=ExtractionConfig(),
        )

        await pipeline.run(ctx, MagicMock())

        assert call_order == ["First", "Second", "Third"]


class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline factory function."""

    def test_returns_extraction_pipeline(self) -> None:
        """create_default_pipeline returns ExtractionPipeline."""
        pipeline = create_default_pipeline()
        assert isinstance(pipeline, ExtractionPipeline)

    def test_has_five_stages(self) -> None:
        """Default pipeline has 5 stages."""
        pipeline = create_default_pipeline()
        assert len(pipeline.stages) == 5

    def test_stage_order(self) -> None:
        """Default pipeline has stages in correct order."""
        pipeline = create_default_pipeline()

        assert isinstance(pipeline.stages[0], ConversionStage)
        assert isinstance(pipeline.stages[1], ExtractionStage)
        assert isinstance(pipeline.stages[2], DeduplicationStage)
        assert isinstance(pipeline.stages[3], ImageStage)
        assert isinstance(pipeline.stages[4], PersistenceStage)

    def test_stage_names(self) -> None:
        """Default pipeline stages have expected names."""
        pipeline = create_default_pipeline()
        stage_names = [s.name for s in pipeline.stages]

        assert stage_names == [
            "Conversion",
            "Extraction",
            "Deduplication",
            "Images",
            "Persistence",
        ]
