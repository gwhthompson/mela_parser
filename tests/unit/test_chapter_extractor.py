"""Unit tests for mela_parser.chapter_extractor module.

Tests Chapter, ExtractionResult data models, and exception classes.
"""

import pytest

from mela_parser.chapter_extractor import (
    Chapter,
    ChapterExtractionError,
    ExtractionResult,
    RecipeExtractionError,
)
from mela_parser.parse import IngredientGroup, MelaRecipe


class TestChapterExtractionExceptions:
    """Tests for chapter extraction exception classes."""

    def test_chapter_extraction_error_is_exception(self) -> None:
        """ChapterExtractionError inherits from Exception."""
        error = ChapterExtractionError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_recipe_extraction_error_inherits(self) -> None:
        """RecipeExtractionError inherits from ChapterExtractionError."""
        error = RecipeExtractionError("recipe failed")
        assert isinstance(error, ChapterExtractionError)
        assert isinstance(error, Exception)

    def test_can_catch_both_as_base(self) -> None:
        """Both exceptions can be caught as ChapterExtractionError."""
        with pytest.raises(ChapterExtractionError):
            raise RecipeExtractionError("nested")


class TestChapterDataclass:
    """Tests for Chapter frozen dataclass."""

    def test_create_chapter(self) -> None:
        """Chapter can be created with valid data."""
        chapter = Chapter(name="Introduction", content="Welcome to the book.", index=0)
        assert chapter.name == "Introduction"
        assert chapter.content == "Welcome to the book."
        assert chapter.index == 0

    def test_chapter_is_frozen(self) -> None:
        """Chapter is immutable (frozen dataclass)."""
        chapter = Chapter(name="Test", content="Content", index=1)
        with pytest.raises(AttributeError):
            chapter.name = "New Name"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        """Chapter with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Chapter name cannot be empty"):
            Chapter(name="", content="Some content", index=0)

    def test_empty_content_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chapter with empty content logs warning but doesn't raise."""
        import logging

        with caplog.at_level(logging.WARNING):
            chapter = Chapter(name="Empty Chapter", content="", index=0)

        assert chapter.name == "Empty Chapter"
        assert "Empty Chapter" in caplog.text
        assert "empty content" in caplog.text

    def test_chapter_equality(self) -> None:
        """Chapter instances with same data are equal."""
        ch1 = Chapter(name="Test", content="Content", index=0)
        ch2 = Chapter(name="Test", content="Content", index=0)
        assert ch1 == ch2

    def test_chapter_inequality(self) -> None:
        """Chapter instances with different data are not equal."""
        ch1 = Chapter(name="Test", content="Content", index=0)
        ch2 = Chapter(name="Test", content="Different", index=0)
        assert ch1 != ch2


class TestExtractionResultDataclass:
    """Tests for ExtractionResult dataclass."""

    @pytest.fixture
    def sample_recipe(self) -> MelaRecipe:
        """Create a sample recipe for testing."""
        return MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup flour"])],
            instructions=["Mix ingredients.", "Bake at 350F."],
        )

    def test_create_extraction_result(self, sample_recipe: MelaRecipe) -> None:
        """ExtractionResult can be created with valid data."""
        result = ExtractionResult(
            chapter_name="Main Dishes",
            recipes=[sample_recipe],
        )
        assert result.chapter_name == "Main Dishes"
        assert len(result.recipes) == 1

    def test_default_values(self) -> None:
        """ExtractionResult has sensible defaults."""
        result = ExtractionResult(chapter_name="Test", recipes=[])
        assert result.error is None
        assert result.retry_count == 0
        assert result.expected_count is None
        assert result.titles_found is None

    def test_is_success_when_no_error(self, sample_recipe: MelaRecipe) -> None:
        """is_success is True when no error."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe],
            error=None,
        )
        assert result.is_success is True

    def test_is_success_false_when_error(self) -> None:
        """is_success is False when there is an error."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[],
            error="Extraction failed",
        )
        assert result.is_success is False

    def test_recipe_count(self, sample_recipe: MelaRecipe) -> None:
        """recipe_count returns number of recipes."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe, sample_recipe, sample_recipe],
        )
        assert result.recipe_count == 3

    def test_recipe_count_empty(self) -> None:
        """recipe_count returns 0 for empty list."""
        result = ExtractionResult(chapter_name="Test", recipes=[])
        assert result.recipe_count == 0

    def test_completeness_calculation(self, sample_recipe: MelaRecipe) -> None:
        """completeness calculates extracted/expected ratio."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe, sample_recipe],
            expected_count=4,
        )
        assert result.completeness == 0.5

    def test_completeness_full(self, sample_recipe: MelaRecipe) -> None:
        """completeness is 1.0 when all expected recipes extracted."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe, sample_recipe, sample_recipe],
            expected_count=3,
        )
        assert result.completeness == 1.0

    def test_completeness_none_when_no_expected(self, sample_recipe: MelaRecipe) -> None:
        """completeness is None when expected_count is None."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe],
            expected_count=None,
        )
        assert result.completeness is None

    def test_completeness_none_when_expected_zero(self) -> None:
        """completeness is None when expected_count is 0."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[],
            expected_count=0,
        )
        assert result.completeness is None

    def test_completeness_over_100_percent(self, sample_recipe: MelaRecipe) -> None:
        """completeness can exceed 1.0 (over 100%)."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe] * 5,
            expected_count=3,
        )
        # 5/3 = 1.666...
        assert result.completeness is not None
        assert result.completeness > 1.0

    def test_with_titles_found(self, sample_recipe: MelaRecipe) -> None:
        """ExtractionResult can store titles found in Stage 1."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[sample_recipe],
            titles_found=["Recipe A", "Recipe B"],
        )
        assert result.titles_found == ["Recipe A", "Recipe B"]

    def test_with_retry_count(self) -> None:
        """ExtractionResult tracks retry attempts."""
        result = ExtractionResult(
            chapter_name="Test",
            recipes=[],
            error="Timeout",
            retry_count=3,
        )
        assert result.retry_count == 3
