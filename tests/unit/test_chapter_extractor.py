"""Unit tests for mela_parser.chapter_extractor module.

Tests Chapter, ExtractionResult data models, and exception classes.
"""

# pyright: reportAttributeAccessIssue=false

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


# ============================================================================
# AsyncChapterExtractor Tests
# ============================================================================


class TestAsyncChapterExtractorInit:
    """Tests for AsyncChapterExtractor initialization."""

    def test_default_initialization(self, mock_async_openai_client) -> None:
        """AsyncChapterExtractor initializes with default values."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )
        assert extractor.model == "gpt-5-nano"
        assert extractor.max_retries == 3
        assert extractor.initial_retry_delay == 1.0
        assert extractor.use_grounded_extraction is True
        assert extractor.debug_dir is None

    def test_custom_model_and_retries(self, mock_async_openai_client) -> None:
        """AsyncChapterExtractor accepts custom model and retry settings."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            model="gpt-5-mini",
            max_retries=5,
            initial_retry_delay=2.0,
            debug=False,
        )
        assert extractor.model == "gpt-5-mini"
        assert extractor.max_retries == 5
        assert extractor.initial_retry_delay == 2.0

    def test_debug_mode_creates_directory(self, mock_async_openai_client, tmp_path) -> None:
        """Debug mode creates the debug directory."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        debug_dir = tmp_path / "debug_test"
        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=True,
            debug_dir=debug_dir,
        )
        assert extractor.debug_dir == debug_dir
        assert debug_dir.exists()

    def test_debug_mode_auto_creates_timestamped_directory(
        self, mock_async_openai_client, tmp_path, monkeypatch
    ) -> None:
        """Debug mode auto-creates timestamped directory when none provided."""

        from mela_parser.chapter_extractor import AsyncChapterExtractor

        # Change to tmp_path so debug directory is created there
        monkeypatch.chdir(tmp_path)

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=True,
            debug_dir=None,
        )
        assert extractor.debug_dir is not None
        assert extractor.debug_dir.exists()
        # Should be in format debug/YYYYMMDD_HHMMSS
        assert "debug" in str(extractor.debug_dir)

    def test_use_grounded_extraction_flag(self, mock_async_openai_client) -> None:
        """use_grounded_extraction flag controls extraction mode."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            use_grounded_extraction=False,
            debug=False,
        )
        assert extractor.use_grounded_extraction is False

    def test_config_integration(self, mock_async_openai_client) -> None:
        """AsyncChapterExtractor uses ExtractionConfig when provided."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor
        from mela_parser.config import ExtractionConfig

        config = ExtractionConfig(
            extraction_concurrency_per_chapter=10,
            extraction_retry_attempts=5,
        )
        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            config=config,
            debug=False,
        )
        assert extractor.config.extraction_concurrency_per_chapter == 10
        assert extractor.config.extraction_retry_attempts == 5


class TestAsyncChapterExtractorExtraction:
    """Tests for AsyncChapterExtractor extraction methods."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with mocked client."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_extract_from_chapters_parallel(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """extract_from_chapters processes chapters in parallel."""
        from unittest.mock import MagicMock

        from mela_parser.chapter_extractor import Chapter
        from mela_parser.parse import ChapterTitles

        # Setup mock response for title enumeration (returns no titles)
        title_response = MagicMock()
        title_response.output_parsed = ChapterTitles(titles=[], chapter_type="recipes")
        extractor.client.responses.parse.return_value = title_response

        chapters = [
            Chapter(name="Chapter 1", content="Content 1", index=0),
            Chapter(name="Chapter 2", content="Content 2", index=1),
        ]

        results = await extractor.extract_from_chapters(chapters, max_concurrent=2)

        assert len(results) == 2
        assert results[0].chapter_name == "Chapter 1"
        assert results[1].chapter_name == "Chapter 2"

    @pytest.mark.asyncio
    async def test_extract_from_chapters_handles_exception(self, extractor, sample_chapter) -> None:
        """extract_from_chapters handles exceptions gracefully."""
        from unittest.mock import AsyncMock

        from mela_parser.chapter_extractor import Chapter

        # Mock extract_from_chapter to raise an exception
        # (This tests the exception handling in extract_from_chapters,
        # not the graceful error handling in enumerate_titles)
        extractor.extract_from_chapter = AsyncMock(side_effect=Exception("API Error"))

        chapters = [Chapter(name="Failing Chapter", content="Content", index=0)]

        results = await extractor.extract_from_chapters(chapters)

        assert len(results) == 1
        assert results[0].is_success is False
        assert "API Error" in results[0].error

    @pytest.mark.asyncio
    async def test_extract_from_chapter_dispatches_to_grounded(
        self, extractor, sample_chapter
    ) -> None:
        """extract_from_chapter uses grounded extraction by default."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        title_response = MagicMock()
        title_response.output_parsed = ChapterTitles(titles=[], chapter_type="recipes")
        extractor.client.responses.parse.return_value = title_response

        result = await extractor.extract_from_chapter(sample_chapter)

        assert result.chapter_name == "Main Dishes"
        # Should have called the API for title enumeration
        extractor.client.responses.parse.assert_called()

    @pytest.mark.asyncio
    async def test_extract_from_chapter_dispatches_to_paginated(
        self, mock_async_openai_client, sample_chapter, sample_mela_recipe
    ) -> None:
        """extract_from_chapter uses paginated extraction when configured."""
        from unittest.mock import MagicMock

        from mela_parser.chapter_extractor import AsyncChapterExtractor
        from mela_parser.parse import CookbookRecipes

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            use_grounded_extraction=False,
            debug=False,
        )

        # Setup mock response for paginated extraction
        response = MagicMock()
        response.output_parsed = CookbookRecipes(
            recipes=[sample_mela_recipe],
            has_more=False,
        )
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        extractor.client.responses.parse.return_value = response

        result = await extractor.extract_from_chapter(sample_chapter)

        assert result.chapter_name == "Main Dishes"
        assert result.recipe_count == 1


class TestAsyncChapterExtractorGroundedExtraction:
    """Tests for grounded (two-stage) extraction."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with mocked client."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_extract_grounded_empty_titles(self, extractor, sample_chapter) -> None:
        """Grounded extraction handles chapters with no titles."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        response = MagicMock()
        response.output_parsed = ChapterTitles(titles=[], chapter_type="recipes")
        extractor.client.responses.parse.return_value = response

        result = await extractor._extract_grounded(sample_chapter)

        assert result.recipe_count == 0
        assert result.expected_count == 0
        assert result.titles_found == []

    @pytest.mark.asyncio
    async def test_extract_grounded_deduplicates_titles(
        self, extractor, sample_mela_recipe
    ) -> None:
        """Grounded extraction deduplicates repeated titles."""
        from unittest.mock import MagicMock

        from mela_parser.chapter_extractor import Chapter
        from mela_parser.parse import ChapterTitles

        # Use a chapter with content that matches the titles
        chapter = Chapter(
            name="Test Chapter",
            content="# Recipe A\nContent for A\n\n# Recipe B\nContent for B",
            index=0,
        )

        # First call returns titles with duplicates
        title_response = MagicMock()
        title_response.output_parsed = ChapterTitles(
            titles=["Recipe A", "Recipe A", "Recipe B"],
            chapter_type="recipes",
        )

        # Recipe extraction returns recipes
        recipe_response = MagicMock()
        recipe_response.output_parsed = sample_mela_recipe

        extractor.client.responses.parse.side_effect = [
            title_response,
            recipe_response,
            recipe_response,
        ]

        result = await extractor._extract_grounded(chapter)

        # Should have deduplicated to 2 unique titles
        assert result.expected_count == 2

    @pytest.mark.asyncio
    async def test_extract_grounded_skips_intro_chapters(self, extractor, sample_chapter) -> None:
        """Grounded extraction skips intro/index chapters."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        response = MagicMock()
        response.output_parsed = ChapterTitles(
            titles=["Some Title"],
            chapter_type="intro",  # Should be skipped
        )
        extractor.client.responses.parse.return_value = response

        result = await extractor._extract_grounded(sample_chapter)

        # enumerate_titles should return empty for intro chapters
        assert result.recipe_count == 0


class TestAsyncChapterExtractorPaginatedExtraction:
    """Tests for paginated extraction with retry logic."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with paginated mode."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            use_grounded_extraction=False,
            max_retries=2,
            initial_retry_delay=0.01,  # Fast retries for testing
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_extract_paginated_success(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """Paginated extraction succeeds on first attempt."""
        from unittest.mock import MagicMock

        from mela_parser.parse import CookbookRecipes

        response = MagicMock()
        response.output_parsed = CookbookRecipes(
            recipes=[sample_mela_recipe],
            has_more=False,
        )
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        extractor.client.responses.parse.return_value = response

        result = await extractor._extract_paginated(sample_chapter)

        assert result.is_success
        assert result.recipe_count == 1
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_extract_paginated_retry_on_failure(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """Paginated extraction retries on failure."""
        from unittest.mock import MagicMock

        from openai import OpenAIError

        from mela_parser.parse import CookbookRecipes

        # First call fails, second succeeds
        success_response = MagicMock()
        success_response.output_parsed = CookbookRecipes(
            recipes=[sample_mela_recipe],
            has_more=False,
        )
        success_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        extractor.client.responses.parse.side_effect = [
            OpenAIError("Temporary failure"),
            success_response,
        ]

        result = await extractor._extract_paginated(sample_chapter)

        assert result.is_success
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_extract_paginated_exhausts_retries(self, extractor, sample_chapter) -> None:
        """Paginated extraction fails after exhausting retries."""
        from openai import OpenAIError

        extractor.client.responses.parse.side_effect = OpenAIError("Persistent failure")

        result = await extractor._extract_paginated(sample_chapter)

        assert result.is_success is False
        assert result.retry_count == 2  # max_retries = 2
        assert "Persistent failure" in result.error


class TestAsyncChapterExtractorLLMExtraction:
    """Tests for _extract_with_llm pagination logic."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with debug disabled for cleaner testing."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            use_grounded_extraction=False,
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_extract_with_llm_single_page(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """LLM extraction completes in single page."""
        from unittest.mock import MagicMock

        from mela_parser.parse import CookbookRecipes

        response = MagicMock()
        response.output_parsed = CookbookRecipes(
            recipes=[sample_mela_recipe],
            has_more=False,
        )
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        extractor.client.responses.parse.return_value = response

        recipes = await extractor._extract_with_llm(sample_chapter)

        assert len(recipes) == 1
        assert recipes[0].title == "Roasted Chicken"

    @pytest.mark.asyncio
    async def test_extract_with_llm_pagination(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """LLM extraction handles pagination across multiple pages."""
        from unittest.mock import MagicMock

        from mela_parser.parse import CookbookRecipes, IngredientGroup, MelaRecipe

        recipe1 = sample_mela_recipe
        recipe2 = MelaRecipe(
            title="Grilled Salmon",
            ingredients=[IngredientGroup(title="", ingredients=["1 salmon fillet"])],
            instructions=["Grill the salmon.", "Serve hot."],
        )

        # First page has more
        page1_response = MagicMock()
        page1_response.output_parsed = CookbookRecipes(
            recipes=[recipe1],
            has_more=True,
            last_content_marker="Roast for 1 hour.",
        )
        page1_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        # Second page is final
        page2_response = MagicMock()
        page2_response.output_parsed = CookbookRecipes(
            recipes=[recipe2],
            has_more=False,
        )
        page2_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        extractor.client.responses.parse.side_effect = [page1_response, page2_response]

        # Use chapter content that contains the marker
        from mela_parser.chapter_extractor import Chapter

        chapter = Chapter(
            name="Test",
            content="Recipe 1\nRoast for 1 hour.\n\n## Grilled Salmon\nGrill instructions.",
            index=0,
        )

        recipes = await extractor._extract_with_llm(chapter)

        assert len(recipes) == 2

    @pytest.mark.asyncio
    async def test_extract_with_llm_none_response_stops(self, extractor, sample_chapter) -> None:
        """LLM extraction stops on None response."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.output_parsed = None
        extractor.client.responses.parse.return_value = response

        recipes = await extractor._extract_with_llm(sample_chapter)

        assert len(recipes) == 0


class TestAsyncChapterExtractorTitleMethods:
    """Tests for title enumeration and verification methods."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with mocked client."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_enumerate_titles_success(self, extractor, sample_chapter) -> None:
        """enumerate_titles returns verified titles."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        response = MagicMock()
        response.output_parsed = ChapterTitles(
            titles=["Roasted Chicken"],  # This title exists in sample_chapter content
            chapter_type="recipes",
        )
        extractor.client.responses.parse.return_value = response

        titles = await extractor.enumerate_titles(sample_chapter)

        assert titles == ["Roasted Chicken"]

    @pytest.mark.asyncio
    async def test_enumerate_titles_filters_unverified(self, extractor, sample_chapter) -> None:
        """enumerate_titles filters out titles not found in content."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        response = MagicMock()
        response.output_parsed = ChapterTitles(
            titles=["Roasted Chicken", "Nonexistent Recipe"],
            chapter_type="recipes",
        )
        extractor.client.responses.parse.return_value = response

        titles = await extractor.enumerate_titles(sample_chapter)

        assert titles == ["Roasted Chicken"]
        assert "Nonexistent Recipe" not in titles

    @pytest.mark.asyncio
    async def test_enumerate_titles_skips_non_recipe_chapters(
        self, extractor, sample_chapter
    ) -> None:
        """enumerate_titles returns empty for non-recipe chapter types."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        for chapter_type in ["intro", "index", "toc", "basics"]:
            response = MagicMock()
            response.output_parsed = ChapterTitles(
                titles=["Some Title"],
                chapter_type=chapter_type,
            )
            extractor.client.responses.parse.return_value = response

            titles = await extractor.enumerate_titles(sample_chapter)

            assert titles == [], f"Should skip {chapter_type} chapters"

    @pytest.mark.asyncio
    async def test_enumerate_titles_handles_api_error(self, extractor, sample_chapter) -> None:
        """enumerate_titles returns empty list on API error."""
        from openai import OpenAIError

        extractor.client.responses.parse.side_effect = OpenAIError("API Error")

        titles = await extractor.enumerate_titles(sample_chapter)

        assert titles == []

    def test_verify_title_exact_match(self, extractor) -> None:
        """_verify_title_in_content finds exact matches."""
        content = "# Roasted Chicken\n\nA delicious recipe."
        assert extractor._verify_title_in_content("Roasted Chicken", content) is True

    def test_verify_title_case_insensitive(self, extractor) -> None:
        """_verify_title_in_content finds case-insensitive matches."""
        content = "# ROASTED CHICKEN\n\nA delicious recipe."
        assert extractor._verify_title_in_content("roasted chicken", content) is True

    def test_verify_title_unicode_normalization(self, extractor) -> None:
        """_verify_title_in_content handles unicode variants."""
        content = "# Café au Lait\n\nFrench coffee."
        assert extractor._verify_title_in_content("Café au Lait", content) is True

    def test_verify_title_special_chars(self, extractor) -> None:
        """_verify_title_in_content handles quote/dash variants."""
        content = "# Mom's Best Pie\n\nClassic recipe."
        # Curly apostrophe in search, straight in content (or vice versa)
        assert extractor._verify_title_in_content("Mom's Best Pie", content) is True

    def test_verify_title_not_found(self, extractor) -> None:
        """_verify_title_in_content returns False for missing titles."""
        content = "# Roasted Chicken\n\nA delicious recipe."
        assert extractor._verify_title_in_content("Grilled Salmon", content) is False

    def test_titles_match_exact(self, extractor) -> None:
        """_titles_match returns True for exact matches after normalization."""
        assert extractor._titles_match("Roasted Chicken", "Roasted Chicken") is True
        assert extractor._titles_match("ROASTED CHICKEN", "roasted chicken") is True

    def test_titles_match_fuzzy(self, extractor) -> None:
        """_titles_match returns True for similar titles."""
        # ~90% similar
        assert extractor._titles_match("Roasted Chicken", "Roasted Chickn") is True

    def test_titles_match_too_different(self, extractor) -> None:
        """_titles_match returns False for very different titles."""
        assert extractor._titles_match("Roasted Chicken", "Grilled Salmon") is False


class TestAsyncChapterExtractorExtractByTitle:
    """Tests for extract_by_title method."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with mocked client."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor
        from mela_parser.config import ExtractionConfig

        config = ExtractionConfig(
            extraction_retry_attempts=1,
            extraction_retry_delay=0.01,
        )
        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            config=config,
            debug=False,
        )

    @pytest.mark.asyncio
    async def test_extract_by_title_success(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """extract_by_title returns recipe on success."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.output_parsed = sample_mela_recipe
        extractor.client.responses.parse.return_value = response

        recipe = await extractor.extract_by_title(sample_chapter, "Roasted Chicken")

        assert recipe is not None
        assert recipe.title == "Roasted Chicken"

    @pytest.mark.asyncio
    async def test_extract_by_title_none_response(self, extractor, sample_chapter) -> None:
        """extract_by_title returns None when API returns None."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.output_parsed = None
        extractor.client.responses.parse.return_value = response

        recipe = await extractor.extract_by_title(sample_chapter, "Test Recipe")

        assert recipe is None

    @pytest.mark.asyncio
    async def test_extract_by_title_retry_on_none(
        self, extractor, sample_chapter, sample_mela_recipe
    ) -> None:
        """extract_by_title retries when receiving None response."""
        from unittest.mock import MagicMock

        none_response = MagicMock()
        none_response.output_parsed = None

        success_response = MagicMock()
        success_response.output_parsed = sample_mela_recipe

        extractor.client.responses.parse.side_effect = [none_response, success_response]

        recipe = await extractor.extract_by_title(sample_chapter, "Roasted Chicken")

        assert recipe is not None
        assert recipe.title == "Roasted Chicken"

    @pytest.mark.asyncio
    async def test_extract_by_title_title_mismatch_warning(
        self, extractor, sample_chapter, caplog
    ) -> None:
        """extract_by_title logs warning on title mismatch."""
        import logging
        from unittest.mock import MagicMock

        from mela_parser.parse import IngredientGroup, MelaRecipe

        # Return a recipe with different title
        wrong_recipe = MelaRecipe(
            title="Wrong Recipe Title",
            ingredients=[IngredientGroup(title="", ingredients=["1 item"])],
            instructions=["Step 1.", "Step 2."],
        )
        response = MagicMock()
        response.output_parsed = wrong_recipe
        extractor.client.responses.parse.return_value = response

        with caplog.at_level(logging.WARNING):
            recipe = await extractor.extract_by_title(sample_chapter, "Roasted Chicken")

        # Recipe is still returned despite mismatch
        assert recipe is not None
        assert "Title mismatch" in caplog.text

    @pytest.mark.asyncio
    async def test_extract_by_title_api_error(self, extractor, sample_chapter) -> None:
        """extract_by_title returns None on API error after retries."""
        from openai import OpenAIError

        extractor.client.responses.parse.side_effect = OpenAIError("API Error")

        recipe = await extractor.extract_by_title(sample_chapter, "Test Recipe")

        assert recipe is None


class TestAsyncChapterExtractorDebugMethods:
    """Tests for debug file saving methods."""

    @pytest.fixture
    def debug_chapter(self):
        """Create a chapter with a simple name for debug testing."""
        from mela_parser.chapter_extractor import Chapter

        return Chapter(
            name="TestChapter",
            content="# Test Recipe\n\nContent here.",
            index=0,
        )

    @pytest.fixture
    def extractor(self, mock_async_openai_client, tmp_path):
        """Create an extractor with debug enabled."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        debug_dir = tmp_path / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=True,
            debug_dir=debug_dir,
        )

    def test_save_debug_input(self, extractor, debug_chapter) -> None:
        """_save_debug_input creates prompt and content files."""
        extractor._save_debug_input(
            debug_chapter, page=1, prompt="Test prompt", content="Test content"
        )

        chapter_dir = extractor.debug_dir / "TestChapter"
        assert chapter_dir.exists()
        assert (chapter_dir / "page_01_prompt.txt").exists()
        assert (chapter_dir / "page_01_content.md").exists()

        prompt_content = (chapter_dir / "page_01_prompt.txt").read_text()
        assert prompt_content == "Test prompt"

    def test_save_debug_output(self, extractor, debug_chapter, sample_mela_recipe) -> None:
        """_save_debug_output creates response JSON file."""
        from unittest.mock import MagicMock

        from mela_parser.parse import CookbookRecipes

        response = MagicMock()
        response.output_parsed = CookbookRecipes(
            recipes=[sample_mela_recipe],
            has_more=False,
        )
        response.usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        extractor._save_debug_output(debug_chapter, page=1, response=response)

        chapter_dir = extractor.debug_dir / "TestChapter"
        response_file = chapter_dir / "page_01_response.json"
        assert response_file.exists()

        import json

        data = json.loads(response_file.read_text())
        assert data["recipe_count"] == 1
        assert data["has_more"] is False

    def test_save_debug_output_none_response(self, extractor, debug_chapter) -> None:
        """_save_debug_output handles None response."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.output_parsed = None

        extractor._save_debug_output(debug_chapter, page=1, response=response)

        chapter_dir = extractor.debug_dir / "TestChapter"
        response_file = chapter_dir / "page_01_response.json"
        assert response_file.exists()

        import json

        data = json.loads(response_file.read_text())
        assert "error" in data

    def test_save_enumeration_debug(self, extractor, debug_chapter) -> None:
        """_save_enumeration_debug creates enumeration JSON file."""
        from unittest.mock import MagicMock

        from mela_parser.parse import ChapterTitles

        response = MagicMock()
        response.output_parsed = ChapterTitles(
            titles=["Recipe A", "Recipe B"],
            chapter_type="recipes",
        )

        extractor._save_enumeration_debug(debug_chapter, response)

        chapter_dir = extractor.debug_dir / "TestChapter"
        enum_file = chapter_dir / "enumeration.json"
        assert enum_file.exists()

        import json

        data = json.loads(enum_file.read_text())
        assert data["titles"] == ["Recipe A", "Recipe B"]
        assert data["title_count"] == 2

    def test_save_grounded_debug(self, extractor, debug_chapter, sample_mela_recipe) -> None:
        """_save_grounded_debug creates recipe-specific JSON file."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.output_parsed = sample_mela_recipe

        extractor._save_grounded_debug(
            debug_chapter, title="Test Recipe", index=0, response=response
        )

        chapter_dir = extractor.debug_dir / "TestChapter"
        recipe_file = chapter_dir / "recipe_00_test_recipe.json"
        assert recipe_file.exists()

    def test_slugify(self, extractor) -> None:
        """_slugify converts text to safe filename."""
        assert extractor._slugify("Roasted Chicken") == "roasted_chicken"
        # Trailing special chars get stripped after conversion
        assert extractor._slugify("Mom's Best Pie!") == "mom_s_best_pie"
        # Long titles are truncated
        long_title = "A" * 100
        assert len(extractor._slugify(long_title)) <= 50

    def test_debug_methods_noop_when_disabled(self, mock_async_openai_client) -> None:
        """Debug methods do nothing when debug is disabled."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor, Chapter

        extractor = AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )

        chapter = Chapter(name="Test", content="Content", index=0)

        # These should not raise or create files
        extractor._save_debug_input(chapter, 1, "prompt", "content")
        extractor._save_debug_output(chapter, 1, None)

        assert extractor.debug_dir is None


class TestAsyncChapterExtractorPromptBuilding:
    """Tests for prompt building methods."""

    @pytest.fixture
    def extractor(self, mock_async_openai_client):
        """Create an extractor with mocked client."""
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        return AsyncChapterExtractor(
            client=mock_async_openai_client,
            debug=False,
        )

    def test_build_pagination_prompt_initial(self, extractor) -> None:
        """_build_pagination_prompt builds initial extraction prompt."""
        content = "Recipe content here"
        prompt = extractor._build_pagination_prompt(content, is_continuation=False)

        assert "<instructions>" in prompt
        assert "<rules>" in prompt
        assert content in prompt
        assert "previous_marker" not in prompt

    def test_build_pagination_prompt_continuation(self, extractor) -> None:
        """_build_pagination_prompt builds continuation prompt."""
        content = "More recipe content"
        marker = "last recipe ended here"
        prompt = extractor._build_pagination_prompt(
            content, is_continuation=True, last_marker=marker
        )

        assert "<previous_marker>" in prompt
        assert marker in prompt
        assert "Continue extracting" in prompt

    def test_build_enumeration_prompt(self, extractor) -> None:
        """_build_enumeration_prompt builds title enumeration prompt."""
        content = "Chapter content with recipes"
        prompt = extractor._build_enumeration_prompt(content)

        assert "<instructions>" in prompt
        assert "List all recipe titles" in prompt
        assert content in prompt

    def test_build_grounded_extraction_prompt(self, extractor) -> None:
        """_build_grounded_extraction_prompt builds title-specific prompt."""
        title = "Roasted Chicken"
        content = "Full chapter content"
        prompt = extractor._build_grounded_extraction_prompt(title, content)

        assert title in prompt
        assert "Extract ONLY this one recipe" in prompt
        assert content in prompt

    def test_find_continuation_point_marker(self, extractor) -> None:
        """_find_continuation_point finds marker position."""
        content = "Start of content. Marker here. Rest of content."
        pos = extractor._find_continuation_point(content, "Marker here.")

        assert pos > 0
        # Should be after the marker
        assert content[pos:].startswith(" Rest of content.")

    def test_find_continuation_point_title_fallback(self, extractor) -> None:
        """_find_continuation_point uses title fallback."""
        content = "## Recipe One\nContent\n\n## Recipe Two\nMore content"
        pos = extractor._find_continuation_point(content, "Recipe One", is_title_fallback=True)

        assert pos > 0
        # Should find the next heading
        assert content[pos:].startswith("## Recipe Two")

    def test_find_continuation_point_not_found(self, extractor) -> None:
        """_find_continuation_point returns -1 when marker not found."""
        content = "Some content without the marker"
        pos = extractor._find_continuation_point(content, "nonexistent marker")

        assert pos == -1
