"""Unit tests for mela_parser.toc_extractor module.

Tests TOC/Index chapter finding, prompt building, and parsing.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from mela_parser.parse import CookbookTOC, TOCEntry
from mela_parser.toc_extractor import IndexRecipes, TOCExtractor


# Simple Chapter mock for testing (matches the real Chapter dataclass)
@dataclass
class MockChapter:
    """Mock Chapter for testing without importing from chapter_extractor."""

    name: str
    content: str
    index: int


class TestIndexRecipesModel:
    """Tests for IndexRecipes Pydantic model."""

    def test_default_empty_list(self) -> None:
        """IndexRecipes defaults to empty list."""
        index = IndexRecipes()
        assert index.recipe_titles == []

    def test_with_titles(self) -> None:
        """IndexRecipes stores recipe titles."""
        titles = ["Chocolate Cake", "Apple Pie", "Bread Pudding"]
        index = IndexRecipes(recipe_titles=titles)
        assert index.recipe_titles == titles
        assert len(index.recipe_titles) == 3

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields raise validation error."""
        with pytest.raises(ValidationError):
            IndexRecipes(recipe_titles=[], extra_field="not allowed")  # type: ignore[call-arg]


class TestTOCExtractorInit:
    """Tests for TOCExtractor initialization."""

    def test_init_with_defaults(self) -> None:
        """TOCExtractor initializes with default model."""
        # Note: This creates a real AsyncOpenAI client
        extractor = TOCExtractor()
        assert extractor.model == "gpt-5-nano"
        assert extractor.client is not None

    def test_init_with_custom_model(self) -> None:
        """TOCExtractor accepts custom model."""
        extractor = TOCExtractor(model="gpt-4o")
        assert extractor.model == "gpt-4o"

    def test_init_with_provided_client(self) -> None:
        """TOCExtractor uses provided client."""
        mock_client = MagicMock()
        extractor = TOCExtractor(client=mock_client, model="custom-model")
        assert extractor.client is mock_client
        assert extractor.model == "custom-model"


class TestTOCExtractorConstants:
    """Tests for TOCExtractor class constants."""

    def test_toc_patterns_exist(self) -> None:
        """TOC_PATTERNS contains expected patterns."""
        assert "table of contents" in TOCExtractor.TOC_PATTERNS
        assert "contents" in TOCExtractor.TOC_PATTERNS
        assert "toc" in TOCExtractor.TOC_PATTERNS

    def test_index_patterns_exist(self) -> None:
        """INDEX_PATTERNS contains expected patterns."""
        assert "index" in TOCExtractor.INDEX_PATTERNS
        assert "recipe index" in TOCExtractor.INDEX_PATTERNS
        assert "alphabetical index" in TOCExtractor.INDEX_PATTERNS

    def test_default_model(self) -> None:
        """DEFAULT_MODEL is set correctly."""
        assert TOCExtractor.DEFAULT_MODEL == "gpt-5-nano"


class TestFindTOCChapter:
    """Tests for _find_toc_chapter method."""

    @pytest.fixture
    def extractor(self) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=MagicMock())

    def test_finds_table_of_contents(self, extractor: TOCExtractor) -> None:
        """Finds chapter named 'Table of Contents'."""
        chapters = [
            MockChapter(name="Introduction", content="intro", index=0),
            MockChapter(name="Table of Contents", content="toc content", index=1),
            MockChapter(name="Recipes", content="recipes", index=2),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "Table of Contents"

    def test_finds_contents(self, extractor: TOCExtractor) -> None:
        """Finds chapter named 'Contents'."""
        chapters = [
            MockChapter(name="Contents", content="toc", index=0),
            MockChapter(name="Chapter 1", content="ch1", index=1),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "Contents"

    def test_finds_toc_case_insensitive(self, extractor: TOCExtractor) -> None:
        """Pattern matching is case-insensitive."""
        chapters = [
            MockChapter(name="TABLE OF CONTENTS", content="toc", index=0),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None

    def test_finds_partial_match(self, extractor: TOCExtractor) -> None:
        """Finds chapters with partial pattern match."""
        chapters = [
            MockChapter(name="Book Contents and Index", content="toc", index=0),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "Book Contents and Index"

    def test_returns_first_match(self, extractor: TOCExtractor) -> None:
        """Returns first matching chapter."""
        chapters = [
            MockChapter(name="First Contents", content="first", index=0),
            MockChapter(name="Second Contents", content="second", index=1),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "First Contents"

    def test_returns_none_when_not_found(self, extractor: TOCExtractor) -> None:
        """Returns None when no TOC chapter found."""
        chapters = [
            MockChapter(name="Introduction", content="intro", index=0),
            MockChapter(name="Recipes", content="recipes", index=1),
        ]
        result = extractor._find_toc_chapter(chapters)  # type: ignore[arg-type]
        assert result is None

    def test_handles_empty_list(self, extractor: TOCExtractor) -> None:
        """Returns None for empty chapter list."""
        result = extractor._find_toc_chapter([])
        assert result is None


class TestFindIndexChapter:
    """Tests for _find_index_chapter method."""

    @pytest.fixture
    def extractor(self) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=MagicMock())

    def test_finds_index(self, extractor: TOCExtractor) -> None:
        """Finds chapter named 'Index'."""
        chapters = [
            MockChapter(name="Recipes", content="recipes", index=0),
            MockChapter(name="Index", content="index content", index=1),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "Index"

    def test_finds_recipe_index(self, extractor: TOCExtractor) -> None:
        """Finds chapter named 'Recipe Index'."""
        chapters = [
            MockChapter(name="Recipe Index", content="index", index=0),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        assert result.name == "Recipe Index"

    def test_finds_alphabetical_index(self, extractor: TOCExtractor) -> None:
        """Finds chapter named 'Alphabetical Index'."""
        chapters = [
            MockChapter(name="Alphabetical Index", content="alpha", index=0),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None

    def test_searches_from_end(self, extractor: TOCExtractor) -> None:
        """Searches from end (index typically at back of book)."""
        chapters = [
            MockChapter(name="Index of Terms", content="terms", index=0),
            MockChapter(name="Recipes", content="recipes", index=1),
            MockChapter(name="Recipe Index", content="recipe index", index=2),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None
        # Should find "Recipe Index" (from end) before "Index of Terms"
        assert result.name == "Recipe Index"

    def test_case_insensitive(self, extractor: TOCExtractor) -> None:
        """Pattern matching is case-insensitive."""
        chapters = [
            MockChapter(name="RECIPE INDEX", content="index", index=0),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is not None

    def test_returns_none_when_not_found(self, extractor: TOCExtractor) -> None:
        """Returns None when no index chapter found."""
        chapters = [
            MockChapter(name="Recipes", content="recipes", index=0),
            MockChapter(name="Conclusion", content="end", index=1),
        ]
        result = extractor._find_index_chapter(chapters)  # type: ignore[arg-type]
        assert result is None

    def test_handles_empty_list(self, extractor: TOCExtractor) -> None:
        """Returns None for empty chapter list."""
        result = extractor._find_index_chapter([])
        assert result is None


class TestBuildTOCPrompt:
    """Tests for _build_toc_prompt method."""

    @pytest.fixture
    def extractor(self) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=MagicMock())

    def test_includes_content(self, extractor: TOCExtractor) -> None:
        """Prompt includes the provided content."""
        content = "Chapter 1: Appetizers\n- Bruschetta\n- Soup"
        prompt = extractor._build_toc_prompt(content)
        assert content in prompt

    def test_includes_instructions_tag(self, extractor: TOCExtractor) -> None:
        """Prompt includes instructions section."""
        prompt = extractor._build_toc_prompt("test content")
        assert "<instructions>" in prompt
        assert "</instructions>" in prompt

    def test_includes_rules_tag(self, extractor: TOCExtractor) -> None:
        """Prompt includes rules section."""
        prompt = extractor._build_toc_prompt("test content")
        assert "<rules>" in prompt
        assert "</rules>" in prompt

    def test_includes_content_tag(self, extractor: TOCExtractor) -> None:
        """Prompt wraps content in tags."""
        prompt = extractor._build_toc_prompt("my content")
        assert "<content>" in prompt
        assert "</content>" in prompt
        assert "<content>\nmy content\n</content>" in prompt

    def test_mentions_chapter_extraction(self, extractor: TOCExtractor) -> None:
        """Prompt mentions extracting by chapter."""
        prompt = extractor._build_toc_prompt("test")
        assert "chapter" in prompt.lower()


class TestBuildIndexPrompt:
    """Tests for _build_index_prompt method."""

    @pytest.fixture
    def extractor(self) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=MagicMock())

    def test_includes_content(self, extractor: TOCExtractor) -> None:
        """Prompt includes the provided content."""
        content = "A\nApple Pie, 45\nB\nBread, 67"
        prompt = extractor._build_index_prompt(content)
        assert content in prompt

    def test_includes_instructions_tag(self, extractor: TOCExtractor) -> None:
        """Prompt includes instructions section."""
        prompt = extractor._build_index_prompt("test")
        assert "<instructions>" in prompt
        assert "</instructions>" in prompt

    def test_includes_rules_tag(self, extractor: TOCExtractor) -> None:
        """Prompt includes rules section."""
        prompt = extractor._build_index_prompt("test")
        assert "<rules>" in prompt
        assert "</rules>" in prompt

    def test_mentions_recipe_titles(self, extractor: TOCExtractor) -> None:
        """Prompt mentions extracting recipe titles."""
        prompt = extractor._build_index_prompt("test")
        assert "recipe" in prompt.lower()
        assert "title" in prompt.lower()

    def test_excludes_page_numbers_rule(self, extractor: TOCExtractor) -> None:
        """Prompt instructs to exclude page numbers."""
        prompt = extractor._build_index_prompt("test")
        assert "page number" in prompt.lower()


class TestExtractTOCAsync:
    """Tests for extract_toc async method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create mock OpenAI client."""
        client = MagicMock()
        client.responses = MagicMock()
        client.responses.parse = AsyncMock()
        return client

    @pytest.fixture
    def extractor(self, mock_client: MagicMock) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=mock_client)

    async def test_returns_empty_when_no_toc(self, extractor: TOCExtractor) -> None:
        """Returns empty CookbookTOC when no TOC chapter found."""
        chapters = [MockChapter(name="Recipes", content="recipes", index=0)]
        result = await extractor.extract_toc(chapters)  # type: ignore[arg-type]

        assert isinstance(result, CookbookTOC)
        assert result.chapters == []

    async def test_calls_parse_on_toc_chapter(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Calls OpenAI parse when TOC chapter found."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_parsed = CookbookTOC(
            chapters=[TOCEntry(chapter_title="Mains", recipes=["Chicken", "Beef"])]
        )
        mock_client.responses.parse.return_value = mock_response

        chapters = [MockChapter(name="Table of Contents", content="toc data", index=0)]
        result = await extractor.extract_toc(chapters)  # type: ignore[arg-type]

        assert mock_client.responses.parse.called
        assert len(result.chapters) == 1
        assert result.chapters[0].chapter_title == "Mains"

    async def test_returns_empty_on_none_response(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Returns empty TOC when parse returns None."""
        mock_response = MagicMock()
        mock_response.output_parsed = None
        mock_client.responses.parse.return_value = mock_response

        chapters = [MockChapter(name="Contents", content="toc", index=0)]
        result = await extractor.extract_toc(chapters)  # type: ignore[arg-type]

        assert result.chapters == []

    async def test_returns_empty_on_exception(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Returns empty TOC when parsing raises exception."""
        mock_client.responses.parse.side_effect = Exception("API error")

        chapters = [MockChapter(name="Contents", content="toc", index=0)]
        result = await extractor.extract_toc(chapters)  # type: ignore[arg-type]

        assert result.chapters == []


class TestExtractIndexTitlesAsync:
    """Tests for extract_index_titles async method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create mock OpenAI client."""
        client = MagicMock()
        client.responses = MagicMock()
        client.responses.parse = AsyncMock()
        return client

    @pytest.fixture
    def extractor(self, mock_client: MagicMock) -> TOCExtractor:
        """Create extractor with mock client."""
        return TOCExtractor(client=mock_client)

    async def test_returns_empty_when_no_index(self, extractor: TOCExtractor) -> None:
        """Returns empty list when no index chapter found."""
        chapters = [MockChapter(name="Recipes", content="recipes", index=0)]
        result = await extractor.extract_index_titles(chapters)  # type: ignore[arg-type]

        assert result == []

    async def test_calls_parse_on_index_chapter(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Calls OpenAI parse when index chapter found."""
        mock_response = MagicMock()
        mock_response.output_parsed = IndexRecipes(recipe_titles=["Apple Pie", "Beef Stew", "Cake"])
        mock_client.responses.parse.return_value = mock_response

        chapters = [MockChapter(name="Recipe Index", content="index data", index=0)]
        result = await extractor.extract_index_titles(chapters)  # type: ignore[arg-type]

        assert mock_client.responses.parse.called
        assert len(result) == 3
        assert "Apple Pie" in result

    async def test_returns_empty_on_none_response(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Returns empty list when parse returns None."""
        mock_response = MagicMock()
        mock_response.output_parsed = None
        mock_client.responses.parse.return_value = mock_response

        chapters = [MockChapter(name="Index", content="index", index=0)]
        result = await extractor.extract_index_titles(chapters)  # type: ignore[arg-type]

        assert result == []

    async def test_returns_empty_on_exception(
        self, extractor: TOCExtractor, mock_client: MagicMock
    ) -> None:
        """Returns empty list when parsing raises exception."""
        mock_client.responses.parse.side_effect = RuntimeError("Network error")

        chapters = [MockChapter(name="Index", content="index", index=0)]
        result = await extractor.extract_index_titles(chapters)  # type: ignore[arg-type]

        assert result == []
