"""Pytest configuration and fixtures for mela_parser tests.

This module provides shared fixtures for testing the mela_parser package.
Fixtures follow pytest best practices:
- Use yield for cleanup
- Use monkeypatch for environment manipulation
- Use tmp_path for file operations
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all MELA_PARSER_* environment variables.

    Use this fixture when testing configuration loading to ensure
    no environment variables interfere with test expectations.
    """
    import os

    for key in list(os.environ.keys()):
        if key.startswith("MELA_PARSER_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide a helper to set MELA_PARSER_* environment variables.

    Returns a dict that, when populated, sets the corresponding env vars.

    Example:
        def test_env_loading(mock_env):
            mock_env["MODEL"] = "gpt-5-mini"
            # MELA_PARSER_MODEL is now set
    """

    class EnvSetter(dict[str, str]):
        def __setitem__(self, key: str, value: str) -> None:
            super().__setitem__(key, value)
            monkeypatch.setenv(f"MELA_PARSER_{key}", value)

    return EnvSetter()


@pytest.fixture
def default_config():
    """Create a default ExtractionConfig instance."""
    from mela_parser.config import ExtractionConfig

    return ExtractionConfig()


@pytest.fixture
def config_dict() -> dict[str, Any]:
    """Provide valid configuration values as a dictionary."""
    return {
        "model": "gpt-5-nano",
        "temperature": 0.0,
        "max_concurrent": 200,
        "retry_attempts": 3,
        "initial_retry_delay": 1.0,
        "extract_images": True,
        "min_image_area": 300000,
        "max_image_width": 600,
        "debug_mode": False,
        "min_ingredients": 1,
        "min_instructions": 2,
        "similarity_threshold": 0.90,
    }


# ============================================================================
# Recipe/Model Fixtures
# ============================================================================


@pytest.fixture
def sample_recipe_dict() -> dict[str, Any]:
    """Provide a sample recipe as a dictionary for testing."""
    return {
        "title": "Test Recipe",
        "recipeYield": "4 servings",
        "totalTime": "30 minutes",
        "prepTime": "10 minutes",
        "cookTime": "20 minutes",
        "ingredients": ["1 cup flour", "2 eggs", "1/2 cup milk"],
        "instructions": "Mix all ingredients. Cook until done.",
        "notes": "This is a test recipe.",
        "nutrition": "200 calories per serving",
        "link": "",
        "categories": ["Main Course"],
        "images": [],
    }


# ============================================================================
# Async Test Helpers
# ============================================================================


@pytest.fixture
def event_loop_policy():
    """Provide event loop policy for async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


# ============================================================================
# EPUB Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_epub_item():
    """Create a mock EPUB item (chapter)."""
    from unittest.mock import MagicMock

    item = MagicMock()
    item.get_content.return_value = (
        b"<html><body><h1>Test Recipe</h1><p>Instructions here.</p></body></html>"
    )
    item.get_name.return_value = "chapter1.xhtml"
    return item


@pytest.fixture
def mock_epub_book(mock_epub_item):
    """Create a mock EpubBook with test items."""
    from unittest.mock import MagicMock

    book = MagicMock()
    book.get_items_of_type.return_value = [mock_epub_item]
    book.get_item_with_href.return_value = mock_epub_item
    return book


@pytest.fixture
def mock_markitdown(monkeypatch: pytest.MonkeyPatch):
    """Mock MarkItDown converter."""
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "# Test Recipe\n\nInstructions here."

    mock_md = MagicMock()
    mock_md.convert_stream.return_value = mock_result

    def mock_init(*args, **kwargs):
        return mock_md

    monkeypatch.setattr("markitdown.MarkItDown", mock_init)
    return mock_md


@pytest.fixture
def mock_epub_read(monkeypatch: pytest.MonkeyPatch, mock_epub_book):
    """Mock ebooklib.epub.read_epub to return mock book."""
    monkeypatch.setattr(
        "ebooklib.epub.read_epub",
        lambda *args, **kwargs: mock_epub_book,
    )
    return mock_epub_book


# ============================================================================
# OpenAI Mocking Fixtures (using pytest-httpx)
# ============================================================================


@pytest.fixture
def sample_chapter():
    """Create a sample Chapter for testing."""
    from mela_parser.chapter_extractor import Chapter

    return Chapter(
        name="Main Dishes",
        content="# Roasted Chicken\n\nA delicious recipe.\n\n## Ingredients\n- 1 chicken\n- Salt",
        index=0,
    )


@pytest.fixture
def sample_mela_recipe():
    """Create a sample MelaRecipe for testing."""
    from mela_parser.parse import IngredientGroup, MelaRecipe

    return MelaRecipe(
        title="Roasted Chicken",
        text="A delicious roasted chicken recipe.",
        recipeYield="Serves 4",
        prepTime=15,
        cookTime=60,
        totalTime=75,
        ingredients=[
            IngredientGroup(
                title="Main",
                ingredients=["1 whole chicken", "2 tbsp olive oil", "Salt and pepper"],
            ),
        ],
        instructions=["Preheat oven to 400F.", "Season chicken.", "Roast for 1 hour."],
        notes="Let rest before carving.",
    )


# ============================================================================
# OpenAI Client Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_async_openai_client():
    """Create a mock AsyncOpenAI client for extraction tests.

    Returns an AsyncMock that can be configured in individual tests
    to return specific responses.
    """
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.parse = AsyncMock()
    return client


@pytest.fixture
def mock_sync_openai_client():
    """Create a mock synchronous OpenAI client for parser tests.

    Returns a MagicMock that can be configured in individual tests
    to return specific responses.
    """
    from unittest.mock import MagicMock

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.parse = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client


@pytest.fixture
def mock_chapter_titles_response(sample_mela_recipe):
    """Create a mock response for title enumeration (ChapterTitles)."""
    from unittest.mock import MagicMock

    from mela_parser.parse import ChapterTitles

    response = MagicMock()
    response.output_parsed = ChapterTitles(
        titles=["Roasted Chicken", "Grilled Salmon"],
        chapter_type="recipes",
    )
    return response


@pytest.fixture
def mock_cookbook_recipes_response(sample_mela_recipe):
    """Create a mock response for cookbook extraction (CookbookRecipes)."""
    from unittest.mock import MagicMock

    from mela_parser.parse import CookbookRecipes

    response = MagicMock()
    response.output_parsed = CookbookRecipes(
        recipes=[sample_mela_recipe],
        has_more=False,
        last_content_marker=None,
    )
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    response.usage.total_tokens = 150
    return response


@pytest.fixture
def mock_recipe_response(sample_mela_recipe):
    """Create a mock response for single recipe extraction (MelaRecipe)."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.output_parsed = sample_mela_recipe
    return response
