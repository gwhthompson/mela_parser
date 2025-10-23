#!/usr/bin/env python3
"""
Tests for the LLM-powered recipe extraction pipeline v2.

These tests verify:
1. Data models and serialization
2. Chapter processing
3. Recipe list discovery
4. Extraction logic
5. Validation
6. Prompt improvement
7. End-to-end pipeline
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from scripts.main_chapters_v2 import (
    ChapterExtractor,
    ChapterProcessor,
    ExtractionPipeline,
    ExtractionResult,
    PromptImprovements,
    PromptLibrary,
    RecipeListDiscoverer,
    ValidationReport,
)
from mela_parser.parse import MelaRecipe, IngredientGroup


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_recipes():
    """Sample recipe data for testing."""
    return [
        MelaRecipe(
            title="Chocolate Cake",
            ingredients=[
                IngredientGroup(
                    title="Main",
                    ingredients=["2 cups flour", "1 cup sugar", "1/2 cup cocoa"],
                )
            ],
            instructions=["Mix dry ingredients", "Bake at 350F"],
        ),
        MelaRecipe(
            title="Vanilla Cookies",
            ingredients=[
                IngredientGroup(
                    title="Main",
                    ingredients=["1 cup flour", "1/2 cup butter", "1/4 cup sugar"],
                )
            ],
            instructions=["Cream butter and sugar", "Mix in flour", "Bake"],
        ),
    ]


@pytest.fixture
def sample_chapters():
    """Sample chapter data."""
    return [
        ("chapter1.html", "# Chapter 1\n\n[Chocolate Cake](page1)\n[Vanilla Cookies](page2)"),
        ("chapter2.html", "# Chocolate Cake\n\nIngredients:\n- 2 cups flour\n\nInstructions:\n1. Mix"),
        ("chapter3.html", "# Vanilla Cookies\n\nIngredients:\n- 1 cup flour\n\nInstructions:\n1. Cream"),
    ]


@pytest.fixture
def prompt_library():
    """Default prompt library."""
    return PromptLibrary.default()


# ============================================================================
# TEST DATA MODELS
# ============================================================================


def test_prompt_library_serialization():
    """Test PromptLibrary serialization and deserialization."""
    original = PromptLibrary(
        discovery_prompt="Test discovery",
        extraction_prompt="Test extraction",
        version=2,
        iteration_history=[{"version": 2, "changes": "Improved accuracy"}],
        locked=True,
    )

    # Serialize
    data = original.to_dict()
    assert data["version"] == 2
    assert data["locked"] is True
    assert len(data["iteration_history"]) == 1

    # Deserialize
    restored = PromptLibrary.from_dict(data)
    assert restored.version == 2
    assert restored.locked is True
    assert restored.discovery_prompt == "Test discovery"


def test_prompt_library_default():
    """Test default prompt library creation."""
    prompts = PromptLibrary.default()

    assert "{combined_lists}" in prompts.discovery_prompt
    assert "{expected_list}" in prompts.extraction_prompt
    assert "{chapter_md}" in prompts.extraction_prompt
    assert prompts.version == 1
    assert prompts.locked is False


def test_extraction_result_dataclass(sample_recipes):
    """Test ExtractionResult dataclass."""
    result = ExtractionResult(
        recipes=sample_recipes,
        total_extracted=3,
        unique_count=2,
        duplicates_removed=1,
        chapters_processed=5,
        chapters_with_recipes=3,
        extraction_time=45.2,
        model_used="gpt-5-nano",
        metadata={"book": "Test Cookbook"},
    )

    assert len(result.recipes) == 2
    assert result.total_extracted == 3
    assert result.unique_count == 2
    assert result.duplicates_removed == 1
    assert result.model_used == "gpt-5-nano"
    assert result.metadata["book"] == "Test Cookbook"


def test_validation_report():
    """Test ValidationReport creation and methods."""
    extracted = {"Recipe A", "Recipe B", "Recipe C"}
    discovered = {"Recipe A", "Recipe B", "Recipe D"}

    report = ValidationReport(
        extracted_titles=extracted,
        discovered_titles=discovered,
        matched_titles={"Recipe A", "Recipe B"},
        missing_titles={"Recipe D"},
        extra_titles={"Recipe C"},
        match_percentage=66.67,
        total_discovered=3,
        total_extracted=3,
        is_perfect_match=False,
    )

    # Test to_dict
    data = report.to_dict()
    assert "Recipe A" in data["matched_titles"]
    assert "Recipe D" in data["missing_titles"]
    assert "Recipe C" in data["extra_titles"]
    assert data["match_percentage"] == 66.67
    assert data["is_perfect_match"] is False


def test_validation_report_perfect_match():
    """Test ValidationReport for perfect match scenario."""
    titles = {"Recipe A", "Recipe B"}

    report = ValidationReport(
        extracted_titles=titles,
        discovered_titles=titles,
        matched_titles=titles,
        missing_titles=set(),
        extra_titles=set(),
        match_percentage=100.0,
        total_discovered=2,
        total_extracted=2,
        is_perfect_match=True,
    )

    assert report.is_perfect_match
    assert report.match_percentage == 100.0
    assert len(report.missing_titles) == 0
    assert len(report.extra_titles) == 0


# ============================================================================
# TEST CHAPTER PROCESSING
# ============================================================================


@pytest.mark.skipif(
    not Path("examples/input").exists(),
    reason="Test EPUB files not available"
)
def test_chapter_processor_integration():
    """Integration test for ChapterProcessor with real EPUB."""
    processor = ChapterProcessor()

    # Find any test EPUB
    epub_files = list(Path("examples/input").glob("*.epub"))
    if not epub_files:
        pytest.skip("No EPUB files found in examples/input")

    epub_path = str(epub_files[0])

    # Convert
    book, chapters = processor.convert_epub_by_chapters(epub_path)

    assert book is not None
    assert len(chapters) > 0
    assert all(isinstance(name, str) and isinstance(md, str) for name, md in chapters)


# ============================================================================
# TEST RECIPE LIST DISCOVERY
# ============================================================================


def test_recipe_list_discoverer_finds_links(sample_chapters, prompt_library):
    """Test that RecipeListDiscoverer identifies link sections."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.output_parsed.titles = ["Chocolate Cake", "Vanilla Cookies"]
    mock_client.responses.parse.return_value = mock_response

    discoverer = RecipeListDiscoverer(mock_client, model="gpt-5-mini")

    # Discover
    titles = discoverer.discover_recipe_list(sample_chapters, prompt_library.discovery_prompt)

    assert titles is not None
    assert len(titles) == 2
    assert "Chocolate Cake" in titles
    assert "Vanilla Cookies" in titles

    # Verify API was called
    mock_client.responses.parse.assert_called_once()


def test_recipe_list_discoverer_no_links():
    """Test behavior when no recipe lists are found."""
    chapters = [("chapter1.html", "Just plain text with no links.")]

    mock_client = Mock()
    discoverer = RecipeListDiscoverer(mock_client, model="gpt-5-mini")

    titles = discoverer.discover_recipe_list(chapters, "{combined_lists}")

    assert titles is None
    mock_client.responses.parse.assert_not_called()


# ============================================================================
# TEST VALIDATION
# ============================================================================


def test_validation_exact_match(sample_recipes):
    """Test validation with exact match."""
    result = ExtractionResult(
        recipes=sample_recipes,
        total_extracted=2,
        unique_count=2,
        duplicates_removed=0,
        chapters_processed=3,
        chapters_with_recipes=2,
        extraction_time=10.0,
        model_used="gpt-5-nano",
    )

    discovered = ["Chocolate Cake", "Vanilla Cookies"]

    pipeline = ExtractionPipeline()
    report = pipeline.validate_extraction(result, discovered)

    assert report.is_perfect_match
    assert report.match_percentage == 100.0
    assert len(report.missing_titles) == 0
    assert len(report.extra_titles) == 0


def test_validation_with_missing(sample_recipes):
    """Test validation with missing recipes."""
    result = ExtractionResult(
        recipes=sample_recipes[:1],  # Only first recipe
        total_extracted=1,
        unique_count=1,
        duplicates_removed=0,
        chapters_processed=3,
        chapters_with_recipes=1,
        extraction_time=10.0,
        model_used="gpt-5-nano",
    )

    discovered = ["Chocolate Cake", "Vanilla Cookies", "Strawberry Pie"]

    pipeline = ExtractionPipeline()
    report = pipeline.validate_extraction(result, discovered)

    assert not report.is_perfect_match
    assert report.match_percentage < 100.0
    assert "Vanilla Cookies" in report.missing_titles
    assert "Strawberry Pie" in report.missing_titles
    assert "Chocolate Cake" in report.matched_titles


def test_validation_with_extras(sample_recipes):
    """Test validation with extra recipes."""
    # Add an extra recipe not in discovered list
    extra_recipe = MelaRecipe(
        title="Strawberry Pie",
        ingredients=[
            IngredientGroup(
                title="Main",
                ingredients=["Strawberries"],
            )
        ],
        instructions=["Make pie"],
    )

    result = ExtractionResult(
        recipes=sample_recipes + [extra_recipe],
        total_extracted=3,
        unique_count=3,
        duplicates_removed=0,
        chapters_processed=3,
        chapters_with_recipes=3,
        extraction_time=10.0,
        model_used="gpt-5-nano",
    )

    discovered = ["Chocolate Cake", "Vanilla Cookies"]

    pipeline = ExtractionPipeline()
    report = pipeline.validate_extraction(result, discovered)

    assert not report.is_perfect_match
    assert "Strawberry Pie" in report.extra_titles
    assert len(report.missing_titles) == 0


# ============================================================================
# TEST ASYNC EXTRACTION
# ============================================================================


@pytest.mark.asyncio
async def test_chapter_extractor_basic():
    """Test basic chapter extraction with mocked API."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.output_parsed.recipes = [
        MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="Main", ingredients=["1 cup flour"])],
            instructions=["Mix and bake"],
        )
    ]
    mock_client.responses.parse.return_value = mock_response

    extractor = ChapterExtractor(mock_client, model="gpt-5-nano")
    semaphore = asyncio.Semaphore(5)

    chapter_md = "# Test Recipe\n\nIngredients:\n- 1 cup flour\n\nInstructions:\n1. Mix and bake"
    expected_titles = ["Test Recipe"]

    recipes = await extractor.extract_from_chapter(
        chapter_md,
        "chapter1.html",
        expected_titles,
        "Extract: {expected_list}\n{chapter_md}",
        semaphore,
    )

    assert len(recipes) == 1
    assert recipes[0].title == "Test Recipe"


@pytest.mark.asyncio
async def test_chapter_extractor_no_expected_titles():
    """Test extraction when no expected titles provided."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.output_parsed.recipes = []
    mock_client.responses.parse.return_value = mock_response

    extractor = ChapterExtractor(mock_client, model="gpt-5-nano")
    semaphore = asyncio.Semaphore(5)

    chapter_md = "Just some text without recipes"

    recipes = await extractor.extract_from_chapter(
        chapter_md,
        "chapter1.html",
        None,  # No expected titles
        "{chapter_md}",
        semaphore,
    )

    assert len(recipes) == 0


@pytest.mark.asyncio
async def test_chapter_extractor_retry_logic():
    """Test exponential backoff retry on API failure."""
    mock_client = AsyncMock()

    # Fail twice, then succeed
    mock_client.responses.parse.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        Mock(output_parsed=Mock(recipes=[])),
    ]

    extractor = ChapterExtractor(mock_client, model="gpt-5-nano")
    semaphore = asyncio.Semaphore(5)

    recipes = await extractor.extract_from_chapter(
        "test content",
        "chapter1.html",
        ["Test"],
        "{expected_list}\n{chapter_md}",
        semaphore,
    )

    # Should succeed after retries
    assert isinstance(recipes, list)
    assert mock_client.responses.parse.call_count == 3


# ============================================================================
# TEST PROMPT IMPROVEMENT
# ============================================================================


@pytest.mark.asyncio
async def test_analyze_gaps_basic():
    """Test gap analysis with mocked LLM."""
    pipeline = ExtractionPipeline()

    validation_report = ValidationReport(
        extracted_titles={"Recipe A"},
        discovered_titles={"Recipe A", "Recipe B"},
        matched_titles={"Recipe A"},
        missing_titles={"Recipe B"},
        extra_titles=set(),
        match_percentage=50.0,
        total_discovered=2,
        total_extracted=1,
        is_perfect_match=False,
    )

    chapters = [
        ("ch1.html", "Recipe list: A, B"),
        ("ch2.html", "# Recipe A\nIngredients: flour\nSteps: mix"),
        ("ch3.html", "# Recipe B\nIngredients: sugar\nSteps: bake"),
    ]

    prompts = PromptLibrary.default()

    # Mock the async client
    with patch.object(pipeline, 'async_client') as mock_client:
        mock_response = Mock()
        mock_response.output_parsed.analysis = "Recipe B was missed because..."
        mock_response.output_parsed.missing_patterns = ["Missing recipes in chapter 3"]
        mock_response.output_parsed.false_positive_patterns = []
        mock_response.output_parsed.discovery_improvements = "Be more specific"
        mock_response.output_parsed.extraction_improvements = "Look harder"
        mock_response.output_parsed.confidence = 0.8
        mock_response.output_parsed.reasoning = "Based on patterns..."

        mock_client.responses.parse.return_value = mock_response

        improvements = await pipeline.analyze_gaps(validation_report, chapters, prompts)

        assert improvements.confidence == 0.8
        assert "Recipe B was missed" in improvements.analysis
        assert len(improvements.missing_recipe_patterns) > 0


# ============================================================================
# TEST UTILITIES
# ============================================================================


def test_extract_snippet():
    """Test recipe snippet extraction."""
    pipeline = ExtractionPipeline()

    chapter_md = "Some intro text. Here is the Chocolate Cake recipe with ingredients and steps. More text."
    snippet = pipeline._extract_recipe_snippet(chapter_md, "Chocolate Cake", context_chars=40)

    assert "Chocolate Cake" in snippet
    assert snippet.startswith("...")
    assert snippet.endswith("...")
    assert len(snippet) < len(chapter_md)


def test_extract_snippet_not_found():
    """Test snippet extraction when title not found."""
    pipeline = ExtractionPipeline()

    chapter_md = "Some text without the recipe"
    snippet = pipeline._extract_recipe_snippet(chapter_md, "Missing Recipe")

    assert "[Recipe title not found in chapter]" in snippet


# ============================================================================
# TEST END-TO-END (MOCKED)
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_perfect_match():
    """Test full pipeline with perfect match (mocked)."""
    pipeline = ExtractionPipeline()

    # Mock all components
    with patch.object(pipeline.chapter_processor, 'convert_epub_by_chapters') as mock_convert, \
         patch('main_chapters_v2.RecipeListDiscoverer') as mock_discoverer_class, \
         patch('main_chapters_v2.ChapterExtractor') as mock_extractor_class:

        # Mock chapter conversion
        mock_book = Mock()
        mock_chapters = [
            ("ch1.html", "[Recipe A](page1)\n[Recipe B](page2)"),
            ("ch2.html", "# Recipe A\nIngredients: flour"),
            ("ch3.html", "# Recipe B\nIngredients: sugar"),
        ]
        mock_convert.return_value = (mock_book, mock_chapters)

        # Mock discoverer
        mock_discoverer = Mock()
        mock_discoverer.discover_recipe_list.return_value = ["Recipe A", "Recipe B"]
        mock_discoverer_class.return_value = mock_discoverer

        # Mock extractor
        mock_extractor = AsyncMock()
        recipe_a = MelaRecipe(
            title="Recipe A",
            ingredients=[IngredientGroup(title="Main", ingredients=["flour"])],
            instructions=["Mix"],
        )
        recipe_b = MelaRecipe(
            title="Recipe B",
            ingredients=[IngredientGroup(title="Main", ingredients=["sugar"])],
            instructions=["Bake"],
        )
        mock_extractor.extract_from_chapter.side_effect = [
            [],  # Chapter 1: no recipes (just links)
            [recipe_a],  # Chapter 2
            [recipe_b],  # Chapter 3
        ]
        mock_extractor_class.return_value = mock_extractor

        # Run extraction
        prompts = PromptLibrary.default()
        result, chapters, discovered = await pipeline.extract_recipes("test.epub", prompts)

        # Verify results
        assert len(result.recipes) == 2
        assert result.unique_count == 2
        assert result.duplicates_removed == 0
        assert discovered == ["Recipe A", "Recipe B"]

        # Validate
        validation = pipeline.validate_extraction(result, discovered)
        assert validation.is_perfect_match
        assert validation.match_percentage == 100.0


# ============================================================================
# TEST FILE I/O
# ============================================================================


def test_prompt_library_file_io():
    """Test saving and loading PromptLibrary to/from file."""
    prompts = PromptLibrary(
        discovery_prompt="Test discovery",
        extraction_prompt="Test extraction",
        version=3,
        iteration_history=[
            {"version": 2, "changes": "First improvement"},
            {"version": 3, "changes": "Second improvement"},
        ],
        locked=True,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(prompts.to_dict(), f)
        temp_path = f.name

    try:
        # Load it back
        with open(temp_path, 'r') as f:
            loaded = PromptLibrary.from_dict(json.load(f))

        assert loaded.version == 3
        assert loaded.locked is True
        assert len(loaded.iteration_history) == 2
        assert loaded.discovery_prompt == "Test discovery"
    finally:
        Path(temp_path).unlink()


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
