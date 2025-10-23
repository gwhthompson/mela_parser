#!/usr/bin/env python3
"""
Test suite for chapter_extractor module.

This module provides comprehensive tests and usage examples for the
async chapter-based EPUB recipe extraction framework.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from mela_parser.chapter_extractor import (
    AsyncChapterExtractor,
    Chapter,
    ChapterProcessor,
    EPUBConversionError,
    RecipeListDiscoverer,
    ValidationEngine,
    process_epub_chapters,
)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


# ============================================================================
# ChapterProcessor Tests
# ============================================================================


class TestChapterProcessor:
    """Tests for ChapterProcessor class."""

    def test_init_nonexistent_file(self):
        """Test initialization with nonexistent EPUB file."""
        with pytest.raises(FileNotFoundError):
            ChapterProcessor("nonexistent.epub")

    @pytest.mark.asyncio
    async def test_split_into_chapters(self, sample_epub_path):
        """Test chapter splitting and conversion."""
        processor = ChapterProcessor(sample_epub_path)
        chapters = await processor.split_into_chapters()

        assert isinstance(chapters, list)
        assert len(chapters) > 0

        # Validate chapter structure
        for chapter in chapters:
            assert isinstance(chapter, Chapter)
            assert chapter.name
            assert chapter.content
            assert chapter.index >= 0

    @pytest.mark.asyncio
    async def test_chapter_content_quality(self, sample_epub_path):
        """Test that converted chapters have meaningful content."""
        processor = ChapterProcessor(sample_epub_path)
        chapters = await processor.split_into_chapters()

        # At least one chapter should have substantial content
        assert any(len(ch.content) > 500 for ch in chapters)

        # Chapters should contain markdown-like content
        assert any("recipe" in ch.content.lower() for ch in chapters)


# ============================================================================
# RecipeListDiscoverer Tests
# ============================================================================


class TestRecipeListDiscoverer:
    """Tests for RecipeListDiscoverer class."""

    @pytest.mark.asyncio
    async def test_discover_from_chapters_with_links(self):
        """Test discovery from chapters with recipe links."""
        # Create mock chapters with recipe links
        chapters = [
            Chapter(
                name="toc.html",
                content="""
                # Table of Contents

                [Chocolate Cake](recipe1.html)
                [Apple Pie](recipe2.html)
                [Banana Bread](recipe3.html)
                [Carrot Cake](recipe4.html)
                [Lemon Tart](recipe5.html)
                [Berry Muffins](recipe6.html)
                """,
                index=0
            ),
            Chapter(
                name="content.html",
                content="Some recipe content here",
                index=1
            )
        ]

        discoverer = RecipeListDiscoverer()
        titles = await discoverer.discover_from_chapters(chapters)

        # Should discover some recipes (exact count depends on LLM)
        assert titles is not None
        assert len(titles) > 0
        assert isinstance(titles, list)
        assert all(isinstance(t, str) for t in titles)

    @pytest.mark.asyncio
    async def test_discover_from_chapters_no_links(self):
        """Test discovery when no recipe lists present."""
        chapters = [
            Chapter(
                name="intro.html",
                content="This is just an introduction with no recipe links.",
                index=0
            )
        ]

        discoverer = RecipeListDiscoverer()
        titles = await discoverer.discover_from_chapters(chapters)

        # Should return None when no links found
        assert titles is None


# ============================================================================
# AsyncChapterExtractor Tests
# ============================================================================


class TestAsyncChapterExtractor:
    """Tests for AsyncChapterExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_from_chapter_with_recipe(self):
        """Test extraction from chapter containing a recipe."""
        chapter = Chapter(
            name="recipe.html",
            content="""
            # Chocolate Cake

            A delicious chocolate cake recipe.

            **Servings:** 8
            **Prep Time:** 20 minutes
            **Cook Time:** 40 minutes

            ## Ingredients

            - 2 cups all-purpose flour
            - 1 3/4 cups sugar
            - 3/4 cup cocoa powder
            - 2 teaspoons baking soda
            - 1 teaspoon salt
            - 2 eggs
            - 1 cup buttermilk
            - 1 cup strong coffee
            - 1/2 cup vegetable oil

            ## Instructions

            1. Preheat oven to 350°F (175°C).
            2. Mix dry ingredients in a large bowl.
            3. Add eggs, buttermilk, coffee, and oil.
            4. Beat until smooth.
            5. Pour into greased pans.
            6. Bake for 30-35 minutes.
            7. Cool before frosting.
            """,
            index=0
        )

        extractor = AsyncChapterExtractor(model="gpt-5-nano")
        result = await extractor.extract_from_chapter(chapter)

        # Should successfully extract the recipe
        assert result.is_success
        assert result.recipe_count > 0
        assert result.error is None

        # Validate recipe structure
        recipe = result.recipes[0]
        assert "chocolate" in recipe.title.lower()
        assert len(recipe.ingredients) > 0
        assert len(recipe.instructions) > 0

    @pytest.mark.asyncio
    async def test_extract_from_chapters_parallel(self):
        """Test parallel extraction from multiple chapters."""
        # Create multiple chapters
        chapters = [
            Chapter(
                name=f"chapter{i}.html",
                content=f"""
                # Recipe {i}

                ## Ingredients
                - 1 cup ingredient A
                - 2 tablespoons ingredient B

                ## Instructions
                1. Mix ingredients
                2. Cook for 10 minutes
                """,
                index=i
            )
            for i in range(3)
        ]

        extractor = AsyncChapterExtractor(model="gpt-5-nano")
        results = await extractor.extract_from_chapters(chapters, max_concurrent=2)

        # Should process all chapters
        assert len(results) == len(chapters)

        # All should be valid extraction results
        for result in results:
            assert hasattr(result, 'chapter_name')
            assert hasattr(result, 'recipes')
            assert hasattr(result, 'is_success')

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test that retry logic handles failures gracefully."""
        # Create a chapter that might cause issues
        chapter = Chapter(
            name="problematic.html",
            content="Not a recipe at all, just random text.",
            index=0
        )

        extractor = AsyncChapterExtractor(model="gpt-5-nano", max_retries=2)
        result = await extractor.extract_from_chapter(chapter)

        # Should complete without raising exception
        assert isinstance(result.chapter_name, str)
        assert isinstance(result.recipes, list)


# ============================================================================
# ValidationEngine Tests
# ============================================================================


class TestValidationEngine:
    """Tests for ValidationEngine class."""

    def test_create_diff_perfect_match(self):
        """Test diff creation with perfect match."""
        from parse import MelaRecipe, IngredientGroup

        expected = ["Chocolate Cake", "Apple Pie", "Banana Bread"]
        extracted = [
            MelaRecipe(
                title=title,
                ingredients=[IngredientGroup(title="Main", ingredients=["flour"])],
                instructions=["Mix and bake"]
            )
            for title in expected
        ]

        engine = ValidationEngine()
        diff = engine.create_diff(expected, extracted)

        assert diff.is_perfect_match
        assert diff.match_rate == 1.0
        assert len(diff.exact_matches) == 3
        assert len(diff.missing_titles) == 0
        assert len(diff.extra_titles) == 0

    def test_create_diff_with_missing(self):
        """Test diff creation with missing recipes."""
        from parse import MelaRecipe, IngredientGroup

        expected = ["Chocolate Cake", "Apple Pie", "Banana Bread"]
        extracted = [
            MelaRecipe(
                title="Chocolate Cake",
                ingredients=[IngredientGroup(title="Main", ingredients=["flour"])],
                instructions=["Mix and bake"]
            )
        ]

        engine = ValidationEngine()
        diff = engine.create_diff(expected, extracted)

        assert not diff.is_perfect_match
        assert diff.match_rate < 1.0
        assert len(diff.exact_matches) == 1
        assert len(diff.missing_titles) == 2
        assert "Apple Pie" in diff.missing_titles
        assert "Banana Bread" in diff.missing_titles

    def test_generate_report(self):
        """Test report generation."""
        from parse import MelaRecipe, IngredientGroup

        expected = ["Chocolate Cake", "Apple Pie"]
        extracted = [
            MelaRecipe(
                title="Chocolate Cake",
                ingredients=[IngredientGroup(title="Main", ingredients=["flour"])],
                instructions=["Mix and bake"]
            ),
            MelaRecipe(
                title="Lemon Tart",
                ingredients=[IngredientGroup(title="Main", ingredients=["lemons"])],
                instructions=["Mix and bake"]
            )
        ]

        engine = ValidationEngine()
        diff = engine.create_diff(expected, extracted)
        report = engine.generate_report(diff)

        # Report should contain key sections
        assert "VALIDATION REPORT" in report
        assert "EXACT MATCHES" in report
        assert "MISSING RECIPES" in report
        assert "UNEXPECTED RECIPES" in report
        assert "Chocolate Cake" in report


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests using real EPUB files."""

    @pytest.mark.asyncio
    async def test_process_epub_chapters_full_pipeline(self, sample_epub_path):
        """Test complete processing pipeline."""
        recipes, diff = await process_epub_chapters(
            sample_epub_path,
            model="gpt-5-nano",
            use_recipe_list=True,
            max_concurrent=3
        )

        # Should extract some recipes
        assert isinstance(recipes, list)

        # If recipe list was found, diff should exist
        if diff is not None:
            assert hasattr(diff, 'match_rate')
            assert 0.0 <= diff.match_rate <= 1.0

    @pytest.mark.asyncio
    async def test_process_without_recipe_list(self, sample_epub_path):
        """Test processing without recipe list discovery."""
        recipes, diff = await process_epub_chapters(
            sample_epub_path,
            model="gpt-5-nano",
            use_recipe_list=False,
            max_concurrent=3
        )

        # Should still extract recipes
        assert isinstance(recipes, list)

        # Diff should be None when recipe list disabled
        assert diff is None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_epub_path():
    """
    Fixture providing path to sample EPUB file.

    For real tests, you would provide an actual EPUB file path.
    This is a placeholder that should be overridden in conftest.py.
    """
    # Look for test EPUB in common locations
    test_paths = [
        Path("test_data/sample.epub"),
        Path("tests/fixtures/sample.epub"),
        Path("sample.epub"),
    ]

    for path in test_paths:
        if path.exists():
            return str(path)

    pytest.skip("No sample EPUB file found for testing")


# ============================================================================
# Usage Examples
# ============================================================================


async def example_basic_usage():
    """
    Example 1: Basic usage of chapter extraction.

    This example shows the simplest way to extract recipes from an EPUB.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80 + "\n")

    epub_path = "cookbook.epub"

    # Use convenience function for simple processing
    recipes, diff = await process_epub_chapters(epub_path)

    print(f"Extracted {len(recipes)} recipes")

    if diff:
        print(f"Match rate: {diff.match_rate:.1%}")
        print(f"Missing: {len(diff.missing_titles)} recipes")


async def example_manual_pipeline():
    """
    Example 2: Manual pipeline with full control.

    This example shows how to use each component individually
    for maximum control over the extraction process.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Manual Pipeline")
    print("=" * 80 + "\n")

    epub_path = "cookbook.epub"

    # Step 1: Split into chapters
    print("Step 1: Splitting EPUB into chapters...")
    processor = ChapterProcessor(epub_path)
    chapters = await processor.split_into_chapters()
    print(f"Found {len(chapters)} chapters")

    # Step 2: Discover recipe list
    print("\nStep 2: Discovering recipe list...")
    discoverer = RecipeListDiscoverer()
    expected_titles = await discoverer.discover_from_chapters(chapters)

    if expected_titles:
        print(f"Discovered {len(expected_titles)} recipes in index")
        print("Sample titles:", expected_titles[:5])
    else:
        print("No recipe list found")

    # Step 3: Extract recipes
    print("\nStep 3: Extracting recipes from chapters...")
    extractor = AsyncChapterExtractor(model="gpt-5-nano")
    results = await extractor.extract_from_chapters(
        chapters,
        expected_titles=expected_titles,
        max_concurrent=5
    )

    # Step 4: Collect and deduplicate
    print("\nStep 4: Collecting unique recipes...")
    all_recipes = []
    seen_titles = set()

    for result in results:
        if result.is_success:
            print(f"  {result.chapter_name}: {result.recipe_count} recipes")
            for recipe in result.recipes:
                if recipe.title not in seen_titles:
                    seen_titles.add(recipe.title)
                    all_recipes.append(recipe)

    print(f"\nTotal unique recipes: {len(all_recipes)}")

    # Step 5: Validate
    if expected_titles:
        print("\nStep 5: Validating extraction...")
        validator = ValidationEngine()
        diff = validator.create_diff(expected_titles, all_recipes)
        report = validator.generate_report(diff)
        print(report)


async def example_custom_configuration():
    """
    Example 3: Custom configuration and error handling.

    This example shows advanced configuration options and
    proper error handling.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Configuration")
    print("=" * 80 + "\n")

    epub_path = "cookbook.epub"

    try:
        # Custom chapter processor
        processor = ChapterProcessor(epub_path)
        chapters = await processor.split_into_chapters()

        # Custom extractor with retry settings
        extractor = AsyncChapterExtractor(
            model="gpt-5-mini",  # Use larger model
            max_retries=5,  # More retries
            initial_retry_delay=2.0  # Longer delay
        )

        # Process with limited concurrency
        results = await extractor.extract_from_chapters(
            chapters,
            max_concurrent=2  # Slower but more stable
        )

        # Check extraction quality
        validator = ValidationEngine()
        is_valid, message = validator.validate_extraction_quality(
            results,
            min_success_rate=0.9  # Require 90% success
        )

        if is_valid:
            print(f"✓ Quality check passed: {message}")
        else:
            print(f"✗ Quality check failed: {message}")

    except EPUBConversionError as e:
        print(f"Error converting EPUB: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def example_targeted_extraction():
    """
    Example 4: Targeted extraction with specific recipes.

    This example shows how to extract specific recipes when you
    already know what you're looking for.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Targeted Extraction")
    print("=" * 80 + "\n")

    epub_path = "cookbook.epub"

    # Specify exact recipes to extract
    target_recipes = [
        "Chocolate Cake",
        "Apple Pie",
        "Banana Bread"
    ]

    processor = ChapterProcessor(epub_path)
    chapters = await processor.split_into_chapters()

    extractor = AsyncChapterExtractor(model="gpt-5-nano")
    results = await extractor.extract_from_chapters(
        chapters,
        expected_titles=target_recipes  # Targeted extraction
    )

    # Collect results
    found_recipes = []
    for result in results:
        found_recipes.extend(result.recipes)

    print(f"Targeted {len(target_recipes)} recipes")
    print(f"Found {len(found_recipes)} recipes")

    for recipe in found_recipes:
        print(f"  ✓ {recipe.title}")


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run example usage demonstrations."""
    logging.basicConfig(level=logging.INFO)

    print("\n")
    print("=" * 80)
    print("CHAPTER EXTRACTOR USAGE EXAMPLES")
    print("=" * 80)

    # Note: These examples require a real EPUB file
    print("\nNote: These examples require a real EPUB file to run.")
    print("Please update the epub_path in each example.\n")

    # Uncomment to run examples:
    # asyncio.run(example_basic_usage())
    # asyncio.run(example_manual_pipeline())
    # asyncio.run(example_custom_configuration())
    # asyncio.run(example_targeted_extraction())


if __name__ == "__main__":
    main()
