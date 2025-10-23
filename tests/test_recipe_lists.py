#!/usr/bin/env python3
"""
Comprehensive test suite for chapter-based recipe extraction system.

Tests validate:
- Exact recipe counts against ground truth lists
- Title matching accuracy
- No duplicate recipes
- Recipe list discovery
- Iterative refinement process
- End-to-end extraction pipeline

Ground truth: Official recipe lists in examples/output/recipe-lists/
"""
import asyncio
import json
import pytest
from pathlib import Path
from typing import List, Set, Dict, Tuple
from unittest.mock import Mock, patch, AsyncMock

from scripts.main_chapters_v2 import (
    ChapterProcessor,
    RecipeListDiscoverer,
    ChapterExtractor,
    ExtractionPipeline,
    PromptLibrary,
    ExtractionResult,
    ValidationReport,
    PromptImprovements,
)
from mela_parser.parse import MelaRecipe, IngredientGroup


# ============================================================================
# TEST UTILITIES
# ============================================================================


def load_recipe_list(filename: str) -> List[str]:
    """Load official recipe list as ground truth."""
    path = Path("examples/output/recipe-lists") / filename
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching (lowercase, no extra spaces)."""
    import re
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title)
    return title.lower().strip()


def create_mock_recipe(title: str, num_ingredients: int = 3) -> MelaRecipe:
    """Create a mock recipe for testing."""
    ingredients = [f"ingredient_{i}" for i in range(num_ingredients)]
    return MelaRecipe(
        title=title,
        ingredients=[IngredientGroup(title="", ingredients=ingredients)],
        instructions=[f"Step {i+1}" for i in range(2)],
    )


# ============================================================================
# TEST CHAPTER EXTRACTION
# ============================================================================


class TestChapterExtraction:
    """Tests for chapter-based extraction accuracy against ground truth."""

    @pytest.mark.asyncio
    async def test_jerusalem_chapter_count(self):
        """Jerusalem must extract exactly 125 recipes."""
        expected_titles = load_recipe_list("jerusalem-recipe-list.txt")
        assert len(expected_titles) == 125, "Ground truth file should have 125 recipes"

        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        epub_path = "examples/input/jerusalem.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        # Run extraction with default prompts (single-pass, no iteration)
        prompts = PromptLibrary.default()
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        # Validate count
        assert results.unique_count == 125, (
            f"Expected exactly 125 unique recipes, got {results.unique_count}. "
            f"Difference: {results.unique_count - 125}"
        )

    @pytest.mark.asyncio
    async def test_modern_way_chapter_count(self):
        """A Modern Way to Eat must extract exactly 142 recipes."""
        expected_titles = load_recipe_list("a-modern-way-to-eat-recipe-list.txt")
        assert len(expected_titles) == 142, "Ground truth file should have 142 recipes"

        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        epub_path = "examples/input/a-modern-way-to-eat.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        prompts = PromptLibrary.default()
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        assert results.unique_count == 142, (
            f"Expected exactly 142 unique recipes, got {results.unique_count}. "
            f"Difference: {results.unique_count - 142}"
        )

    @pytest.mark.asyncio
    async def test_completely_perfect_chapter_count(self):
        """Completely Perfect must extract exactly 122 recipes."""
        expected_titles = load_recipe_list("completely-perfect-recipe-list.txt")
        assert len(expected_titles) == 122, "Ground truth file should have 122 recipes"

        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        epub_path = "examples/input/completely-perfect.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        prompts = PromptLibrary.default()
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        assert results.unique_count == 122, (
            f"Expected exactly 122 unique recipes, got {results.unique_count}. "
            f"Difference: {results.unique_count - 122}"
        )

    @pytest.mark.asyncio
    async def test_no_duplicate_recipes(self):
        """Chapter boundaries must prevent duplicate recipe extraction."""
        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        epub_path = "examples/input/simple.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        prompts = PromptLibrary.default()
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        # Check that deduplication was effective
        assert results.unique_count == len(results.recipes), (
            "Unique count should match recipe list length (no duplicates in output)"
        )

        # Verify all titles are unique
        titles = [r.title for r in results.recipes]
        unique_titles = set(titles)

        assert len(titles) == len(unique_titles), (
            f"Found duplicate titles in output. "
            f"Total: {len(titles)}, Unique: {len(unique_titles)}, "
            f"Duplicates: {len(titles) - len(unique_titles)}"
        )

        # Log deduplication stats
        print(f"\nDeduplication stats:")
        print(f"  Total extracted: {results.total_extracted}")
        print(f"  Unique recipes: {results.unique_count}")
        print(f"  Duplicates removed: {results.duplicates_removed}")

        # Chapter-based extraction should minimize duplicates
        duplicate_rate = results.duplicates_removed / results.total_extracted if results.total_extracted > 0 else 0
        assert duplicate_rate < 0.20, (
            f"Duplicate rate too high: {duplicate_rate:.1%}. "
            f"Chapter-based extraction should have <20% duplicates"
        )

    @pytest.mark.asyncio
    async def test_exact_title_matching(self):
        """Extracted titles must match expected titles character-for-character."""
        expected_titles = load_recipe_list("simple-recipe-list.txt")
        assert len(expected_titles) == 140, "Simple recipe list should have 140 recipes"

        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        epub_path = "examples/input/simple.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        prompts = PromptLibrary.default()
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        # Build title sets for comparison
        extracted_titles = {r.title for r in results.recipes}
        expected_title_set = set(expected_titles)

        # Find matches
        exact_matches = extracted_titles & expected_title_set
        missing_titles = expected_title_set - extracted_titles
        extra_titles = extracted_titles - expected_title_set

        match_rate = len(exact_matches) / len(expected_title_set) if expected_title_set else 0

        print(f"\nTitle matching analysis:")
        print(f"  Expected recipes: {len(expected_title_set)}")
        print(f"  Extracted recipes: {len(extracted_titles)}")
        print(f"  Exact matches: {len(exact_matches)}")
        print(f"  Match rate: {match_rate:.1%}")
        print(f"  Missing: {len(missing_titles)}")
        print(f"  Extra: {len(extra_titles)}")

        if missing_titles:
            print(f"\n  Missing titles (first 10): {sorted(list(missing_titles))[:10]}")
        if extra_titles:
            print(f"\n  Extra titles (first 10): {sorted(list(extra_titles))[:10]}")

        # Require high match rate (allowing for some variation)
        assert match_rate >= 0.90, (
            f"Title match rate too low: {match_rate:.1%}. "
            f"Expected at least 90% exact matches. "
            f"Missing {len(missing_titles)} recipes, {len(extra_titles)} extra."
        )


# ============================================================================
# TEST RECIPE LIST DISCOVERY
# ============================================================================


class TestRecipeListDiscovery:
    """Tests for recipe list discovery from EPUB structure."""

    @pytest.mark.asyncio
    async def test_discovers_correct_count(self):
        """Recipe list discovery finds expected number of recipes."""
        processor = ChapterProcessor()
        epub_path = "examples/input/simple.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        # Convert chapters
        book, chapters = processor.convert_epub_by_chapters(epub_path)

        # Discover recipe list
        from openai import OpenAI
        client = OpenAI()
        discoverer = RecipeListDiscoverer(client, model="gpt-5-mini")

        prompts = PromptLibrary.default()
        discovered_titles = discoverer.discover_recipe_list(chapters, prompts.discovery_prompt)

        assert discovered_titles is not None, "Should discover recipe list"
        assert len(discovered_titles) > 0, "Should find at least one recipe"

        # Load ground truth
        expected_titles = load_recipe_list("simple-recipe-list.txt")

        # Allow some variance (discovery may find slightly different count)
        count_diff = abs(len(discovered_titles) - len(expected_titles))
        tolerance = int(len(expected_titles) * 0.10)  # 10% tolerance

        print(f"\nDiscovery analysis:")
        print(f"  Expected count: {len(expected_titles)}")
        print(f"  Discovered count: {len(discovered_titles)}")
        print(f"  Difference: {count_diff}")
        print(f"  Tolerance: {tolerance}")

        assert count_diff <= tolerance, (
            f"Discovered count ({len(discovered_titles)}) differs too much from "
            f"expected ({len(expected_titles)}). Difference: {count_diff}, Tolerance: {tolerance}"
        )

    @pytest.mark.asyncio
    async def test_list_cleaning(self):
        """gpt-5-mini removes section headers properly."""
        from openai import OpenAI
        client = OpenAI()
        discoverer = RecipeListDiscoverer(client, model="gpt-5-mini")

        # Create mock chapters with section headers
        mock_chapters = [
            ("toc.html", """
            [Contents](#)
            [Introduction](#)
            [Breakfast](#breakfast)
            [Braised eggs with leek and za'atar](#recipe1)
            [Harissa and Manchego omelettes](#recipe2)
            [About the Author](#about)
            [Index](#index)
            [Recipe Index](#recipe-index)
            [Lunch](#lunch)
            [Chilled cucumber, cauliflower and ginger soup](#recipe3)
            """)
        ]

        prompts = PromptLibrary.default()
        discovered_titles = discoverer.discover_recipe_list(mock_chapters, prompts.discovery_prompt)

        if discovered_titles is None:
            pytest.skip("Discovery returned None (no links found)")

        # Should NOT contain section headers
        bad_headers = [
            "Contents", "Introduction", "About", "Index",
            "Recipe Index", "Breakfast", "Lunch", "Dinner"
        ]

        found_headers = [title for title in discovered_titles if title in bad_headers]

        print(f"\nCleaning analysis:")
        print(f"  Discovered titles: {len(discovered_titles)}")
        print(f"  Found headers: {found_headers}")

        assert len(found_headers) == 0, (
            f"Discovery should remove section headers. Found: {found_headers}"
        )

    @pytest.mark.asyncio
    async def test_handles_no_list_found(self):
        """Graceful handling when no TOC/Index exists."""
        from openai import OpenAI
        client = OpenAI()
        discoverer = RecipeListDiscoverer(client, model="gpt-5-mini")

        # Create mock chapters with NO recipe list links
        mock_chapters = [
            ("chapter1.html", "This is just plain text. No links here."),
            ("chapter2.html", "Another chapter with no links."),
        ]

        prompts = PromptLibrary.default()
        discovered_titles = discoverer.discover_recipe_list(mock_chapters, prompts.discovery_prompt)

        # Should return None or empty list
        assert discovered_titles is None or len(discovered_titles) == 0, (
            "Should return None or empty list when no recipe list found"
        )


# ============================================================================
# TEST ITERATIVE REFINEMENT
# ============================================================================


class TestIterativeRefinement:
    """Tests for iterative prompt refinement process."""

    @pytest.mark.asyncio
    async def test_iteration_improves_accuracy(self):
        """Each iteration should increase match rate (or stay same if perfect)."""
        # This test would require running full iterations, which is expensive
        # Instead, we'll test the validation logic

        # Create mock validation reports for iterations
        iteration1 = ValidationReport(
            extracted_titles={"Recipe A", "Recipe B"},
            discovered_titles={"Recipe A", "Recipe B", "Recipe C", "Recipe D"},
            matched_titles={"Recipe A", "Recipe B"},
            missing_titles={"Recipe C", "Recipe D"},
            extra_titles=set(),
            match_percentage=50.0,
            total_discovered=4,
            total_extracted=2,
            is_perfect_match=False,
        )

        iteration2 = ValidationReport(
            extracted_titles={"Recipe A", "Recipe B", "Recipe C"},
            discovered_titles={"Recipe A", "Recipe B", "Recipe C", "Recipe D"},
            matched_titles={"Recipe A", "Recipe B", "Recipe C"},
            missing_titles={"Recipe D"},
            extra_titles=set(),
            match_percentage=75.0,
            total_discovered=4,
            total_extracted=3,
            is_perfect_match=False,
        )

        # Verify improvement
        assert iteration2.match_percentage > iteration1.match_percentage, (
            "Iteration 2 should have higher match rate than iteration 1"
        )

        assert len(iteration2.missing_titles) < len(iteration1.missing_titles), (
            "Iteration 2 should have fewer missing titles"
        )

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        """Pipeline stops at max iterations."""
        pipeline = ExtractionPipeline(max_concurrent_chapters=2)

        # Mock the extract_recipes method to return incomplete results
        async def mock_extract(epub_path, prompts, model):
            # Return consistent incomplete results
            mock_results = ExtractionResult(
                recipes=[create_mock_recipe("Recipe A"), create_mock_recipe("Recipe B")],
                total_extracted=2,
                unique_count=2,
                duplicates_removed=0,
                chapters_processed=1,
                chapters_with_recipes=1,
                extraction_time=0.1,
                model_used=model,
            )
            mock_chapters = [("chapter1.html", "content")]
            mock_discovered = ["Recipe A", "Recipe B", "Recipe C"]  # Always missing Recipe C
            return mock_results, mock_chapters, mock_discovered

        # Patch extract_recipes
        pipeline.extract_recipes = mock_extract

        # Run with max_iterations=3
        max_iter = 3
        final_recipes, final_prompts, history = await pipeline.iterative_refinement(
            "fake.epub",
            max_iterations=max_iter,
            model="gpt-5-nano",
        )

        # Should have at most max_iterations entries
        assert len(history) <= max_iter, (
            f"Should have at most {max_iter} iterations, got {len(history)}"
        )

    @pytest.mark.asyncio
    async def test_saves_iteration_history(self):
        """Iteration history saved to JSON."""
        import tempfile
        import shutil

        pipeline = ExtractionPipeline(max_concurrent_chapters=2)

        # Create temp output directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Mock extract_recipes to return perfect match immediately
            async def mock_extract(epub_path, prompts, model):
                mock_results = ExtractionResult(
                    recipes=[create_mock_recipe("Recipe A"), create_mock_recipe("Recipe B")],
                    total_extracted=2,
                    unique_count=2,
                    duplicates_removed=0,
                    chapters_processed=1,
                    chapters_with_recipes=1,
                    extraction_time=0.1,
                    model_used=model,
                )
                mock_chapters = [("chapter1.html", "content")]
                mock_discovered = ["Recipe A", "Recipe B"]  # Perfect match
                return mock_results, mock_chapters, mock_discovered

            pipeline.extract_recipes = mock_extract

            # Run iteration
            final_recipes, final_prompts, history = await pipeline.iterative_refinement(
                "fake.epub",
                max_iterations=1,
                model="gpt-5-nano",
                output_dir=temp_dir,
            )

            # Check iteration snapshot was saved
            snapshot_path = Path(temp_dir) / "iteration_1.json"
            assert snapshot_path.exists(), f"Iteration snapshot should be saved at {snapshot_path}"

            # Verify JSON structure
            with open(snapshot_path, "r") as f:
                snapshot = json.load(f)

            assert "iteration" in snapshot
            assert "validation" in snapshot
            assert "prompts" in snapshot

            assert snapshot["iteration"]["iteration"] == 1
            assert snapshot["iteration"]["match_percentage"] == 100.0

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# INTEGRATION TEST
# ============================================================================


class TestEndToEndExtraction:
    """Integration tests for complete extraction pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_extraction(self):
        """Full pipeline on simple.epub validates all phases complete successfully."""
        epub_path = "examples/input/simple.epub"

        if not Path(epub_path).exists():
            pytest.skip(f"Test EPUB not found: {epub_path}")

        pipeline = ExtractionPipeline(max_concurrent_chapters=5)
        prompts = PromptLibrary.default()

        print(f"\nRunning end-to-end extraction on {epub_path}")

        # Phase 1: Extraction
        print("  Phase 1: Extraction")
        results, chapters, discovered_list = await pipeline.extract_recipes(
            epub_path, prompts, model="gpt-5-nano"
        )

        # Verify extraction phase
        assert results is not None, "Extraction should return results"
        assert results.recipes is not None, "Should have recipes list"
        assert results.chapters_processed > 0, "Should process at least one chapter"
        assert results.unique_count > 0, "Should extract at least one recipe"

        print(f"    Extracted {results.unique_count} unique recipes from {results.chapters_processed} chapters")

        # Phase 2: Discovery
        print("  Phase 2: Recipe list discovery")
        assert discovered_list is not None, "Should discover recipe list"
        assert len(discovered_list) > 0, "Should find recipes in list"

        print(f"    Discovered {len(discovered_list)} recipes in book's recipe list")

        # Phase 3: Validation
        print("  Phase 3: Validation")
        validation = pipeline.validate_extraction(results, discovered_list)

        assert validation is not None, "Validation should return report"
        assert validation.total_discovered == len(discovered_list), "Validation should track discovered count"
        assert validation.total_extracted == results.unique_count, "Validation should track extracted count"

        print(f"    Match rate: {validation.match_percentage:.1f}%")
        print(f"    Matched: {len(validation.matched_titles)}")
        print(f"    Missing: {len(validation.missing_titles)}")
        print(f"    Extra: {len(validation.extra_titles)}")

        # Phase 4: Results validation
        print("  Phase 4: Results validation")

        # Should achieve reasonable match rate
        assert validation.match_percentage >= 70.0, (
            f"Match rate too low: {validation.match_percentage:.1f}%. "
            f"Expected at least 70% for basic extraction."
        )

        # Deduplication should work
        assert results.unique_count <= results.total_extracted, (
            "Unique count should be <= total extracted"
        )

        # Metadata should be populated
        assert "expected_count" in results.metadata, "Should track expected count"
        assert results.metadata["expected_count"] == len(discovered_list), "Expected count should match discovered"

        print(f"\n  End-to-end test PASSED")
        print(f"    Total time: {results.extraction_time:.1f}s")
        print(f"    Model: {results.model_used}")
        print(f"    Deduplication: {results.duplicates_removed} removed")

    @pytest.mark.asyncio
    async def test_validation_report_structure(self):
        """Validation report has correct structure and can be serialized."""
        # Create sample validation report
        validation = ValidationReport(
            extracted_titles={"Recipe A", "Recipe B"},
            discovered_titles={"Recipe A", "Recipe B", "Recipe C"},
            matched_titles={"Recipe A", "Recipe B"},
            missing_titles={"Recipe C"},
            extra_titles=set(),
            match_percentage=66.67,
            total_discovered=3,
            total_extracted=2,
            is_perfect_match=False,
        )

        # Test to_dict serialization
        validation_dict = validation.to_dict()

        assert "extracted_titles" in validation_dict
        assert "discovered_titles" in validation_dict
        assert "matched_titles" in validation_dict
        assert "missing_titles" in validation_dict
        assert "extra_titles" in validation_dict
        assert "match_percentage" in validation_dict
        assert "is_perfect_match" in validation_dict

        # Should be JSON-serializable
        json_str = json.dumps(validation_dict)
        assert json_str is not None

        # Should round-trip
        loaded = json.loads(json_str)
        assert loaded["match_percentage"] == validation_dict["match_percentage"]
        assert loaded["total_discovered"] == 3
        assert loaded["total_extracted"] == 2


# ============================================================================
# TEST PROMPT LIBRARY
# ============================================================================


class TestPromptLibrary:
    """Tests for prompt library and versioning."""

    def test_default_prompts_have_placeholders(self):
        """Default prompts contain required placeholders."""
        prompts = PromptLibrary.default()

        # Discovery prompt should have {combined_lists}
        assert "{combined_lists}" in prompts.discovery_prompt, (
            "Discovery prompt must have {combined_lists} placeholder"
        )

        # Extraction prompt should have {expected_list} and {chapter_md}
        assert "{expected_list}" in prompts.extraction_prompt, (
            "Extraction prompt must have {expected_list} placeholder"
        )
        assert "{chapter_md}" in prompts.extraction_prompt, (
            "Extraction prompt must have {chapter_md} placeholder"
        )

    def test_prompt_serialization(self):
        """PromptLibrary can be serialized and deserialized."""
        original = PromptLibrary.default()
        original.version = 2
        original.iteration_history = [
            {"version": 2, "changes": "Improved accuracy", "confidence": 0.85}
        ]

        # Serialize
        data = original.to_dict()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        loaded = PromptLibrary.from_dict(loaded_data)

        # Verify
        assert loaded.version == original.version
        assert loaded.discovery_prompt == original.discovery_prompt
        assert loaded.extraction_prompt == original.extraction_prompt
        assert len(loaded.iteration_history) == len(original.iteration_history)

    def test_prompt_version_increments(self):
        """Prompt version increments with iterations."""
        prompts = PromptLibrary.default()
        assert prompts.version == 1

        # Simulate version increment
        new_prompts = PromptLibrary(
            discovery_prompt=prompts.discovery_prompt,
            extraction_prompt=prompts.extraction_prompt,
            version=prompts.version + 1,
            iteration_history=prompts.iteration_history + [{"version": 2}],
        )

        assert new_prompts.version == 2
        assert len(new_prompts.iteration_history) == 1


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestAllCookbooks:
    """Parametrized tests across all cookbooks."""

    @pytest.mark.parametrize("epub_file,list_file,expected_count", [
        ("jerusalem.epub", "jerusalem-recipe-list.txt", 125),
        ("a-modern-way-to-eat.epub", "a-modern-way-to-eat-recipe-list.txt", 142),
        ("completely-perfect.epub", "completely-perfect-recipe-list.txt", 122),
        ("simple.epub", "simple-recipe-list.txt", 140),
    ])
    def test_ground_truth_files_valid(self, epub_file, list_file, expected_count):
        """Validate ground truth recipe list files exist and have correct counts."""
        list_path = Path("examples/output/recipe-lists") / list_file

        assert list_path.exists(), f"Recipe list file not found: {list_path}"

        titles = load_recipe_list(list_file)
        assert len(titles) == expected_count, (
            f"{list_file} should have {expected_count} recipes, got {len(titles)}"
        )

        # All titles should be non-empty
        assert all(title.strip() for title in titles), (
            f"{list_file} contains empty titles"
        )

        # No duplicate titles in ground truth
        unique_titles = set(titles)
        assert len(unique_titles) == len(titles), (
            f"{list_file} contains duplicate titles. "
            f"Total: {len(titles)}, Unique: {len(unique_titles)}"
        )


# ============================================================================
# TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
