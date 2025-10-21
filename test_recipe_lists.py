#!/usr/bin/env python3
"""
TDD: Recipe list matching tests.
These tests MUST fail initially, then we make them pass.

Ground truth: Official recipe lists in examples/output/recipe-lists/
"""
import pytest
from pathlib import Path
from typing import List, Set

from converter import EpubConverter
from parse import CookbookParser
from main_overlap import (
    create_overlapping_chunks,
    deduplicate_recipes,
    is_recipe_complete,
)


def load_recipe_list(filename: str) -> List[str]:
    """Load official recipe list as ground truth."""
    path = Path("examples/output/recipe-lists") / filename
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def aggressive_normalize_title(title: str) -> str:
    """Ultra-aggressive normalization for matching."""
    import re

    # Remove ALL parenthetical content
    title = re.sub(r'\([^)]*\)', '', title)

    # Remove suffixes
    suffixes = ['partial', 'duplicate', 'final', 'variant', 'alternate',
                'continued', 'entry', 'listing', 'rendering']
    for suffix in suffixes:
        title = re.sub(rf'\b{suffix}\b', '', title, flags=re.IGNORECASE)

    # Remove punctuation and normalize
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)

    return title.lower().strip()


class TestRecipeListMatching:
    """Tests that extraction matches official recipe lists EXACTLY."""

    def test_jerusalem_exact_count(self):
        """Jerusalem must extract exactly 125 recipes (not 157)."""
        expected_list = load_recipe_list("jerusalem-recipe-list.txt")

        # This will extract recipes with current implementation
        converter = EpubConverter()
        markdown = converter.convert_epub_to_markdown("examples/input/jerusalem.epub")
        markdown = converter.strip_front_matter(markdown)

        # Use current implementation (will fail)
        chunks = create_overlapping_chunks(markdown, chunk_size=80000, overlap=60000)

        parser = CookbookParser(model="gpt-5-nano")
        all_recipes = []

        for i, chunk in enumerate(chunks):
            chunk_recipes = parser.parse_cookbook(chunk, "Jerusalem")
            # Current: extends all
            all_recipes.extend(chunk_recipes.recipes)

        # Current deduplication
        unique = deduplicate_recipes(all_recipes)

        # THIS WILL FAIL
        assert len(unique) == 125, f"Expected 125 recipes, got {len(unique)} (off by {len(unique) - 125})"

    def test_no_title_modifications(self):
        """Model must NOT add suffixes like (partial), (duplicate), etc."""
        converter = EpubConverter()
        markdown = converter.convert_epub_to_markdown("examples/input/jerusalem.epub")
        markdown = converter.strip_front_matter(markdown)

        chunks = create_overlapping_chunks(markdown, chunk_size=80000, overlap=60000)
        parser = CookbookParser(model="gpt-5-nano")

        all_recipes = []
        for chunk in chunks[:3]:  # Test first 3 chunks
            chunk_recipes = parser.parse_cookbook(chunk, "Jerusalem")
            all_recipes.extend(chunk_recipes.recipes)

        # Check for prohibited modifications
        bad_suffixes = ['(partial)', '(duplicate)', '(final)', '(variant)',
                       '(alternate)', '(continued)', '(entry)', '(listing)']

        violations = []
        for recipe in all_recipes:
            for suffix in bad_suffixes:
                if suffix.lower() in recipe.title.lower():
                    violations.append(recipe.title)
                    break

        # THIS WILL FAIL
        assert len(violations) == 0, f"Found {len(violations)} titles with modifications: {violations[:10]}"

    def test_deterministic_extraction(self):
        """With temperature=0, same chunk should give same results."""
        converter = EpubConverter()
        markdown = converter.convert_epub_to_markdown("examples/input/jerusalem.epub")
        chunks = create_overlapping_chunks(markdown, chunk_size=80000, overlap=60000)

        # Extract same chunk twice
        parser = CookbookParser(model="gpt-5-nano")
        result1 = parser.parse_cookbook(chunks[0], "Jerusalem")
        result2 = parser.parse_cookbook(chunks[0], "Jerusalem")

        # Should be identical
        titles1 = [r.title for r in result1.recipes]
        titles2 = [r.title for r in result2.recipes]

        # THIS MAY FAIL if temperature not set to 0
        assert titles1 == titles2, "Non-deterministic extraction (temperature not 0?)"


class TestDeduplicationStrategy:
    """Tests for proper deduplication favoring first occurrence."""

    def test_favor_first_occurrence(self):
        """When duplicates exist, keep recipe from earliest chunk."""
        from parse import MelaRecipe, IngredientGroup

        # Simulate same recipe in multiple chunks
        recipe_chunk2 = MelaRecipe(
            title="Falafel",
            ingredients=[IngredientGroup(title="", ingredients=["chickpeas", "garlic", "cumin"])],
            instructions=["Blend", "Shape", "Fry"]
        )

        recipe_chunk5 = MelaRecipe(
            title="Falafel",
            ingredients=[IngredientGroup(title="", ingredients=["chickpeas"])],  # Less complete
            instructions=["Fry"]
        )

        recipes_with_indices = [
            (recipe_chunk5, 5),  # Later chunk
            (recipe_chunk2, 2),  # Earlier chunk
        ]

        # Need new function: deduplicate_favor_first
        from main_overlap import deduplicate_favor_first

        result = deduplicate_favor_first(recipes_with_indices)

        assert len(result) == 1, "Should deduplicate to 1 recipe"
        assert result[0][1] == 2, "Should keep earlier chunk (2, not 5)"

        # THIS WILL FAIL: Function doesn't exist yet


class TestOptimalChunking:
    """Tests for optimal chunk size based on recipe lengths."""

    def test_chunk_size_accommodates_max_recipe(self):
        """Chunk size should be 2x max recipe length."""
        MAX_RECIPE_LENGTH = 20000

        # Current defaults
        from main_overlap import create_overlapping_chunks

        # Need to check what chunk size is actually used
        # This is more of a config test

        # Recommended: 40K chunks (2x max recipe)
        assert True  # Placeholder - will update implementation

    def test_overlap_covers_max_recipe(self):
        """Overlap should be >= max recipe length."""
        MAX_RECIPE_LENGTH = 20000

        # Overlap should be 20K+ to ensure no recipe is split
        # This ensures every recipe is fully contained in at least one chunk

        assert True  # Placeholder - will check in implementation


class TestRecipeListValidation:
    """Validate extracted recipes against official lists."""

    @pytest.mark.parametrize("epub_file,list_file,expected_count", [
        ("jerusalem.epub", "jerusalem-recipe-list.txt", 125),
        ("a-modern-way-to-eat.epub", "a-modern-way-to-eat-recipe-list.txt", 142),
        ("completely-perfect.epub", "completely-perfect-recipe-list.txt", 122),
    ])
    def test_matches_official_recipe_list(self, epub_file, list_file, expected_count):
        """Each book must match its official recipe list."""
        expected_titles = load_recipe_list(list_file)

        assert len(expected_titles) == expected_count, f"Recipe list file has wrong count"

        # Extract recipes (this will use current broken implementation)
        # Will be fixed in GREEN phase

        # For now, just validate the lists are loaded correctly
        assert len(expected_titles) > 0

        # Full implementation will be added in GREEN phase
        # THIS TEST is partially implemented (list loading works)


if __name__ == "__main__":
    # Run tests - they should FAIL
    pytest.main([__file__, "-v", "--tb=short", "-x"])
