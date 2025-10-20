#!/usr/bin/env python3
"""
Test-driven quality improvements for recipe extraction.
Tests are written FIRST, then implementations make them pass.
"""
import pytest
from typing import List, Optional

from parse import CookbookParser, MelaRecipe, IngredientGroup, Category
from main_overlap import (
    deduplicate_recipes,
    normalize_title,
    count_recipe_fields,
    is_recipe_complete,
    create_overlapping_chunks
)


class TestRecipeFiltering:
    """Tests for filtering out non-recipes"""

    def test_filters_section_headers(self):
        """Section headers should not be extracted as recipes"""
        markdown = """
# VEGETABLES

This section contains delicious vegetable recipes.

## Roasted Carrots

A simple and delicious side dish.

SERVES 4

500g carrots, peeled and halved
2 tbsp olive oil
1 tsp salt
½ tsp black pepper

Preheat oven to 200°C.
Toss carrots with oil, salt and pepper.
Roast for 30 minutes until tender and caramelized.
Serve hot.

# DESSERTS

Sweet endings to your meal.
"""

        parser = CookbookParser(model="gpt-5-nano")
        result = parser.parse_cookbook(markdown, "Test Book")

        # Should extract only 1 recipe, not section headers
        assert len(result.recipes) == 1, f"Expected 1 recipe, got {len(result.recipes)}"
        assert result.recipes[0].title == "Roasted Carrots"

    def test_filters_recipe_lists(self):
        """Recipe overview lists without full details should be filtered"""
        markdown = """
Ten ways with avocado on toast

Here are ten quick ideas for topping avocado toast:

1. With tomatoes and basil
2. With feta and olives
3. With smoked salmon
4. With poached eggs
5. With chili flakes

## Proper Avocado Toast Recipe

A complete recipe with full details.

SERVES 2

2 slices sourdough bread
1 ripe avocado
1 tbsp lemon juice
Salt and pepper

Toast the bread.
Mash avocado with lemon juice, salt and pepper.
Spread on toast and serve.
"""

        parser = CookbookParser(model="gpt-5-nano")
        result = parser.parse_cookbook(markdown, "Test Book")

        # Should only extract the complete recipe, not the list
        titles = [r.title for r in result.recipes]
        assert "Proper Avocado Toast Recipe" in titles
        # Should NOT include the list overview
        assert not any("ten ways" in t.lower() for t in titles)


class TestRecipeCompleteness:
    """Tests for validating recipe completeness"""

    def test_rejects_recipe_without_ingredients(self):
        """Recipes missing ingredients should be rejected"""
        markdown = """
Quick Soup

A delicious soup recipe.

SERVES 4

Boil water in a large pot.
Add your favorite vegetables.
Simmer for 20 minutes.
Season to taste and serve.
"""

        parser = CookbookParser(model="gpt-5-nano")
        result = parser.parse_cookbook(markdown, "Test Book")

        # Should reject this as incomplete (no ingredients)
        if result.recipes:
            # If extracted, should not be written (validation catches it)
            recipe = result.recipes[0]
            assert not is_recipe_complete(recipe), "Recipe without ingredients should be incomplete"

    def test_rejects_recipe_without_instructions(self):
        """Recipes missing instructions should be rejected"""
        markdown = """
Mystery Dish

SERVES 2

200g rice
100g vegetables
2 tbsp soy sauce
1 tsp sesame oil
"""

        parser = CookbookParser(model="gpt-5-nano")
        result = parser.parse_cookbook(markdown, "Test Book")

        # Should reject this as incomplete (no instructions)
        if result.recipes:
            recipe = result.recipes[0]
            assert not is_recipe_complete(recipe), "Recipe without instructions should be incomplete"

    def test_accepts_complete_recipe(self):
        """Complete recipes should pass validation"""
        markdown = """
Perfect Pasta

A well-documented pasta recipe.

SERVES 4

400g pasta
2 tbsp olive oil
3 cloves garlic, minced
Salt and pepper to taste

# Instructions

Boil pasta according to package directions.
Heat olive oil and sauté garlic until fragrant.
Drain pasta and toss with garlic oil.
Season with salt and pepper.
Serve immediately.
"""

        parser = CookbookParser(model="gpt-5-nano")
        result = parser.parse_cookbook(markdown, "Test Book")

        assert len(result.recipes) >= 1, "Should extract complete recipe"
        recipe = result.recipes[0]
        assert is_recipe_complete(recipe), "Complete recipe should pass validation"


class TestDeduplication:
    """Tests for smart deduplication logic"""

    def test_keeps_most_complete_duplicate(self):
        """When duplicates exist, keep version with most data"""
        recipes = [
            # Incomplete version
            MelaRecipe(
                title="Chocolate Cake",
                ingredients=[IngredientGroup(title="", ingredients=["chocolate", "flour"])],
                instructions=["Mix and bake"],
                text=None,
                yield_=None,
                prepTime=None,
                cookTime=None,
                totalTime=None,
                notes=None,
                categories=None
            ),
            # Complete version
            MelaRecipe(
                title="Chocolate Cake",  # Same title
                ingredients=[IngredientGroup(title="", ingredients=["200g chocolate", "150g flour", "100g sugar"])],
                instructions=["Preheat oven to 180°C", "Mix ingredients", "Bake for 30 minutes"],
                text="A rich, decadent chocolate cake",
                yield_="Serves 8",
                prepTime=15,
                cookTime=30,
                totalTime=45,
                notes="Best served warm",
                categories=[Category.Desserts]
            )
        ]

        result = smart_deduplicate(recipes)

        assert len(result) == 1, "Should deduplicate to 1 recipe"
        kept = result[0]

        # Should keep the more complete version
        assert kept.yield_ == "Serves 8", "Should keep version with yield"
        assert kept.prepTime == 15, "Should keep version with prep time"
        assert len(kept.ingredients[0].ingredients) == 3, "Should keep version with more ingredients"
        assert kept.text is not None, "Should keep version with description"

    def test_handles_title_variations(self):
        """Deduplicate similar titles (case, punctuation)"""
        recipes = [
            MelaRecipe(
                title="Baby Spinach Salad",
                ingredients=[IngredientGroup(title="", ingredients=["spinach"])],
                instructions=["Toss and serve"],
            ),
            MelaRecipe(
                title="baby spinach salad",  # Different case
                ingredients=[IngredientGroup(title="", ingredients=["spinach"])],
                instructions=["Toss and serve"],
            ),
            MelaRecipe(
                title="Baby Spinach Salad (alternate entry)",  # With suffix
                ingredients=[IngredientGroup(title="", ingredients=["spinach"])],
                instructions=["Toss and serve"],
            )
        ]

        result = smart_deduplicate(recipes)

        # Should recognize all as the same recipe
        assert len(result) == 1, f"Should deduplicate to 1 recipe, got {len(result)}"


class TestEndToEnd:
    """End-to-end tests with real cookbook samples"""

    def test_jerusalem_known_recipe(self):
        """Test extraction of known recipe from Jerusalem cookbook"""
        # This is an integration test - requires actual EPUB file
        # Skip if file not available
        import os
        if not os.path.exists("examples/input/jerusalem.epub"):
            pytest.skip("Jerusalem EPUB not available")

        from converter import EpubConverter
        from main_overlap import create_overlapping_chunks, deduplicate_recipes

        # Convert and process
        converter = EpubConverter()
        markdown = converter.convert_epub_to_markdown("examples/input/jerusalem.epub")
        markdown = converter.strip_front_matter(markdown)

        chunks = create_overlapping_chunks(markdown, chunk_size=80000, overlap=40000)

        parser = CookbookParser(model="gpt-5-nano")
        all_recipes = []

        for i, chunk in enumerate(chunks):
            chunk_recipes = parser.parse_cookbook(chunk, "Jerusalem")
            # Discard last from non-final chunks
            if i < len(chunks) - 1 and len(chunk_recipes.recipes) > 1:
                all_recipes.extend(chunk_recipes.recipes[:-1])
            else:
                all_recipes.extend(chunk_recipes.recipes)

        unique_recipes = deduplicate_recipes(all_recipes)

        # Validate we got a good number of recipes
        assert len(unique_recipes) >= 100, f"Expected 100+ recipes, got {len(unique_recipes)}"

        # Check for specific known recipe
        spinach_salad = [
            r for r in unique_recipes
            if "baby spinach" in r.title.lower() and "dates" in r.title.lower()
        ]

        assert len(spinach_salad) == 1, "Should find Baby Spinach Salad"
        recipe = spinach_salad[0]

        # Validate completeness
        assert recipe.ingredients, "Should have ingredients"
        assert recipe.instructions, "Should have instructions"
        assert any("date" in ing.lower() for ing in recipe.ingredients[0].ingredients), "Should mention dates in ingredients"


# Note: Helper functions (is_recipe_complete, count_recipe_fields, etc.)
# are imported from main_overlap.py

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
