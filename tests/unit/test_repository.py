"""Unit tests for mela_parser.repository module.

Tests file repository operations, slugification, and deduplication.
"""

import json
from pathlib import Path

import pytest

from mela_parser.parse import Category, IngredientGroup, MelaRecipe
from mela_parser.repository import FileRecipeRepository, slugify


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_text(self) -> None:
        """Basic text is slugified correctly."""
        assert slugify("Hello World") == "hello-world"

    def test_special_characters_removed(self) -> None:
        """Special characters are removed."""
        assert slugify("Roasted Chicken & Vegetables!") == "roasted-chicken-vegetables"
        assert slugify("Test (with) [brackets]") == "test-with-brackets"

    def test_multiple_spaces_collapsed(self) -> None:
        """Multiple spaces become single hyphen."""
        assert slugify("Multiple   Spaces   Here") == "multiple-spaces-here"

    def test_leading_trailing_hyphens_removed(self) -> None:
        """Leading and trailing hyphens are removed."""
        assert slugify("  Leading Spaces  ") == "leading-spaces"
        assert slugify("---Hyphens---") == "hyphens"

    def test_empty_string_returns_recipe(self) -> None:
        """Empty string returns 'recipe' as fallback."""
        assert slugify("") == "recipe"
        assert slugify("!!!") == "recipe"

    def test_unicode_characters(self) -> None:
        """Unicode characters are handled."""
        # Non-word characters are removed
        result = slugify("Café Latté")
        assert "caf" in result

    def test_numbers_preserved(self) -> None:
        """Numbers are preserved in slug."""
        assert slugify("Recipe 123") == "recipe-123"

    def test_underscores_become_hyphens(self) -> None:
        """Underscores are converted to hyphens."""
        assert slugify("test_recipe_name") == "test-recipe-name"


class TestFileRecipeRepository:
    """Tests for FileRecipeRepository class."""

    @pytest.fixture
    def repository(self) -> FileRecipeRepository:
        """Create a repository instance."""
        return FileRecipeRepository()

    @pytest.fixture
    def sample_recipe(self) -> MelaRecipe:
        """Create a sample recipe for testing."""
        return MelaRecipe(
            title="Test Recipe",
            text="A delicious test recipe.",
            recipeYield="Serves 4",
            prepTime=15,
            cookTime=30,
            totalTime=45,
            ingredients=[
                IngredientGroup(
                    title="Main ingredients",
                    ingredients=["400g flour", "2 eggs", "100ml milk"],
                ),
            ],
            instructions=[
                "Mix the flour and eggs.",
                "Add milk gradually.",
                "Knead until smooth.",
            ],
            notes="Best served fresh.",
            categories=[Category.Baking],
        )

    @pytest.fixture
    def minimal_recipe(self) -> MelaRecipe:
        """Create a minimal valid recipe."""
        return MelaRecipe(
            title="Minimal Recipe",
            ingredients=[
                IngredientGroup(title="", ingredients=["1 cup flour"]),
            ],
            instructions=["Mix.", "Bake."],
        )

    def test_save_creates_file(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """save() creates a .melarecipe file."""
        result = repository.save(sample_recipe, tmp_path)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".melarecipe"

    def test_save_file_contains_json(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """Saved file contains valid JSON."""
        filepath = repository.save(sample_recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert data["title"] == "Test Recipe"
        assert "400g flour" in data["ingredients"]

    def test_save_formats_time_correctly(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """Time values are formatted as human-readable strings."""
        filepath = repository.save(sample_recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert data["prepTime"] == "15 min"
        assert data["cookTime"] == "30 min"
        assert data["totalTime"] == "45 min"

    def test_save_formats_hours_correctly(
        self, repository: FileRecipeRepository, tmp_path: Path
    ) -> None:
        """Time values over 60 minutes include hours."""
        recipe = MelaRecipe(
            title="Long Cook",
            prepTime=90,
            cookTime=120,
            totalTime=210,
            ingredients=[IngredientGroup(title="", ingredients=["1 item"])],
            instructions=["Step 1.", "Step 2."],
        )

        filepath = repository.save(recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert data["prepTime"] == "1 hr 30 min"
        assert data["cookTime"] == "2 hr"
        assert data["totalTime"] == "3 hr 30 min"

    def test_save_creates_output_directory(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """save() creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "output"
        result = repository.save(sample_recipe, output_dir)

        assert result is not None
        assert output_dir.exists()

    def test_save_skips_duplicate_title(
        self, repository: FileRecipeRepository, minimal_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """save() skips recipes with duplicate titles."""
        # Save first time
        result1 = repository.save(minimal_recipe, tmp_path)
        assert result1 is not None

        # Try to save again with same title
        result2 = repository.save(minimal_recipe, tmp_path)
        assert result2 is None

    def test_save_duplicate_case_insensitive(
        self, repository: FileRecipeRepository, tmp_path: Path
    ) -> None:
        """Duplicate detection is case-insensitive."""
        recipe1 = MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1.", "Step 2."],
        )
        recipe2 = MelaRecipe(
            title="TEST RECIPE",
            ingredients=[IngredientGroup(title="", ingredients=["2 cups"])],
            instructions=["Do this.", "Do that."],
        )

        result1 = repository.save(recipe1, tmp_path)
        result2 = repository.save(recipe2, tmp_path)

        assert result1 is not None
        assert result2 is None  # Skipped as duplicate

    def test_save_skips_existing_file(
        self, repository: FileRecipeRepository, tmp_path: Path
    ) -> None:
        """save() skips if file already exists on disk."""
        recipe = MelaRecipe(
            title="Existing",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1.", "Step 2."],
        )

        # Create file manually
        (tmp_path / "existing.melarecipe").write_text("{}")

        # New repository instance (no memory of previous saves)
        new_repo = FileRecipeRepository()
        result = new_repo.save(recipe, tmp_path)

        assert result is None  # Skipped

    def test_save_generates_uuid(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """Saved recipe has a UUID id field."""
        filepath = repository.save(sample_recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert "id" in data
        assert len(data["id"]) == 36  # UUID format

    def test_save_handles_multiple_ingredient_groups(
        self, repository: FileRecipeRepository, tmp_path: Path
    ) -> None:
        """Multiple ingredient groups are formatted with headers."""
        recipe = MelaRecipe(
            title="Multi Group",
            ingredients=[
                IngredientGroup(title="For the dough", ingredients=["2 cups flour"]),
                IngredientGroup(title="For the filling", ingredients=["1 cup sugar"]),
            ],
            instructions=["Make dough.", "Add filling."],
        )

        filepath = repository.save(recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert "# For the dough" in data["ingredients"]
        assert "# For the filling" in data["ingredients"]

    def test_save_extracts_category_values(
        self, repository: FileRecipeRepository, sample_recipe: MelaRecipe, tmp_path: Path
    ) -> None:
        """Categories are saved as string values."""
        filepath = repository.save(sample_recipe, tmp_path)
        assert filepath is not None

        with filepath.open() as f:
            data = json.load(f)

        assert data["categories"] == ["Baking"]

    def test_deduplicate_removes_duplicates(self, repository: FileRecipeRepository) -> None:
        """deduplicate() removes recipes with duplicate titles."""
        recipes = [
            MelaRecipe(
                title="Recipe A",
                ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
                instructions=["Step 1.", "Step 2."],
            ),
            MelaRecipe(
                title="Recipe B",
                ingredients=[IngredientGroup(title="", ingredients=["2 cups"])],
                instructions=["Do this.", "Do that."],
            ),
            MelaRecipe(
                title="Recipe A",  # Duplicate
                ingredients=[IngredientGroup(title="", ingredients=["3 cups"])],
                instructions=["Another.", "Version."],
            ),
        ]

        unique = repository.deduplicate(recipes)

        assert len(unique) == 2
        assert unique[0].title == "Recipe A"
        assert unique[1].title == "Recipe B"

    def test_deduplicate_case_insensitive(self, repository: FileRecipeRepository) -> None:
        """Deduplication is case-insensitive."""
        recipes = [
            MelaRecipe(
                title="Test Recipe",
                ingredients=[IngredientGroup(title="", ingredients=["1"])],
                instructions=["S1.", "S2."],
            ),
            MelaRecipe(
                title="TEST RECIPE",
                ingredients=[IngredientGroup(title="", ingredients=["2"])],
                instructions=["S3.", "S4."],
            ),
            MelaRecipe(
                title="test recipe",
                ingredients=[IngredientGroup(title="", ingredients=["3"])],
                instructions=["S5.", "S6."],
            ),
        ]

        unique = repository.deduplicate(recipes)

        assert len(unique) == 1
        assert unique[0].title == "Test Recipe"  # First occurrence kept

    def test_deduplicate_preserves_order(self, repository: FileRecipeRepository) -> None:
        """Deduplication preserves order of first occurrences."""
        recipes = [
            MelaRecipe(
                title="C",
                ingredients=[IngredientGroup(title="", ingredients=["1"])],
                instructions=["S1.", "S2."],
            ),
            MelaRecipe(
                title="A",
                ingredients=[IngredientGroup(title="", ingredients=["2"])],
                instructions=["S3.", "S4."],
            ),
            MelaRecipe(
                title="B",
                ingredients=[IngredientGroup(title="", ingredients=["3"])],
                instructions=["S5.", "S6."],
            ),
        ]

        unique = repository.deduplicate(recipes)

        assert [r.title for r in unique] == ["C", "A", "B"]

    def test_deduplicate_empty_list(self, repository: FileRecipeRepository) -> None:
        """deduplicate() handles empty list."""
        unique = repository.deduplicate([])
        assert unique == []

    def test_is_valid_requires_all_fields(self, repository: FileRecipeRepository) -> None:
        """_is_valid checks for required fields."""
        # Valid
        assert repository._is_valid(
            {"title": "Test", "ingredients": "flour", "instructions": "mix"}
        )

        # Missing title
        assert not repository._is_valid(
            {"title": "", "ingredients": "flour", "instructions": "mix"}
        )

        # Missing ingredients
        assert not repository._is_valid({"title": "Test", "ingredients": "", "instructions": "mix"})

        # Missing instructions
        assert not repository._is_valid(
            {"title": "Test", "ingredients": "flour", "instructions": ""}
        )

    def test_slugify_static_method(self, repository: FileRecipeRepository) -> None:
        """Repository has static _slugify method."""
        assert FileRecipeRepository._slugify("Test Recipe") == "test-recipe"
