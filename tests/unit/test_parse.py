"""Unit tests for mela_parser.parse module.

Tests Pydantic models, enums, and data structures for recipe parsing.
"""

import pytest
from pydantic import ValidationError

from mela_parser.parse import (
    Category,
    ChapterTitles,
    CookbookRecipes,
    CookbookTOC,
    IngredientGroup,
    MelaRecipe,
    TOCEntry,
)


class TestCategory:
    """Tests for Category enum."""

    def test_all_categories_exist(self) -> None:
        """All expected categories are defined."""
        expected = [
            "Breakfasts",
            "Starters",
            "Soups",
            "Salads",
            "Mains",
            "Sides",
            "Desserts",
            "Meat",
            "Seafood",
            "Vegetarian",
            "Vegan",
            "Pasta",
            "Drinks",
            "Sauces",
            "Baking",
            "Holiday",
        ]
        for cat in expected:
            assert hasattr(Category, cat) or cat in [c.value for c in Category]

    def test_category_is_string_enum(self) -> None:
        """Category values are strings."""
        assert Category.Mains.value == "Mains"
        assert Category.Desserts.value == "Desserts"
        assert isinstance(Category.Mains.value, str)

    def test_category_cuisine_types(self) -> None:
        """Cuisine-specific categories exist."""
        cuisines = ["Italian", "Mexican", "Indian", "Chinese", "Thai", "Japanese", "Greek"]
        for cuisine in cuisines:
            assert cuisine in [c.value for c in Category]


class TestIngredientGroup:
    """Tests for IngredientGroup Pydantic model."""

    def test_valid_ingredient_group(self) -> None:
        """Valid ingredient group is created successfully."""
        group = IngredientGroup(
            title="Main ingredients",
            ingredients=["400g flour", "2 eggs", "100ml milk"],
        )
        assert group.title == "Main ingredients"
        assert len(group.ingredients) == 3

    def test_requires_at_least_one_ingredient(self) -> None:
        """Ingredient group requires at least one ingredient."""
        with pytest.raises(ValidationError):
            IngredientGroup(title="Empty", ingredients=[])

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValidationError):
            IngredientGroup(
                title="Test",
                ingredients=["1 cup flour"],
                extra_field="not allowed",  # type: ignore[call-arg]
            )

    def test_empty_title_allowed(self) -> None:
        """Empty title is allowed for ungrouped ingredients."""
        group = IngredientGroup(title="", ingredients=["1 cup flour"])
        assert group.title == ""


class TestMelaRecipe:
    """Tests for MelaRecipe Pydantic model."""

    @pytest.fixture
    def valid_recipe_data(self) -> dict:
        """Provide minimal valid recipe data."""
        return {
            "title": "Test Recipe",
            "ingredients": [
                {"title": "Main", "ingredients": ["1 cup flour", "2 eggs"]},
            ],
            "instructions": ["Mix ingredients.", "Bake until done."],
        }

    def test_valid_recipe_minimal(self, valid_recipe_data: dict) -> None:
        """Recipe with minimal required fields is valid."""
        recipe = MelaRecipe(**valid_recipe_data)
        assert recipe.title == "Test Recipe"
        assert len(recipe.ingredients) == 1
        assert len(recipe.instructions) == 2

    def test_valid_recipe_full(self) -> None:
        """Recipe with all fields is valid."""
        recipe = MelaRecipe(
            title="Full Recipe",
            text="A delicious dish.",
            images=["../images/recipe.jpg"],
            recipeYield="Serves 4",
            prepTime=15,
            cookTime=30,
            totalTime=45,
            ingredients=[
                IngredientGroup(title="Main", ingredients=["400g flour", "2 eggs"]),
                IngredientGroup(title="Sauce", ingredients=["100ml cream"]),
            ],
            instructions=["Prepare.", "Cook.", "Serve."],
            notes="Best served warm.",
            categories=[Category.Mains, Category.Italian],
            is_standalone_recipe=True,
        )
        assert recipe.title == "Full Recipe"
        assert recipe.prepTime == 15
        assert recipe.cookTime == 30
        assert recipe.categories is not None
        assert len(recipe.categories) == 2

    def test_requires_title(self) -> None:
        """Recipe requires a title."""
        with pytest.raises(ValidationError):
            MelaRecipe(
                title="",  # Empty title should fail validation
                ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
                instructions=["Step 1.", "Step 2."],
            )

    def test_requires_at_least_one_ingredient_group(self) -> None:
        """Recipe requires at least one ingredient group."""
        with pytest.raises(ValidationError):
            MelaRecipe(
                title="No Ingredients",
                ingredients=[],
                instructions=["Step 1.", "Step 2."],
            )

    def test_requires_at_least_two_instructions(self) -> None:
        """Recipe requires at least two instruction steps."""
        with pytest.raises(ValidationError):
            MelaRecipe(
                title="One Step",
                ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
                instructions=["Only one step."],
            )

    def test_optional_fields_default_to_none(self, valid_recipe_data: dict) -> None:
        """Optional fields default to None."""
        recipe = MelaRecipe(**valid_recipe_data)
        assert recipe.text is None
        assert recipe.images is None
        assert recipe.recipeYield is None
        assert recipe.prepTime is None
        assert recipe.cookTime is None
        assert recipe.totalTime is None
        assert recipe.notes is None
        assert recipe.categories is None

    def test_is_standalone_recipe_defaults_true(self, valid_recipe_data: dict) -> None:
        """is_standalone_recipe defaults to True."""
        recipe = MelaRecipe(**valid_recipe_data)
        assert recipe.is_standalone_recipe is True

    def test_extra_fields_forbidden(self, valid_recipe_data: dict) -> None:
        """Extra fields are not allowed."""
        valid_recipe_data["unknown_field"] = "value"
        with pytest.raises(ValidationError):
            MelaRecipe(**valid_recipe_data)


class TestChapterTitles:
    """Tests for ChapterTitles model."""

    def test_default_values(self) -> None:
        """ChapterTitles has sensible defaults."""
        chapter = ChapterTitles()
        assert chapter.titles == []
        assert chapter.chapter_type == "recipes"

    def test_with_titles(self) -> None:
        """ChapterTitles can store recipe titles."""
        chapter = ChapterTitles(
            titles=["Recipe One", "Recipe Two", "Recipe Three"],
            chapter_type="recipes",
        )
        assert len(chapter.titles) == 3

    def test_different_chapter_types(self) -> None:
        """Different chapter types are accepted."""
        for chapter_type in ["recipes", "intro", "index", "basics", "toc"]:
            chapter = ChapterTitles(chapter_type=chapter_type)
            assert chapter.chapter_type == chapter_type


class TestTOCEntry:
    """Tests for TOCEntry model."""

    def test_basic_entry(self) -> None:
        """TOCEntry stores chapter and recipe titles."""
        entry = TOCEntry(
            chapter_title="Main Courses",
            recipes=["Roasted Chicken", "Beef Stew"],
        )
        assert entry.chapter_title == "Main Courses"
        assert len(entry.recipes) == 2

    def test_empty_recipes_default(self) -> None:
        """recipes defaults to empty list."""
        entry = TOCEntry(chapter_title="Introduction")
        assert entry.recipes == []


class TestCookbookTOC:
    """Tests for CookbookTOC model."""

    @pytest.fixture
    def sample_toc(self) -> CookbookTOC:
        """Provide a sample TOC."""
        return CookbookTOC(
            chapters=[
                TOCEntry(chapter_title="Starters", recipes=["Soup", "Salad"]),
                TOCEntry(chapter_title="Mains", recipes=["Chicken", "Fish", "Beef"]),
                TOCEntry(chapter_title="Desserts", recipes=["Cake", "Pie"]),
            ]
        )

    def test_all_recipe_titles(self, sample_toc: CookbookTOC) -> None:
        """all_recipe_titles returns all recipes from all chapters."""
        all_titles = sample_toc.all_recipe_titles()
        assert len(all_titles) == 7
        assert "Soup" in all_titles
        assert "Chicken" in all_titles
        assert "Cake" in all_titles

    def test_recipes_for_chapter_exists(self, sample_toc: CookbookTOC) -> None:
        """recipes_for_chapter returns recipes for existing chapter."""
        mains = sample_toc.recipes_for_chapter("Mains")
        assert mains == ["Chicken", "Fish", "Beef"]

    def test_recipes_for_chapter_case_insensitive(self, sample_toc: CookbookTOC) -> None:
        """recipes_for_chapter is case-insensitive."""
        mains = sample_toc.recipes_for_chapter("MAINS")
        assert mains == ["Chicken", "Fish", "Beef"]

        mains_lower = sample_toc.recipes_for_chapter("mains")
        assert mains_lower == ["Chicken", "Fish", "Beef"]

    def test_recipes_for_chapter_not_found(self, sample_toc: CookbookTOC) -> None:
        """recipes_for_chapter returns empty list for non-existent chapter."""
        result = sample_toc.recipes_for_chapter("Nonexistent")
        assert result == []

    def test_empty_toc(self) -> None:
        """Empty TOC has no chapters."""
        toc = CookbookTOC()
        assert toc.chapters == []
        assert toc.all_recipe_titles() == []


class TestCookbookRecipes:
    """Tests for CookbookRecipes model."""

    def test_default_values(self) -> None:
        """CookbookRecipes has sensible defaults."""
        result = CookbookRecipes()
        assert result.recipes == []
        assert result.has_more is False
        assert result.last_content_marker is None

    def test_with_recipes(self) -> None:
        """CookbookRecipes can store multiple recipes."""
        recipe1 = MelaRecipe(
            title="Recipe 1",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1.", "Step 2."],
        )
        recipe2 = MelaRecipe(
            title="Recipe 2",
            ingredients=[IngredientGroup(title="", ingredients=["2 cups"])],
            instructions=["Do this.", "Do that."],
        )

        result = CookbookRecipes(
            recipes=[recipe1, recipe2],
            has_more=True,
            last_content_marker="...end of recipe 2...",
        )

        assert len(result.recipes) == 2
        assert result.has_more is True
        assert result.last_content_marker == "...end of recipe 2..."

    def test_max_15_recipes(self) -> None:
        """CookbookRecipes enforces max 15 recipes."""
        recipes = [
            MelaRecipe(
                title=f"Recipe {i}",
                ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
                instructions=["Step 1.", "Step 2."],
            )
            for i in range(16)
        ]

        with pytest.raises(ValidationError):
            CookbookRecipes(recipes=recipes)


class TestRecipeModelSerialization:
    """Tests for recipe model JSON serialization."""

    def test_recipe_to_dict(self) -> None:
        """Recipe can be serialized to dictionary."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="Main", ingredients=["1 cup flour"])],
            instructions=["Mix.", "Bake."],
            categories=[Category.Baking],
        )
        data = recipe.model_dump()

        assert data["title"] == "Test"
        assert len(data["ingredients"]) == 1
        assert data["categories"] == [Category.Baking]

    def test_recipe_to_json(self) -> None:
        """Recipe can be serialized to JSON string."""
        recipe = MelaRecipe(
            title="JSON Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 item"])],
            instructions=["Step 1.", "Step 2."],
        )
        json_str = recipe.model_dump_json()

        assert "JSON Test" in json_str
        assert '"title"' in json_str


# ============================================================================
# RecipeParser Tests
# ============================================================================


class TestRecipeParser:
    """Tests for RecipeParser class."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a sample MelaRecipe for mock responses."""
        return MelaRecipe(
            title="Chocolate Cake",
            ingredients=[IngredientGroup(title="", ingredients=["200g flour", "100g sugar"])],
            instructions=["Mix ingredients.", "Bake at 350F."],
        )

    def test_parser_init(self) -> None:
        """RecipeParser initializes with recipe text and model."""
        from unittest.mock import patch

        from mela_parser.parse import RecipeParser

        with patch("mela_parser.parse.OpenAI"):
            parser = RecipeParser("Test recipe text", model="gpt-5-mini")
            assert parser.recipe_text == "Test recipe text"
            assert parser.model == "gpt-5-mini"

    def test_parser_parse_success(self, sample_recipe) -> None:
        """RecipeParser.parse returns recipe on success."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import RecipeParser

        mock_response = MagicMock()
        mock_response.output_parsed = sample_recipe

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            parser = RecipeParser("Test recipe text")
            result = parser.parse()

            assert result.title == "Chocolate Cake"
            mock_client.responses.parse.assert_called_once()

    def test_parser_parse_none_response(self) -> None:
        """RecipeParser.parse raises on None response."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import RecipeParser

        mock_response = MagicMock()
        mock_response.output_parsed = None

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            parser = RecipeParser("Test recipe text")
            with pytest.raises(ValueError, match="returned None"):
                parser.parse()

    def test_parser_parse_openai_error(self) -> None:
        """RecipeParser.parse raises on OpenAI error."""
        from unittest.mock import MagicMock, patch

        from openai import OpenAIError

        from mela_parser.parse import RecipeParser

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.side_effect = OpenAIError("API Error")
            mock_openai.return_value = mock_client

            parser = RecipeParser("Test recipe text")
            with pytest.raises(OpenAIError):
                parser.parse()


# ============================================================================
# RecipeMarkerInserter Tests
# ============================================================================


class TestRecipeMarkerInserter:
    """Tests for RecipeMarkerInserter class."""

    def test_inserter_init(self) -> None:
        """RecipeMarkerInserter initializes with model and marker."""
        from unittest.mock import patch

        from mela_parser.parse import RecipeMarkerInserter

        with patch("mela_parser.parse.OpenAI"):
            inserter = RecipeMarkerInserter(model="gpt-5-mini")
            assert inserter.model == "gpt-5-mini"
            assert inserter.marker == "===RECIPE_START==="

    def test_insert_markers_success(self) -> None:
        """insert_markers returns marked text on success."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import RecipeMarkerInserter

        markdown = "# Cookbook\n\n# Chocolate Cake\nIngredients..."
        marked_text = "# Cookbook\n\n===RECIPE_START===\n# Chocolate Cake\nIngredients..."

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = marked_text
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=110, total_tokens=210)

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            inserter = RecipeMarkerInserter()
            result = inserter.insert_markers(markdown, book_title="Test Cookbook")

            assert "===RECIPE_START===" in result
            mock_client.chat.completions.create.assert_called_once()

    def test_insert_markers_empty_response(self) -> None:
        """insert_markers raises on empty response."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import RecipeMarkerInserter

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            inserter = RecipeMarkerInserter()
            with pytest.raises(ValueError, match="empty response"):
                inserter.insert_markers("Test markdown")

    def test_insert_markers_openai_error(self) -> None:
        """insert_markers raises on OpenAI error."""
        from unittest.mock import MagicMock, patch

        from openai import OpenAIError

        from mela_parser.parse import RecipeMarkerInserter

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = OpenAIError("API Error")
            mock_openai.return_value = mock_client

            inserter = RecipeMarkerInserter()
            with pytest.raises(OpenAIError):
                inserter.insert_markers("Test markdown")

    def test_split_by_markers(self) -> None:
        """split_by_markers returns list of recipe sections."""
        from unittest.mock import patch

        from mela_parser.parse import RecipeMarkerInserter

        marked_text = (
            "Introduction text here\n"
            "===RECIPE_START===\n"
            "# Recipe One\nIngredients...\n"
            "===RECIPE_START===\n"
            "# Recipe Two\nMore ingredients..."
        )

        with patch("mela_parser.parse.OpenAI"):
            inserter = RecipeMarkerInserter()
            sections = inserter.split_by_markers(marked_text)

            assert len(sections) == 2
            assert "# Recipe One" in sections[0]
            assert "# Recipe Two" in sections[1]

    def test_split_by_markers_empty_sections_filtered(self) -> None:
        """split_by_markers filters out empty sections."""
        from unittest.mock import patch

        from mela_parser.parse import RecipeMarkerInserter

        marked_text = (
            "Intro\n"
            "===RECIPE_START===\n"
            "   \n"  # Empty/whitespace section
            "===RECIPE_START===\n"
            "# Real Recipe\nContent"
        )

        with patch("mela_parser.parse.OpenAI"):
            inserter = RecipeMarkerInserter()
            sections = inserter.split_by_markers(marked_text)

            assert len(sections) == 1
            assert "# Real Recipe" in sections[0]


# ============================================================================
# CookbookParser Tests
# ============================================================================


class TestCookbookParser:
    """Tests for CookbookParser class."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a sample MelaRecipe for mock responses."""
        return MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup flour"])],
            instructions=["Step 1.", "Step 2."],
        )

    def test_parser_init(self, caplog) -> None:
        """CookbookParser initializes with model and logs."""
        import logging
        from unittest.mock import patch

        from mela_parser.parse import CookbookParser

        with patch("mela_parser.parse.OpenAI"):
            with caplog.at_level(logging.INFO):
                parser = CookbookParser(model="gpt-5-mini")

            assert parser.model == "gpt-5-mini"
            assert "gpt-5-mini" in caplog.text

    def test_parse_cookbook_success(self, sample_recipe) -> None:
        """parse_cookbook returns CookbookRecipes on success."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import CookbookParser, CookbookRecipes

        mock_response = MagicMock()
        mock_response.output_parsed = CookbookRecipes(
            recipes=[sample_recipe],
            has_more=False,
        )
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50, total_tokens=150)

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            parser = CookbookParser()
            result = parser.parse_cookbook("Cookbook content", book_title="Test Book")

            assert len(result.recipes) == 1
            assert result.recipes[0].title == "Test Recipe"
            assert result.has_more is False

    def test_parse_cookbook_none_response(self) -> None:
        """parse_cookbook raises on None response."""
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import CookbookParser

        mock_response = MagicMock()
        mock_response.output_parsed = None

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            parser = CookbookParser()
            with pytest.raises(ValueError, match="returned None"):
                parser.parse_cookbook("Cookbook content")

    def test_parse_cookbook_empty_recipes_warning(self, caplog) -> None:
        """parse_cookbook logs warning when no recipes extracted."""
        import logging
        from unittest.mock import MagicMock, patch

        from mela_parser.parse import CookbookParser, CookbookRecipes

        mock_response = MagicMock()
        mock_response.output_parsed = CookbookRecipes(recipes=[], has_more=False)
        mock_response.usage = None

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            parser = CookbookParser()
            with caplog.at_level(logging.WARNING):
                result = parser.parse_cookbook("Cookbook content")

            assert len(result.recipes) == 0
            assert "No recipes extracted" in caplog.text

    def test_parse_cookbook_openai_error(self) -> None:
        """parse_cookbook raises on OpenAI error."""
        from unittest.mock import MagicMock, patch

        from openai import OpenAIError

        from mela_parser.parse import CookbookParser

        with patch("mela_parser.parse.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.parse.side_effect = OpenAIError("API Error")
            mock_openai.return_value = mock_client

            parser = CookbookParser()
            with pytest.raises(OpenAIError):
                parser.parse_cookbook("Cookbook content")

    def test_build_extraction_prompt(self) -> None:
        """_build_extraction_prompt builds correct prompt format."""
        from unittest.mock import patch

        from mela_parser.parse import CookbookParser

        with patch("mela_parser.parse.OpenAI"):
            parser = CookbookParser()
            prompt = parser._build_extraction_prompt("Test content", "My Cookbook")

            assert "My Cookbook" in prompt
            assert "Test content" in prompt
            assert "Extract EVERY COMPLETE recipe" in prompt
