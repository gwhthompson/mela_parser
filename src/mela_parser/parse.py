"""Recipe parsing and extraction using OpenAI structured output.

This module provides classes and schemas for parsing recipes from cookbook text
using OpenAI's structured output API. It defines Pydantic models for recipe data
and implements parsers for both single recipes and bulk cookbook extraction.

The module supports:
- Single recipe parsing with detailed validation
- Bulk cookbook parsing with pagination
- Recipe marker insertion for boundary detection
- Multiple recipe extraction with completeness checking
"""

import logging
from enum import Enum

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    """Recipe categories for classification.

    These categories follow the Mela app's classification system and can be
    used to organize and filter recipes.
    """

    Breakfasts = "Breakfasts"
    Starters = "Starters"
    Soups = "Soups"
    Salads = "Salads"
    Mains = "Mains"
    Sides = "Sides"
    Desserts = "Desserts"
    Meat = "Meat"
    Seafood = "Seafood"
    Vegetarian = "Vegetarian"
    Vegan = "Vegan"
    Pasta = "Pasta"
    Drinks = "Drinks"
    Sauces = "Sauces"
    Baking = "Baking"
    Holiday = "Holiday"
    Italian = "Italian"
    Mexican = "Mexican"
    Indian = "Indian"
    Chinese = "Chinese"
    Thai = "Thai"
    Mediterranean = "Mediterranean"
    Japanese = "Japanese"
    MiddleEastern = "Middle Eastern"
    Greek = "Greek"


class IngredientGroup(BaseModel):
    """A group of ingredients with an optional section heading.

    Recipes often organize ingredients into sections (e.g., "For the dough",
    "For the filling"). This model captures both the section title and the
    ingredients within that section.
    """

    title: str = Field(
        description=(
            "Section heading for this group of ingredients "
            "(e.g., 'Main ingredients', 'For the sauce', 'To serve')"
        )
    )
    ingredients: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of ingredients with measurements. Each ingredient must include "
            "quantity and unit (e.g., '400g white fish fillets', '2 tbsp tahini', "
            "'1 red chilli, finely chopped')"
        ),
        examples=[
            [
                "400g white fish fillets",
                "2 tbsp tahini",
                "1 red chilli, finely chopped",
                "2 cloves garlic, minced",
            ]
        ],
    )

    model_config = ConfigDict(extra="forbid")


class MelaRecipe(BaseModel):
    """Complete recipe schema compatible with Mela recipe manager.

    This Pydantic model defines the structure for a complete recipe with all
    necessary fields for the Mela app. It includes strict validation to ensure
    recipe completeness and data quality.
    """

    title: str = Field(
        description=(
            "The complete recipe name exactly as written in the text. "
            "Do not add alternative names or translations."
        ),
        examples=["Chilli Fish with Tahini", "Roasted Cauliflower & Hazelnut Salad"],
    )
    text: str | None = Field(
        None,
        description=(
            "Optional introduction or description paragraph that appears before the ingredients"
        ),
    )
    images: list[str] | None = Field(
        None,
        description=(
            "Optional list of image paths from markdown (e.g., ['../images/pg_65.jpg']). "
            "Extract paths from markdown image syntax like "
            "![Recipe Name](../images/pg_65.jpg) that appear near this recipe."
        ),
        examples=[
            ["../images/pg_65.jpg"],
            ["../images/recipe_photo.jpg", "../images/plated_dish.jpg"],
        ],
    )
    # Using 'recipeYield' instead of 'yield' to avoid JSON schema issues
    # with reserved keywords
    recipeYield: str | None = Field(  # noqa: N815
        None,
        description=(
            "Number of servings or yield "
            "(e.g., 'Serves 4', '12 cookies', '1 loaf'). Leave null if not stated."
        ),
        examples=["Serves 4", "Makes 12", "6-8 servings"],
    )
    prepTime: int | None = Field(  # noqa: N815
        None,
        description=(
            "Preparation time in minutes. Convert hours to minutes "
            "(e.g., '1 hour' becomes 60). Leave null if not stated."
        ),
        examples=[15, 30, 60],
    )
    cookTime: int | None = Field(  # noqa: N815
        None,
        description=(
            "Cooking/baking time in minutes. Convert hours to minutes. Leave null if not stated."
        ),
        examples=[20, 45, 90],
    )
    totalTime: int | None = Field(  # noqa: N815
        None,
        description=("Total time from start to finish in minutes. Leave null if not stated."),
        examples=[35, 75, 150],
    )
    ingredients: list[IngredientGroup] = Field(
        ...,
        min_length=1,
        description=(
            "One or more groups of ingredients. Use a single group with empty title "
            "if no groupings. Each group must have at least one ingredient "
            "with measurements."
        ),
        examples=[
            [
                {"title": "Main ingredients", "ingredients": ["400g fish", "2 tbsp oil"]},
                {"title": "For the sauce", "ingredients": ["100ml tahini", "1 lemon"]},
            ]
        ],
    )
    instructions: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "Step-by-step cooking instructions. Each step should be a complete sentence "
            "describing one action. Must have at least 2 steps showing how to prepare "
            "and cook the dish."
        ),
        examples=[
            [
                "Preheat oven to 200°C (180°C fan).",
                "Season the fish with salt and pepper.",
                "Heat oil in a large pan over medium-high heat.",
                "Fry the fish for 3-4 minutes on each side until golden.",
                "Drizzle with tahini sauce and serve.",
            ]
        ],
    )
    notes: str | None = Field(
        None,
        description=(
            "Optional notes, tips, variations, or storage instructions that appear after the recipe"
        ),
    )
    categories: list[Category] | None = Field(
        None,
        description=(
            "Recipe categories (e.g., Mains, Desserts, Vegetarian). "
            "Select from the available enum values."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RecipeParser:
    """Parse a single recipe from text using OpenAI structured output.

    This parser uses OpenAI's structured output API to extract recipe data
    from unstructured text and return it as a validated MelaRecipe object.
    """

    def __init__(self, recipe_text: str, model: str = "gpt-5-nano") -> None:
        """Initialize the recipe parser.

        Args:
            recipe_text: Raw recipe text to parse
            model: OpenAI model to use for parsing. Defaults to "gpt-5-nano".
        """
        self.recipe_text = recipe_text
        self.client = OpenAI()
        self.model = model

    def parse(self) -> MelaRecipe:
        """Parse the recipe text into a structured MelaRecipe object.

        Returns:
            A validated MelaRecipe object containing all extracted recipe data.

        Raises:
            Exception: If the OpenAI API call fails or parsing errors occur.
        """
        prompt = (
            f"Extract the recipe from this text:\n\n{self.recipe_text}\n\n"
            "IMPORTANT:\n"
            "- If prep/cook time is not stated, leave those fields blank\n"
            "- If yield/servings is not stated, leave it blank\n"
            "- Do NOT guess or use placeholders like 'N/A'\n"
            "- Preserve ingredient groupings if they have section headings\n"
            "- Convert times to minutes (e.g., '1 hour' → 60 minutes)"
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=MelaRecipe,
            )
            parsed_result = response.output_parsed
            if parsed_result is None:
                raise ValueError("Recipe parsing returned None")
            return parsed_result
        except Exception as e:
            logging.error(f"Error in RecipeParser.parse with {self.model}: {e}")
            raise


class CookbookRecipes(BaseModel):
    """Schema for extracting multiple recipes from a cookbook section.

    This model is used for bulk extraction with pagination support. It can
    extract up to 15 recipes per batch and indicates whether more recipes
    exist beyond the current batch.
    """

    recipes: list[MelaRecipe] = Field(
        default_factory=list,
        max_length=15,
        description=(
            "Up to 15 complete recipes found in this batch. A complete recipe must have "
            "a title, ingredients with measurements, and cooking instructions "
            "(at least 2 steps). Extract in order of appearance."
        ),
    )
    has_more: bool = Field(
        default=False,
        description=(
            "True if there are more recipes after this batch that couldn't fit "
            "due to output token limits. False if this is the last batch or "
            "no more recipes exist."
        ),
    )

    model_config = ConfigDict(extra="forbid")


class RecipeMarkerInserter:
    """Insert delimiters before each recipe in markdown for boundary detection.

    This class uses an LLM to identify recipe boundaries in cookbook markdown
    and insert markers that can later be used to split the text into individual
    recipe sections.
    """

    def __init__(self, model: str = "gpt-5-nano") -> None:
        """Initialize the marker inserter.

        Args:
            model: OpenAI model to use. Defaults to "gpt-5-nano".
        """
        self.client = OpenAI()
        self.model = model
        self.marker = "===RECIPE_START==="

    def insert_markers(self, markdown: str, book_title: str = "") -> str:
        """Insert markers before each recipe in the markdown.

        Args:
            markdown: Full cookbook markdown content
            book_title: Optional book title for context to improve accuracy

        Returns:
            Markdown with ===RECIPE_START=== markers inserted before each recipe

        Raises:
            Exception: If the OpenAI API call fails.
        """
        title_context = f"Cookbook: {book_title}\n\n" if book_title else ""

        prompt = (
            f'{title_context}Insert the marker "===RECIPE_START===" '
            "immediately before EVERY recipe in this cookbook.\n\n"
            "A recipe typically has:\n"
            "- A title/name\n"
            "- Ingredients with measurements "
            "(tablespoons, teaspoons, cups, grams, ml, etc.)\n"
            "- Cooking instructions/steps (often numbered or in paragraphs)\n"
            '- Often starts with "SERVES" or "MAKES" or "YIELD"\n'
            "- Often preceded by a description or story\n\n"
            "Rules:\n"
            "- Insert ===RECIPE_START=== on its own line right before "
            "each recipe title\n"
            "- Do NOT change any other text\n"
            "- Do NOT remove or restructure anything\n"
            "- ONLY insert the markers\n"
            "- Work through the ENTIRE text systematically\n"
            "- Every recipe must get a marker\n"
            "- Skip non-recipe content (forewords, introductions, indexes)\n\n"
            f"{markdown}"
        )

        try:
            logging.info(
                f"Inserting recipe markers with {self.model} (text length: {len(markdown)} chars)"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # Note: gpt-5-nano only supports default temperature (1)
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Marker insertion returned empty response")

            marked_text = content

            # Count markers
            marker_count = marked_text.count(self.marker)
            logging.info(f"Inserted {marker_count} recipe markers")

            # Log token usage
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logging.info(
                    f"Token usage - Input: {usage.prompt_tokens}, "
                    f"Output: {usage.completion_tokens}, "
                    f"Total: {usage.total_tokens}"
                )

            return marked_text

        except Exception as e:
            logging.error(f"Error inserting markers with {self.model}: {e}")
            raise

    def split_by_markers(self, marked_text: str) -> list[str]:
        """Split marked text into recipe sections.

        Args:
            marked_text: Text with ===RECIPE_START=== markers

        Returns:
            List of recipe text sections (markers removed, pre-recipe content skipped)
        """
        sections = marked_text.split(self.marker)

        # First section is pre-recipe content (skip it)
        recipe_sections = [s.strip() for s in sections[1:] if s.strip()]

        logging.info(f"Split into {len(recipe_sections)} recipe sections")

        return recipe_sections


class CookbookParser:
    """Unified parser that extracts all recipes from cookbook markdown in a single pass.

    This parser is designed for bulk extraction of recipes from complete cookbook
    markdown. It supports both GPT-5-nano and GPT-5-mini models and uses OpenAI's
    structured output API for reliable recipe extraction.
    """

    def __init__(self, model: str = "gpt-5-nano") -> None:
        """Initialize the cookbook parser.

        Args:
            model: Model to use ("gpt-5-nano" or "gpt-5-mini"). Defaults to "gpt-5-nano".
        """
        self.client = OpenAI()
        self.model = model
        logging.info(f"Initialized CookbookParser with model: {model}")

    def parse_cookbook(self, markdown_content: str, book_title: str = "") -> CookbookRecipes:
        """Extract all recipes from cookbook markdown in a single pass.

        This method sends the entire cookbook markdown to the LLM with instructions
        to extract all complete recipes. It returns a CookbookRecipes object containing
        the list of parsed recipes and pagination information.

        Args:
            markdown_content: Full markdown content of the cookbook
            book_title: Title of the book (for context). Defaults to empty string.

        Returns:
            CookbookRecipes object containing list of parsed recipes and has_more flag

        Raises:
            Exception: If the OpenAI API call fails or parsing errors occur.
        """
        prompt = self._build_extraction_prompt(markdown_content, book_title)

        try:
            logging.info(
                f"Parsing cookbook with {self.model} "
                f"(content length: {len(markdown_content)} chars)"
            )

            response = self.client.responses.parse(
                model=self.model,
                input=[
                    EasyInputMessageParam(role="user", content=prompt),
                ],
                text_format=CookbookRecipes,
            )

            parsed_result = response.output_parsed
            if parsed_result is None:
                raise ValueError("Cookbook parsing returned None")

            recipes = parsed_result

            if not recipes.recipes:
                logging.warning(f"No recipes extracted! Response type: {type(response)}")
                logging.warning(f"Response output_parsed: {response.output_parsed}")

            logging.info(f"Successfully extracted {len(recipes.recipes)} recipes")

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                try:
                    usage = response.usage
                    # Try different attribute names for different API versions
                    input_tokens = getattr(
                        usage, "input_tokens", getattr(usage, "prompt_tokens", None)
                    )
                    output_tokens = getattr(
                        usage, "output_tokens", getattr(usage, "completion_tokens", None)
                    )
                    total_tokens = getattr(usage, "total_tokens", None)

                    if input_tokens is not None and output_tokens is not None:
                        logging.info(
                            f"Token usage - Input: {input_tokens}, "
                            f"Output: {output_tokens}, "
                            f"Total: {total_tokens or (input_tokens + output_tokens)}"
                        )
                except Exception as e:
                    logging.debug(f"Could not log token usage: {e}")

            return recipes

        except Exception as e:
            logging.error(f"Error in CookbookParser.parse_cookbook: {e}")
            raise

    def _build_extraction_prompt(self, markdown_content: str, book_title: str) -> str:
        """Build the extraction prompt for the LLM.

        Args:
            markdown_content: Full cookbook markdown
            book_title: Book title for context

        Returns:
            Formatted prompt string with instructions and content
        """
        return f"""You are extracting recipes from the cookbook: "{book_title}"

TASK: Extract EVERY COMPLETE recipe you find in the text below.

A COMPLETE recipe MUST have ALL of these:
1. A title/name
2. A list of ingredients with measurements (e.g., "200g flour", "2 tbsp olive oil")
3. Cooking instructions/steps (how to prepare the dish)

DO NOT EXTRACT:
- Section headers (e.g., "VEGETABLES", "DESSERTS", "MY FAVORITES")
- Recipe lists/overviews (e.g., "Ten ways to cook eggs" without full details)
- Incomplete recipes missing ingredients OR instructions
- Cross-references (e.g., "See page 45 for recipe")
- Recipe titles without the full recipe
- Ingredient lists without instructions
- General cooking tips or techniques
- Recipe continuations (titles with "continued", "(cont)", or appearing to start mid-recipe)

EXTRACTION RULES:
- ONLY extract if you can fill ingredients AND instructions fields
- If prep/cook time is not explicitly stated, leave those fields blank
- If yield/servings is not stated, leave it blank
- Group related ingredients together if they have section headings
- Categorize recipes appropriately (Breakfasts, Mains, Desserts, etc.)
- When unsure if something is a complete recipe, skip it

START EXTRACTING:

{markdown_content}"""
