from enum import Enum
from typing import List, Optional
import logging

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, Field


class Category(str, Enum):
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
    title: str
    ingredients: List[str]

    class Config:
        extra = "forbid"


class MelaRecipe(BaseModel):
    title: str
    text: Optional[str] = None
    # Removed images field – images are now sourced from the raw input
    yield_: Optional[str] = Field(None, alias="yield")
    prepTime: Optional[int] = Field(None, description="Prep time in minutes")
    cookTime: Optional[int] = Field(None, description="Cook time in minutes")
    totalTime: Optional[int] = Field(None, description="Total time in minutes")
    ingredients: List[IngredientGroup]
    instructions: List[str]
    notes: Optional[str] = None
    categories: Optional[List[Category]] = None

    class Config:
        extra = "forbid"
        populate_by_name = True


class RecipeParser:
    def __init__(self, recipe_text: str, model: str = "gpt-5-nano"):
        self.recipe_text = recipe_text
        self.client = OpenAI()
        self.model = model

    def parse(self):
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
            return response.output_parsed
        except Exception as e:
            logging.error(f"Error in RecipeParser.parse with {self.model}: {e}")
            raise


class CookbookRecipes(BaseModel):
    """Schema for extracting multiple recipes from a cookbook at once."""

    recipes: List[MelaRecipe]

    class Config:
        extra = "forbid"


class RecipeMarkerInserter:
    """Inserts delimiters before each recipe in markdown for boundary detection."""

    def __init__(self, model: str = "gpt-5-nano"):
        self.client = OpenAI()
        self.model = model
        self.marker = "===RECIPE_START==="

    def insert_markers(self, markdown: str, book_title: str = "") -> str:
        """
        Insert markers before each recipe in the markdown.

        Args:
            markdown: Full cookbook markdown
            book_title: Optional book title for context

        Returns:
            Markdown with ===RECIPE_START=== markers inserted
        """
        title_context = f"Cookbook: {book_title}\n\n" if book_title else ""

        prompt = f"""{title_context}Insert the marker "===RECIPE_START===" immediately before EVERY recipe in this cookbook.

A recipe typically has:
- A title/name
- Ingredients with measurements (tablespoons, teaspoons, cups, grams, ml, etc.)
- Cooking instructions/steps (often numbered or in paragraphs)
- Often starts with "SERVES" or "MAKES" or "YIELD"
- Often preceded by a description or story

Rules:
- Insert ===RECIPE_START=== on its own line right before each recipe title
- Do NOT change any other text
- Do NOT remove or restructure anything
- ONLY insert the markers
- Work through the ENTIRE text systematically
- Every recipe must get a marker
- Skip non-recipe content (forewords, introductions, indexes)

{markdown}"""

        try:
            logging.info(f"Inserting recipe markers with {self.model} (text length: {len(markdown)} chars)")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # Note: gpt-5-nano only supports default temperature (1)
            )

            marked_text = response.choices[0].message.content

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

    def split_by_markers(self, marked_text: str) -> List[str]:
        """
        Split marked text into recipe sections.

        Args:
            marked_text: Text with ===RECIPE_START=== markers

        Returns:
            List of recipe text sections (without markers)
        """
        sections = marked_text.split(self.marker)

        # First section is pre-recipe content (skip it)
        recipe_sections = [s.strip() for s in sections[1:] if s.strip()]

        logging.info(f"Split into {len(recipe_sections)} recipe sections")

        return recipe_sections


class CookbookParser:
    """
    Unified parser that extracts all recipes from cookbook markdown in a single pass.
    Supports both GPT-5-nano and GPT-5-mini models.
    """

    def __init__(self, model: str = "gpt-5-nano"):
        """
        Initialize the parser.

        Args:
            model: Model to use ("gpt-5-nano" or "gpt-5-mini")
        """
        self.client = OpenAI()
        self.model = model
        logging.info(f"Initialized CookbookParser with model: {model}")

    def parse_cookbook(self, markdown_content: str, book_title: str = "") -> CookbookRecipes:
        """
        Extract all recipes from cookbook markdown in a single pass.

        Args:
            markdown_content: Full markdown content of the cookbook
            book_title: Title of the book (for context)

        Returns:
            CookbookRecipes object containing list of parsed recipes
        """
        prompt = self._build_extraction_prompt(markdown_content, book_title)

        try:
            logging.info(f"Parsing cookbook with {self.model} (content length: {len(markdown_content)} chars)")

            response = self.client.responses.parse(
                model=self.model,
                input=[
                    EasyInputMessageParam(role="user", content=prompt),
                ],
                text_format=CookbookRecipes,
            )

            recipes = response.output_parsed

            if not recipes or not recipes.recipes:
                logging.warning(f"No recipes extracted! Response type: {type(response)}")
                logging.warning(f"Response output_parsed: {response.output_parsed}")

            logging.info(f"Successfully extracted {len(recipes.recipes)} recipes")

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                try:
                    usage = response.usage
                    # Try different attribute names for different API versions
                    input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
                    output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None))
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
        """Build the extraction prompt for the LLM."""

        return f"""You are extracting recipes from the cookbook: "{book_title}"

TASK: Extract EVERY recipe you find in the text below. A recipe typically has:
- A title/name
- A list of ingredients
- Cooking instructions/steps

IMPORTANT:
- Extract ALL recipes, no matter how many there are
- If prep/cook time is not explicitly stated, leave those fields blank
- If yield/servings is not stated, leave it blank
- Group related ingredients together if they have section headings
- Categorize recipes appropriately (Breakfasts, Mains, Desserts, etc.)
- Skip non-recipe content like introductions, forewords, and tables of contents

START EXTRACTING:

{markdown_content}"""
