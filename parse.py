from enum import Enum
from typing import List, Optional

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
    def __init__(self, recipe_text: str):
        self.recipe_text = recipe_text
        self.client = OpenAI()

    def parse(self):
        prompt = (
            f"Extract the recipe:\n<recipe_text>{self.recipe_text}</recipe_text>\n\n"
            "If an optional value cannot be found, leave it blank, do not guess or put a placeholder like 'N/A'."
        )
        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=MelaRecipe,
            )
            return response.output_parsed
        except Exception as e:
            import logging

            logging.error(f"Error in RecipeParser.parse: {e}")
            raise
