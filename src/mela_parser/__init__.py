"""
Mela Parser - Extract recipes from EPUB cookbooks to Mela format.

This package provides tools for parsing EPUB cookbook files and extracting
structured recipe data in a format compatible with the Mela recipe manager app.
"""

__version__ = "0.1.0"

from .parse import Category, CookbookParser, IngredientGroup, MelaRecipe, RecipeParser
from .recipe import RecipeProcessor

__all__ = [
    "Category",
    "CookbookParser",
    "IngredientGroup",
    "MelaRecipe",
    "RecipeParser",
    "RecipeProcessor",
]
