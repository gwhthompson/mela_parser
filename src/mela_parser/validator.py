"""Recipe validation and quality scoring for mela_parser.

This module provides comprehensive validation of extracted recipes to ensure
quality and completeness. It implements scoring systems to detect:
- Incomplete recipes (missing components)
- Low-quality extractions (sparse content)
- Recipe continuations (multi-page recipes)
- Component recipes (sub-recipes that should be excluded)

The validator is used in the self-correction loop to identify recipes that
need improvement before being written to output.

Example:
    >>> validator = RecipeValidator()
    >>> score = validator.score_recipe(recipe)
    >>> if not score.is_high_quality:
    ...     print(f"Issues: {score.issues}")
    ...     # Trigger self-correction
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from .parse import IngredientGroup, MelaRecipe


@dataclass
class RecipeQualityScore:
    """Quality metrics for an extracted recipe.

    This class provides a comprehensive quality assessment of a recipe,
    identifying specific issues and computing an overall completeness score.

    Attributes:
        recipe_title: Title of the recipe being scored
        has_title: Whether recipe has a non-empty title
        ingredient_count: Total number of ingredients across all groups
        instruction_count: Number of instruction steps
        has_times: Whether recipe has any time information
        has_yield: Whether recipe has yield/serving information
        issues: List of specific quality issues found
        completeness_score: Overall quality score (0.0-1.0)
        is_continuation: Whether this appears to be a recipe continuation
        is_component: Whether this appears to be a component/sub-recipe
    """

    recipe_title: str
    has_title: bool
    ingredient_count: int
    instruction_count: int
    has_times: bool
    has_yield: bool
    issues: list[str]
    completeness_score: float
    is_continuation: bool = False
    is_component: bool = False

    @property
    def is_high_quality(self) -> bool:
        """Check if recipe meets high quality threshold.

        Returns:
            True if completeness score >= 0.8
        """
        return self.completeness_score >= 0.8

    @property
    def is_acceptable(self) -> bool:
        """Check if recipe meets minimum acceptable threshold.

        Returns:
            True if completeness score >= 0.6
        """
        return self.completeness_score >= 0.6

    def __str__(self) -> str:
        """Human-readable quality summary."""
        quality = (
            "HIGH" if self.is_high_quality else ("ACCEPTABLE" if self.is_acceptable else "LOW")
        )
        return (
            f"Quality: {quality} (score: {self.completeness_score:.2f}) - "
            f"{self.ingredient_count} ingredients, {self.instruction_count} instructions"
        )


class RecipeValidator:
    """Validates recipes and scores their quality.

    The validator checks recipes for completeness, detects common issues,
    and provides actionable feedback for self-correction.

    Example:
        >>> validator = RecipeValidator(min_ingredients=2, min_instructions=2)
        >>> score = validator.score_recipe(recipe)
        >>> if not score.is_acceptable:
        ...     print("Needs improvement:", score.issues)
    """

    def __init__(
        self,
        min_ingredients: int = 2,
        min_instructions: int = 2,
        min_title_length: int = 3,
    ) -> None:
        """Initialize validator with quality thresholds.

        Args:
            min_ingredients: Minimum number of ingredients for valid recipe
            min_instructions: Minimum number of instructions for valid recipe
            min_title_length: Minimum title length (characters)
        """
        self.min_ingredients = min_ingredients
        self.min_instructions = min_instructions
        self.min_title_length = min_title_length

    def score_recipe(self, recipe: MelaRecipe) -> RecipeQualityScore:
        """Score a recipe's quality and completeness.

        Analyzes all aspects of the recipe and computes a comprehensive
        quality score with specific issue identification.

        Args:
            recipe: Recipe to validate

        Returns:
            RecipeQualityScore with detailed analysis

        Example:
            >>> recipe = MelaRecipe(title="Test", ...)
            >>> score = validator.score_recipe(recipe)
            >>> print(f"Score: {score.completeness_score:.2f}")
            >>> print(f"Issues: {', '.join(score.issues)}")
        """
        issues: list[str] = []

        # Check title
        has_title = bool(recipe.title and len(recipe.title.strip()) >= self.min_title_length)
        if not has_title:
            issues.append("Missing or too short title")

        # Check for continuation indicators
        is_continuation = self.detect_continuation(recipe)
        if is_continuation:
            issues.append("Appears to be a recipe continuation")

        # Check for component recipe indicators
        is_component = self.detect_component_recipe(recipe)
        if is_component:
            issues.append("Appears to be a component/sub-recipe")

        # Count ingredients
        ingredient_count = sum(len(group.ingredients) for group in recipe.ingredients)
        if ingredient_count < self.min_ingredients:
            issues.append(f"Insufficient ingredients ({ingredient_count} < {self.min_ingredients})")

        # Check ingredient quality
        if not self.validate_ingredients(recipe.ingredients):
            issues.append("Ingredients lack measurements or quantities")

        # Count instructions
        instruction_count = len(recipe.instructions)
        if instruction_count < self.min_instructions:
            issues.append(
                f"Insufficient instructions ({instruction_count} < {self.min_instructions})"
            )

        # Check instruction quality
        if not self.validate_instructions(recipe.instructions):
            issues.append("Instructions are not actionable steps")

        # Check metadata
        has_times = any([recipe.prepTime, recipe.cookTime, recipe.totalTime])
        has_yield = bool(recipe.recipeYield)

        # Calculate completeness score (weighted)
        score_components = []

        # Title (weight: 0.15)
        score_components.append(0.15 if has_title else 0.0)

        # Ingredients (weight: 0.35)
        if ingredient_count >= self.min_ingredients:
            # Full points if >= min, partial if some but not enough
            score_components.append(0.35)
        elif ingredient_count > 0:
            score_components.append(0.35 * (ingredient_count / self.min_ingredients))
        else:
            score_components.append(0.0)

        # Instructions (weight: 0.35)
        if instruction_count >= self.min_instructions:
            score_components.append(0.35)
        elif instruction_count > 0:
            score_components.append(0.35 * (instruction_count / self.min_instructions))
        else:
            score_components.append(0.0)

        # Metadata (weight: 0.15)
        metadata_score = 0.0
        if has_yield:
            metadata_score += 0.075
        if has_times:
            metadata_score += 0.075
        score_components.append(metadata_score)

        completeness_score = sum(score_components)

        # Penalize continuations and components
        if is_continuation:
            completeness_score *= 0.5
        if is_component:
            completeness_score *= 0.7

        return RecipeQualityScore(
            recipe_title=recipe.title,
            has_title=has_title,
            ingredient_count=ingredient_count,
            instruction_count=instruction_count,
            has_times=has_times,
            has_yield=has_yield,
            issues=issues,
            completeness_score=completeness_score,
            is_continuation=is_continuation,
            is_component=is_component,
        )

    def detect_continuation(self, recipe: MelaRecipe) -> bool:
        """Detect if recipe is a continuation from a previous page.

        Args:
            recipe: Recipe to check

        Returns:
            True if recipe appears to be a continuation

        Example:
            >>> recipe = MelaRecipe(title="Roasted Chicken (continued)", ...)
            >>> validator.detect_continuation(recipe)
            True
        """
        if not recipe.title:
            return False

        title_lower = recipe.title.lower()

        # Check for continuation markers in title
        continuation_markers = [
            "continued",
            "(cont)",
            "(cont.)",
            "cont'd",
            "...continued",
            "- continued",
        ]

        if any(marker in title_lower for marker in continuation_markers):
            return True

        # Check if instructions start with high numbers (e.g., "4. Mix...")
        # indicating missing earlier steps
        if recipe.instructions:
            first_instruction = recipe.instructions[0].strip()
            # Check if starts with a number > 3
            if first_instruction and first_instruction[0].isdigit():
                step_num = int(first_instruction.split(".", 1)[0])
                if step_num > 3:
                    return True

        return False

    def detect_component_recipe(self, recipe: MelaRecipe) -> bool:
        """Detect if recipe is a component/sub-recipe that should be excluded.

        Uses the is_standalone_recipe field from the schema, which is set by the
        LLM during extraction when it has full context.

        Args:
            recipe: Recipe to check

        Returns:
            True if recipe appears to be a component (is_standalone_recipe=False)

        Example:
            >>> recipe = MelaRecipe(title="For the sauce:", is_standalone_recipe=False, ...)
            >>> validator.detect_component_recipe(recipe)
            True
        """
        # Use the LLM's classification from the schema
        return not recipe.is_standalone_recipe

    def validate_ingredients(self, ingredient_groups: list[IngredientGroup]) -> bool:
        """Validate that ingredients have proper measurements.

        Args:
            ingredient_groups: List of ingredient groups to validate

        Returns:
            True if ingredients appear to have measurements

        Example:
            >>> groups = [IngredientGroup(title="", ingredients=["2 cups flour", "1 egg"])]
            >>> validator.validate_ingredients(groups)
            True
        """
        if not ingredient_groups:
            return False

        # Count ingredients with measurements
        measured_count = 0
        total_count = 0

        for group in ingredient_groups:
            for ingredient in group.ingredients:
                total_count += 1
                # Simple heuristic: ingredient with a number likely has measurement
                if any(char.isdigit() for char in ingredient):
                    measured_count += 1

        # At least 50% should have measurements
        if total_count == 0:
            return False

        return (measured_count / total_count) >= 0.5

    def validate_instructions(self, instructions: list[str]) -> bool:
        """Validate that instructions are actionable steps.

        Args:
            instructions: List of instruction strings

        Returns:
            True if instructions appear to be proper steps

        Example:
            >>> instructions = ["Preheat oven", "Mix ingredients", "Bake"]
            >>> validator.validate_instructions(instructions)
            True
        """
        if not instructions:
            return False

        # Check that instructions have some minimum length
        # (not just "Mix" or "Bake")
        actionable_count = 0
        for instruction in instructions:
            # Instruction should have at least one verb and some detail
            if len(instruction.strip()) >= 10:
                actionable_count += 1

        # At least 50% should be detailed
        return (actionable_count / len(instructions)) >= 0.5

    def validate_title_similarity(
        self, extracted_title: str, expected_title: str, threshold: float = 0.85
    ) -> bool:
        """Check if extracted title matches expected title.

        Uses fuzzy string matching to allow for minor variations while
        catching major discrepancies.

        Args:
            extracted_title: Title extracted from recipe
            expected_title: Expected title (e.g., from TOC)
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            True if similarity >= threshold

        Example:
            >>> validator.validate_title_similarity(
            ...     "Roasted Chicken",
            ...     "Roast Chicken",
            ...     threshold=0.85
            ... )
            True
        """
        similarity = SequenceMatcher(None, extracted_title.lower(), expected_title.lower()).ratio()

        return similarity >= threshold


def fuzzy_match_titles(title1: str, title2: str, threshold: float = 0.90) -> tuple[bool, float]:
    """Check if two recipe titles are likely the same recipe.

    Uses fuzzy string matching to detect duplicates even with minor
    variations in spelling, punctuation, or wording.

    Args:
        title1: First recipe title
        title2: Second recipe title
        threshold: Similarity threshold for match (0.0-1.0)

    Returns:
        Tuple of (is_match, similarity_score)

    Example:
        >>> is_match, score = fuzzy_match_titles(
        ...     "Roasted Chicken with Lemon",
        ...     "Roast Chicken with Lemons"
        ... )
        >>> print(f"Match: {is_match}, Score: {score:.2f}")
        Match: True, Score: 0.93
    """
    similarity = SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
    return (similarity >= threshold, similarity)
