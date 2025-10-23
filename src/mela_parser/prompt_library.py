#!/usr/bin/env python3
"""
Prompt Library for Chapter-Based Recipe Extraction
===================================================

This module contains versioned, optimized prompts for achieving 100% accuracy
in recipe extraction from cookbook EPUBs using a chapter-based approach.

Design Philosophy:
- Constitutional AI: Extract exactly what exists, never infer or label
- Explicit negative constraints: Tell the model what NOT to do
- Few-shot examples: Show correct behavior with concrete examples
- Structured output: Use XML tags for clear boundaries
- Fail-safe defaults: When uncertain, skip rather than guess
- Version control: Track prompt performance and enable A/B testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


# =============================================================================
# PROMPT VERSIONS AND METADATA
# =============================================================================


class PromptType(str, Enum):
    """Types of prompts in the library."""
    RECIPE_LIST_DISCOVERY = "recipe_list_discovery"
    CHAPTER_EXTRACTION = "chapter_extraction"
    PROMPT_IMPROVEMENT = "prompt_improvement"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a prompt version."""
    total_recipes: int = 0
    expected_recipes: int = 0
    correctly_extracted: int = 0
    missing_recipes: int = 0
    extra_recipes: int = 0
    title_modifications: int = 0
    accuracy_percent: float = 0.0
    precision_percent: float = 0.0
    recall_percent: float = 0.0

    def calculate(self) -> None:
        """Calculate derived metrics."""
        if self.expected_recipes > 0:
            self.recall_percent = (self.correctly_extracted / self.expected_recipes) * 100
        if self.total_recipes > 0:
            self.precision_percent = (self.correctly_extracted / self.total_recipes) * 100
        if self.expected_recipes > 0:
            self.accuracy_percent = (self.correctly_extracted / self.expected_recipes) * 100


@dataclass
class PromptVersion:
    """
    A versioned prompt with metadata and performance tracking.

    Attributes:
        version: Semantic version (e.g., "1.0.0", "1.1.0", "2.0.0")
        prompt_type: Type of prompt (discovery, extraction, improvement)
        prompt_text: The actual prompt text
        model: Target model (e.g., "gpt-5-mini", "gpt-5-nano")
        timestamp: When this version was created
        description: Human-readable description of changes
        temperature: Recommended temperature setting
        performance: Performance metrics from validation runs
        notes: Additional notes about this version
    """
    version: str
    prompt_type: PromptType
    prompt_text: str
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    temperature: float = 0.0
    performance: Optional[PerformanceMetrics] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "prompt_type": self.prompt_type.value,
            "prompt_text": self.prompt_text,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "temperature": self.temperature,
            "performance": self.performance.__dict__ if self.performance else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Deserialize from dictionary."""
        perf_data = data.get("performance")
        performance = PerformanceMetrics(**perf_data) if perf_data else None

        return cls(
            version=data["version"],
            prompt_type=PromptType(data["prompt_type"]),
            prompt_text=data["prompt_text"],
            model=data["model"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data.get("description", ""),
            temperature=data.get("temperature", 0.0),
            performance=performance,
            notes=data.get("notes", ""),
        )


# =============================================================================
# RECIPE LIST DISCOVERY PROMPT (for gpt-5-mini)
# =============================================================================

RECIPE_LIST_DISCOVERY_PROMPT_V1_0 = """You are extracting the master list of recipe titles from a cookbook.

<task>
Extract ONLY the unique recipe titles from the potential recipe list sections below.
Return a clean list with one recipe title per line.
</task>

<input>
You will receive combined sections that may contain:
- Table of contents with recipe names and page numbers
- Recipe indexes
- Chapter summaries
- Navigation links
- Section headers (like "VEGETABLES", "DESSERTS", "CONTENTS")
{input_sections}
</input>

<rules>
REMOVE these items:
- Section headers and chapter names (e.g., "Contents", "Index", "Vegetables", "About the Author")
- Page numbers and navigation elements
- Duplicate recipe titles
- Non-recipe links (e.g., "Introduction", "Acknowledgments", "Metric Conversions")
- Formatting artifacts (bullets, numbers, dashes)

PRESERVE these items:
- Recipe titles EXACTLY as written - do NOT modify spelling, capitalization, or punctuation
- Special characters and accents in recipe names
- Numbers that are part of recipe titles (e.g., "10-minute pasta")
- Ampersands, hyphens, and other punctuation within titles

CRITICAL CONSTRAINT:
- Output ONLY the recipe titles, one per line
- NO commentary, NO labels, NO explanations
- If a title appears multiple times, include it ONLY ONCE
</rules>

<examples>
EXAMPLE 1 - Good:
Input sections contain:
  "Contents"
  "[Roasted chicken with lemon](#page45)"
  "[Chocolate cake](#page89)"
  "[Introduction](#intro)"
  "[Roasted chicken with lemon](#page45)"

Correct output:
Roasted chicken with lemon
Chocolate cake

EXAMPLE 2 - Bad (DO NOT DO THIS):
Roasted chicken with lemon (main course)
Chocolate cake - page 89
See Introduction

Why this is wrong:
- Added "(main course)" label
- Added "- page 89"
- Included non-recipe "Introduction"

EXAMPLE 3 - Good (preserving exact titles):
Input: "Na'ama's Fattoush", "A'ja (bread fritters)", "Mac and greens"
Correct output:
Na'ama's Fattoush
A'ja (bread fritters)
Mac and greens

Notice: Preserved apostrophes, parentheses, and "and" vs "&"
</examples>

<output_format>
Return titles in this exact format:
- One title per line
- No numbering, bullets, or prefixes
- No page numbers or suffixes
- Preserve exact spelling and capitalization
</output_format>

Begin extraction:
"""

RECIPE_LIST_DISCOVERY_PROMPT_V1_1 = """You are a precise recipe title extractor for cookbook digitization.

<objective>
Extract the complete, unique list of recipe titles from this cookbook's table of contents
and index sections. Achieve 100% accuracy by preserving EXACT titles without any modifications.
</objective>

<input>
The sections below are from a cookbook's navigation (TOC, indexes, chapter lists).
They contain a mix of recipe titles, section headers, and non-recipe links.

{input_sections}
</input>

<extraction_rules>
1. INCLUDE:
   ✓ Recipe titles exactly as written
   ✓ Each unique recipe only once
   ✓ All punctuation: apostrophes ('), parentheses (), hyphens (-), ampersands (&)
   ✓ All capitalization as shown
   ✓ Numbers that are part of titles

2. EXCLUDE:
   ✗ Section headers: "Contents", "Index", "Vegetables", "Desserts", "Chapter 1"
   ✗ Navigation: "Previous", "Next", "Home", "Introduction", "Acknowledgments"
   ✗ Page numbers or references: "p.45", "page 89", "#section2"
   ✗ Duplicates (same title appearing multiple times)
   ✗ Formatting artifacts: [brackets], (page refs), bullet points

3. FORBIDDEN MODIFICATIONS:
   ✗ NEVER add labels: "(main)", "(partial)", "(continued)", "(duplicate)"
   ✗ NEVER add descriptions or categories
   ✗ NEVER "fix" or standardize spelling/capitalization
   ✗ NEVER add or remove punctuation
   ✗ NEVER translate or paraphrase

4. CONSTITUTIONAL PRINCIPLE:
   "Extract exactly what exists. Never infer, never label, never enhance."
</extraction_rules>

<examples>
Example 1 - Mixed content with duplicates:
<input_text>
CONTENTS
Soups and Starters
[Tomato & sourdough soup](#recipe1)
[Watercress & chickpea soup with rose water & ras el hanout](#recipe2)
Main Courses
[Roasted chicken with clementines & arak](#recipe3)
[Tomato & sourdough soup](#recipe1)  <!-- duplicate -->
About the Author
</input_text>

<correct_output>
Tomato & sourdough soup
Watercress & chickpea soup with rose water & ras el hanout
Roasted chicken with clementines & arak
</correct_output>

<incorrect_output>
CONTENTS (wrong - section header)
Tomato and sourdough soup (wrong - changed "&" to "and")
Watercress & chickpea soup (wrong - truncated title)
Roasted chicken with clementines & arak (partial) (wrong - added label)
</incorrect_output>

Example 2 - Preserving exact punctuation:
<input_text>
[Na'ama's Fattoush](#r1)
[A'ja (bread fritters)](#r2)
[Swiss chard with tahini, yoghurt & buttered pine nuts](#r3)
</input_text>

<correct_output>
Na'ama's Fattoush
A'ja (bread fritters)
Swiss chard with tahini, yoghurt & buttered pine nuts
</correct_output>

Notice preservation of:
- Apostrophes: Na'ama's, A'ja
- Parentheses: (bread fritters)
- Commas: tahini, yoghurt
- Ampersands: &
- British spelling: yoghurt (not yogurt)
</examples>

<output_format>
Return ONLY recipe titles, one per line, with NO additional text:
Title One
Title Two
Title Three
</output_format>

<quality_check>
Before returning results, verify:
□ No section headers included
□ No page numbers or navigation
□ No duplicate titles
□ No added labels or suffixes
□ Exact spelling and punctuation preserved
□ One title per line, clean format
</quality_check>

Extract recipe titles now:
"""


# =============================================================================
# CHAPTER EXTRACTION PROMPT (for gpt-5-nano)
# =============================================================================

CHAPTER_EXTRACTION_PROMPT_V1_0 = """You are extracting complete recipes from a cookbook chapter.

<task>
Extract ALL complete recipes from the chapter markdown below.
A complete recipe MUST have a title, ingredients list, AND cooking instructions.
</task>

<input>
{chapter_markdown}
</input>

<extraction_criteria>
A COMPLETE recipe has ALL THREE components:
1. Title/name
2. Ingredients with measurements (e.g., "200g flour", "2 tbsp olive oil", "1 onion, chopped")
3. Instructions/method (step-by-step cooking directions)

SKIP these (DO NOT extract):
- Recipe titles without full ingredients or instructions
- Partial recipes or recipe fragments
- Cross-references (e.g., "See recipe on page 45")
- Recipe continuations (titles with "continued", "(cont)", or mid-recipe sections)
- Ingredient lists without instructions
- Section headers (e.g., "SOUPS", "DESSERTS")
- Recipe lists or overviews without full details
</extraction_criteria>

<critical_rules>
1. TITLE ACCURACY:
   - Use the EXACT title from the text
   - NEVER add suffixes like "(partial)", "(duplicate)", "(continued)", "(complete)"
   - NEVER add descriptions or categories to titles
   - If title appears multiple times, extract the recipe only once

2. COMPLETENESS:
   - Extract ONLY if you can fill: title AND ingredients AND instructions
   - If any component is missing, SKIP the recipe entirely
   - When uncertain if a recipe is complete, SKIP IT

3. METADATA:
   - Leave prep/cook/total time blank if not explicitly stated
   - Leave yield/servings blank if not stated
   - DO NOT guess or infer missing information
   - DO NOT use placeholders like "N/A" or "Unknown"

4. CONSTITUTIONAL PRINCIPLE:
   "Extract exactly what exists as a complete recipe. Never infer, never label, never guess."
</critical_rules>

<examples>
EXAMPLE 1 - Complete recipe (EXTRACT THIS):
<text>
Chocolate Chip Cookies

Makes 24 cookies
Prep time: 15 minutes
Cook time: 12 minutes

Ingredients:
- 200g butter
- 150g sugar
- 2 eggs
- 300g flour
- 200g chocolate chips

Method:
1. Cream butter and sugar
2. Beat in eggs
3. Fold in flour and chocolate chips
4. Bake at 180°C for 12 minutes
</text>

Correct extraction:
- title: "Chocolate Chip Cookies"
- ingredients: ["200g butter", "150g sugar", "2 eggs", "300g flour", "200g chocolate chips"]
- instructions: ["Cream butter and sugar", "Beat in eggs", ...]
- yield: "24 cookies"
- prepTime: 15 (minutes)
- cookTime: 12 (minutes)

EXAMPLE 2 - Incomplete recipe (DO NOT EXTRACT):
<text>
Amazing Pasta Sauce

This sauce is perfect for any pasta dish. See the full recipe on page 145.
</text>

Why SKIP: No ingredients, no instructions - just a teaser/cross-reference

EXAMPLE 3 - Recipe continuation (DO NOT EXTRACT):
<text>
Beef Bourguignon (continued)

5. Simmer for 2 hours
6. Add pearl onions
7. Serve hot
</text>

Why SKIP: Title has "(continued)" - this is the middle/end of a recipe, not the start

EXAMPLE 4 - Wrong title modification (DO NOT DO THIS):
<text>
Roasted Vegetables

Ingredients: carrots, potatoes, olive oil
Instructions: Roast at 200°C for 30 minutes
</text>

WRONG extraction:
- title: "Roasted Vegetables (partial)"  ← WRONG: Never add labels!

CORRECT extraction:
- title: "Roasted Vegetables"  ← Use exact title from text
</examples>

{expected_titles_section}

Extract all complete recipes now:
"""

CHAPTER_EXTRACTION_PROMPT_V1_1 = """You are a precise recipe extractor for cookbook digitization. Your goal is 100% accuracy.

<objective>
Extract complete recipes from this cookbook chapter.
Use EXACT titles. Never add labels or modify text.
Skip anything incomplete or ambiguous.
</objective>

<input>
<chapter>
{chapter_markdown}
</chapter>
</input>

<complete_recipe_definition>
A recipe is COMPLETE if and ONLY if it contains ALL THREE:

1. TITLE
   - Clear recipe name (e.g., "Chocolate Cake", "Tomato Soup")
   - Not a section header (e.g., "DESSERTS", "CHAPTER 3")
   - Not a continuation marker (e.g., "Recipe continued", "...continued from page 12")

2. INGREDIENTS
   - List of ingredients with measurements
   - Examples: "2 cups flour", "500g chicken", "3 tbsp olive oil", "1 onion, diced"
   - Must have at least 2 ingredients with quantities
   - Generic lists without measurements don't count

3. INSTRUCTIONS
   - Step-by-step cooking/preparation directions
   - Examples: "Preheat oven to 180°C", "Mix flour and sugar", "Bake for 30 minutes"
   - Must have at least 2 instruction steps
   - Generic advice doesn't count as instructions

If ANY component is missing or unclear, DO NOT extract the recipe.
</complete_recipe_definition>

<exclusion_rules>
DO NOT EXTRACT these items:

1. INCOMPLETE RECIPES:
   ✗ Title only without ingredients/instructions
   ✗ Ingredients only without instructions
   ✗ Instructions only without full context
   ✗ Partial recipes missing key components

2. CROSS-REFERENCES:
   ✗ "See recipe on page 45"
   ✗ "Recipe continues on next page"
   ✗ "Full recipe in previous chapter"
   ✗ References to other recipes

3. RECIPE CONTINUATIONS:
   ✗ Titles ending with "continued", "(cont)", "(continued from p.X)"
   ✗ Instruction lists that appear mid-recipe (steps 5-10 without steps 1-4)
   ✗ Orphaned ingredient sections

4. NON-RECIPES:
   ✗ Section headers ("VEGETABLES", "DESSERTS", "CHAPTER 1")
   ✗ Recipe lists/indexes (titles without full recipes)
   ✗ Cooking tips or techniques
   ✗ Ingredient descriptions
   ✗ Forewords, introductions, author notes

5. VARIATIONS OR MENTIONS:
   ✗ "For a vegetarian version, see..."
   ✗ "This pairs well with [Recipe Name]"
   ✗ Brief mentions of recipes in narrative text
</exclusion_rules>

<title_accuracy_rules>
CRITICAL: Recipe title accuracy is paramount.

✓ DO:
- Copy the exact title character-by-character
- Preserve all punctuation: apostrophes ('), hyphens (-), ampersands (&), commas (,)
- Preserve capitalization exactly as written
- Keep parenthetical content: "A'ja (bread fritters)"
- Keep special characters and accents: "Na'ama's Fattoush"

✗ NEVER:
- Add labels: "(partial)", "(duplicate)", "(continued)", "(complete)", "(from chapter X)"
- Add categories: "(main course)", "(dessert)", "(vegetarian)"
- Add descriptions: "- a delicious recipe", "- traditional style"
- Standardize spelling: Keep "yoghurt" if written, don't change to "yogurt"
- Remove punctuation: Keep "Mac and greens", don't change to "Mac & greens"
- Add clarifications: Don't change "Brick" to "Brick pastry" even if that's what it is

CONSTITUTIONAL PRINCIPLE:
"The title in the source text is sacred. Extract it exactly, or don't extract at all."
</title_accuracy_rules>

<metadata_handling>
For prep/cook/total time and yield:
- Extract ONLY if explicitly stated in the recipe
- Convert times to minutes: "1 hour" → 60, "30 mins" → 30, "1hr 15min" → 75
- Leave blank if not stated - DO NOT guess, infer, or use defaults
- DO NOT use placeholders: Never use "N/A", "Unknown", "0", or "See recipe"

Ingredient grouping:
- If ingredients have section headings, preserve them
- Examples: "For the dough:", "For the filling:", "Garnish:"
- Create separate ingredient groups with appropriate titles
</metadata_handling>

<examples>
CORRECT EXTRACTION #1:
<source>
Roasted Chicken with Lemon

Serves 4
Prep: 15 minutes
Cook: 1 hour

Ingredients:
- 1 whole chicken (1.5kg)
- 2 lemons, halved
- 3 tbsp olive oil
- Salt and pepper

Instructions:
1. Preheat oven to 200°C
2. Rub chicken with oil, salt, and pepper
3. Stuff cavity with lemon halves
4. Roast for 1 hour until golden
5. Rest for 10 minutes before serving
</source>

EXTRACT AS:
{
  "title": "Roasted Chicken with Lemon",
  "yield": "4",
  "prepTime": 15,
  "cookTime": 60,
  "ingredients": [
    {
      "title": "",
      "ingredients": [
        "1 whole chicken (1.5kg)",
        "2 lemons, halved",
        "3 tbsp olive oil",
        "Salt and pepper"
      ]
    }
  ],
  "instructions": [
    "Preheat oven to 200°C",
    "Rub chicken with oil, salt, and pepper",
    "Stuff cavity with lemon halves",
    "Roast for 1 hour until golden",
    "Rest for 10 minutes before serving"
  ]
}

INCORRECT EXTRACTION #2 (DO NOT DO THIS):
<source>
Quick Pasta Sauce

This sauce is incredibly versatile. For the full recipe, see page 87.
</source>

WRONG: Extracting this as:
{
  "title": "Quick Pasta Sauce (partial)",
  "ingredients": ["See page 87"],
  "instructions": ["See full recipe on page 87"]
}

Why WRONG:
- Added "(partial)" label to title
- No actual ingredients or instructions present
- This is a cross-reference, not a complete recipe

CORRECT: Skip this entirely, don't extract.

INCORRECT EXTRACTION #3 (DO NOT DO THIS):
<source>
Chocolate Brownies (continued from previous page)

4. Pour batter into pan
5. Bake for 25 minutes
6. Cool before cutting
</source>

WRONG: Extracting with title "Chocolate Brownies"

Why WRONG:
- This is a recipe continuation (instructions 4-6, missing 1-3)
- Missing ingredients section (on previous page)
- Title has "(continued from previous page)"
- Not a complete, self-contained recipe

CORRECT: Skip this entirely, don't extract.

CORRECT EXTRACTION #4 (Exact title preservation):
<source>
Swiss chard with tahini, yoghurt & buttered pine nuts

A traditional Middle Eastern preparation.

Serves 4 | Prep: 10 min | Cook: 15 min

Ingredients:
- 500g Swiss chard, chopped
- 3 tbsp tahini
- 100g Greek yoghurt
- 50g pine nuts
- 2 tbsp butter

Method:
1. Blanch Swiss chard in boiling water for 3 minutes
2. Toast pine nuts in butter until golden
3. Mix tahini and yoghurt
4. Plate chard, drizzle with tahini yoghurt, top with pine nuts
</source>

EXTRACT AS:
{
  "title": "Swiss chard with tahini, yoghurt & buttered pine nuts",
  // Note: Preserved commas, ampersand, British "yoghurt" spelling
  "yield": "4",
  "prepTime": 10,
  "cookTime": 15,
  "ingredients": [...],
  "instructions": [...]
}

DO NOT extract as:
- "Swiss chard with tahini, yogurt & buttered pine nuts" (wrong: changed yoghurt)
- "Swiss Chard With Tahini, Yoghurt & Buttered Pine Nuts" (wrong: changed capitalization)
- "Swiss chard with tahini and yoghurt and buttered pine nuts" (wrong: changed &)
</examples>

{expected_titles_section}

<quality_checks>
Before finalizing extraction, verify EACH recipe:
□ Title copied exactly character-by-character from source
□ No labels added to title (no suffixes or prefixes)
□ Has actual ingredients with measurements (not "see page X")
□ Has actual step-by-step instructions (not "continued from...")
□ Times converted to minutes if provided, blank if not
□ Yield preserved exactly if provided, blank if not
□ No placeholder values used for missing data
</quality_checks>

<failure_modes>
When in doubt, SKIP the recipe. It's better to miss a recipe than extract it incorrectly.

Common failure modes to avoid:
- Extracting recipe teasers or previews as complete recipes
- Extracting recipe continuations that span pages
- Adding clarifying labels to titles
- Standardizing or "fixing" title spelling/punctuation
- Guessing at missing metadata (times, yields)
- Extracting partial recipes hoping they're "good enough"

Remember: 100% accuracy means EXACT titles and COMPLETE recipes only.
</failure_modes>

Extract all complete recipes now:
"""

# Template for expected titles section
EXPECTED_TITLES_TEMPLATE = """
<expected_recipes>
The following recipes are expected to be in this chapter based on the book's table of contents.
Extract ONLY these specific recipes if they appear COMPLETE in the text:

{title_list}

CRITICAL:
- Use the EXACT titles from this list
- Only extract if the recipe is COMPLETE (title + ingredients + instructions)
- If you find a recipe with one of these titles but it's incomplete, SKIP IT
- If you find a complete recipe NOT on this list, SKIP IT (it may be in another chapter)
</expected_recipes>
"""


# =============================================================================
# PROMPT IMPROVEMENT META-PROMPT (for automatic iteration)
# =============================================================================

PROMPT_IMPROVEMENT_META_PROMPT_V1_0 = """You are an expert prompt engineer specializing in LLM optimization for 100% accuracy.

<task>
Analyze the current prompt and validation results, then suggest specific improvements
to eliminate extraction errors and achieve perfect accuracy.
</task>

<current_prompt>
{current_prompt}
</current_prompt>

<validation_results>
Total recipes extracted: {total_extracted}
Expected recipes: {expected_count}
Correctly matched: {correct_count}

Missing recipes (expected but not found):
{missing_list}

Extra recipes (found but not expected):
{extra_list}

Title modifications (incorrect titles):
{modified_list}
</validation_results>

<failure_pattern_analysis>
Analyze the errors and identify patterns:

1. MISSING RECIPES:
   - Are they incomplete in the source (missing ingredients/instructions)?
   - Are they cross-references or continuations being skipped correctly?
   - Are they complete but LLM is too conservative?

2. EXTRA RECIPES:
   - Are these recipe teasers/previews extracted incorrectly?
   - Are these section headers misidentified as recipes?
   - Are these recipe continuations extracted incorrectly?

3. TITLE MODIFICATIONS:
   - What suffixes/prefixes were added (e.g., "(partial)", "(continued)")?
   - Were punctuation or spelling changed?
   - Were categories or descriptions added?

4. ROOT CAUSES:
   - Ambiguous instructions allowing interpretation?
   - Insufficient negative examples?
   - Missing specific constraints?
   - Unclear definition of "complete recipe"?
</failure_pattern_analysis>

<improvement_strategies>
For each error pattern, suggest specific prompt changes:

1. ADD EXPLICIT CONSTRAINTS:
   - Add negative examples showing what NOT to do
   - Add constitutional AI principles
   - Add specific checks before extraction

2. STRENGTHEN DEFINITIONS:
   - Make "complete recipe" definition more precise
   - Add minimum requirements (e.g., "at least 2 ingredients")
   - Clarify edge cases

3. IMPROVE EXAMPLES:
   - Add examples matching observed failure modes
   - Show correct handling of continuations, cross-refs, etc.
   - Demonstrate exact title preservation

4. ADD VERIFICATION STEPS:
   - Add pre-extraction quality checks
   - Add title accuracy verification
   - Add completeness verification
</improvement_strategies>

<output_format>
Return a structured JSON object with improvement suggestions:

{
  "error_summary": {
    "missing_recipes_pattern": "Brief description of why recipes were missed",
    "extra_recipes_pattern": "Brief description of why extra recipes were included",
    "title_modification_pattern": "Brief description of title changes made"
  },
  "suggested_improvements": [
    {
      "section": "Section of prompt to modify (e.g., 'extraction_criteria', 'title_rules')",
      "issue": "Specific problem being addressed",
      "current_text": "Existing prompt text causing the issue",
      "improved_text": "Suggested replacement text",
      "rationale": "Why this change will improve accuracy"
    }
  ],
  "new_examples_to_add": [
    {
      "scenario": "Description of edge case",
      "example_text": "Example text to add to prompt",
      "correct_behavior": "What LLM should do",
      "incorrect_behavior": "What to avoid"
    }
  ],
  "new_constraints_to_add": [
    "Specific constraint to add to rule section"
  ],
  "expected_impact": "Expected improvement in accuracy percentage"
}
</output_format>

Analyze and provide improvement suggestions:
"""


# =============================================================================
# PROMPT LIBRARY CLASS
# =============================================================================


class PromptLibrary:
    """
    Manages versioned prompts with performance tracking and A/B testing.

    Features:
    - Load/save prompts to JSON
    - Version management
    - Performance tracking
    - A/B testing support
    - Rollback capability
    - Best prompt selection based on metrics
    """

    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize the prompt library.

        Args:
            library_path: Path to JSON file storing prompt versions.
                         Defaults to ./prompt_versions.json
        """
        self.library_path = library_path or Path("prompt_versions.json")
        self.prompts: Dict[str, List[PromptVersion]] = {
            pt.value: [] for pt in PromptType
        }

        # Load existing library if available
        if self.library_path.exists():
            self.load()
        else:
            # Initialize with default prompts
            self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize library with default prompt versions."""
        # Recipe List Discovery prompts
        self.add_prompt(PromptVersion(
            version="1.0.0",
            prompt_type=PromptType.RECIPE_LIST_DISCOVERY,
            prompt_text=RECIPE_LIST_DISCOVERY_PROMPT_V1_0,
            model="gpt-5-mini",
            description="Initial recipe list discovery prompt with basic rules",
            temperature=0.0,
        ))

        self.add_prompt(PromptVersion(
            version="1.1.0",
            prompt_type=PromptType.RECIPE_LIST_DISCOVERY,
            prompt_text=RECIPE_LIST_DISCOVERY_PROMPT_V1_1,
            model="gpt-5-mini",
            description="Enhanced with constitutional AI, quality checks, and more examples",
            temperature=0.0,
        ))

        # Chapter Extraction prompts
        self.add_prompt(PromptVersion(
            version="1.0.0",
            prompt_type=PromptType.CHAPTER_EXTRACTION,
            prompt_text=CHAPTER_EXTRACTION_PROMPT_V1_0,
            model="gpt-5-nano",
            description="Initial chapter extraction with basic completeness rules",
            temperature=0.0,
        ))

        self.add_prompt(PromptVersion(
            version="1.1.0",
            prompt_type=PromptType.CHAPTER_EXTRACTION,
            prompt_text=CHAPTER_EXTRACTION_PROMPT_V1_1,
            model="gpt-5-nano",
            description="Enhanced with detailed examples, exclusion rules, and quality checks",
            temperature=0.0,
        ))

        # Prompt Improvement meta-prompt
        self.add_prompt(PromptVersion(
            version="1.0.0",
            prompt_type=PromptType.PROMPT_IMPROVEMENT,
            prompt_text=PROMPT_IMPROVEMENT_META_PROMPT_V1_0,
            model="gpt-5-mini",
            description="Meta-prompt for analyzing failures and suggesting improvements",
            temperature=0.3,  # Slightly higher temp for creative suggestions
        ))

    def add_prompt(self, prompt: PromptVersion) -> None:
        """Add a new prompt version to the library."""
        self.prompts[prompt.prompt_type.value].append(prompt)

    def get_prompt(
        self,
        prompt_type: PromptType,
        version: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[PromptVersion]:
        """
        Get a specific prompt version.

        Args:
            prompt_type: Type of prompt to retrieve
            version: Specific version (e.g., "1.1.0"). If None, returns latest.
            model: Filter by target model. If None, any model matches.

        Returns:
            PromptVersion or None if not found
        """
        prompts = self.prompts.get(prompt_type.value, [])

        # Filter by model if specified
        if model:
            prompts = [p for p in prompts if p.model == model]

        if not prompts:
            return None

        # Get specific version or latest
        if version:
            matches = [p for p in prompts if p.version == version]
            return matches[0] if matches else None
        else:
            # Return latest version (sort by version string)
            return max(prompts, key=lambda p: p.version)

    def get_best_prompt(
        self,
        prompt_type: PromptType,
        model: Optional[str] = None,
        metric: str = "accuracy_percent",
    ) -> Optional[PromptVersion]:
        """
        Get the best performing prompt based on metrics.

        Args:
            prompt_type: Type of prompt
            model: Filter by model
            metric: Metric to optimize ("accuracy_percent", "recall_percent", etc.)

        Returns:
            Best performing PromptVersion or None
        """
        prompts = self.prompts.get(prompt_type.value, [])

        if model:
            prompts = [p for p in prompts if p.model == model]

        # Filter to prompts with performance data
        prompts = [p for p in prompts if p.performance is not None]

        if not prompts:
            return None

        # Return prompt with best metric
        return max(prompts, key=lambda p: getattr(p.performance, metric, 0))

    def update_performance(
        self,
        prompt_type: PromptType,
        version: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Update performance metrics for a prompt version.

        Args:
            prompt_type: Type of prompt
            version: Version to update
            metrics: Performance metrics
        """
        prompt = self.get_prompt(prompt_type, version)
        if prompt:
            metrics.calculate()  # Calculate derived metrics
            prompt.performance = metrics
            self.save()

    def list_versions(
        self,
        prompt_type: PromptType,
        model: Optional[str] = None,
    ) -> List[PromptVersion]:
        """
        List all versions of a prompt type.

        Args:
            prompt_type: Type of prompt
            model: Filter by model

        Returns:
            List of PromptVersions sorted by version
        """
        prompts = self.prompts.get(prompt_type.value, [])

        if model:
            prompts = [p for p in prompts if p.model == model]

        return sorted(prompts, key=lambda p: p.version)

    def save(self) -> None:
        """Save library to JSON file."""
        data = {
            prompt_type: [p.to_dict() for p in versions]
            for prompt_type, versions in self.prompts.items()
        }

        with self.library_path.open("w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load library from JSON file."""
        with self.library_path.open("r") as f:
            data = json.load(f)

        self.prompts = {
            prompt_type: [PromptVersion.from_dict(p) for p in versions]
            for prompt_type, versions in data.items()
        }

    def format_prompt(
        self,
        prompt_type: PromptType,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt with provided variables.

        Args:
            prompt_type: Type of prompt
            version: Specific version (None for latest)
            **kwargs: Variables to inject into prompt template

        Returns:
            Formatted prompt text
        """
        prompt = self.get_prompt(prompt_type, version)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_type}, version={version}")

        return prompt.prompt_text.format(**kwargs)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_expected_titles_section(titles: List[str]) -> str:
    """
    Create the expected titles section for chapter extraction.

    Args:
        titles: List of expected recipe titles for this chapter

    Returns:
        Formatted expected titles section
    """
    if not titles:
        return ""

    title_list = "\n".join(f"- {title}" for title in titles)
    return EXPECTED_TITLES_TEMPLATE.format(title_list=title_list)


def format_recipe_list_prompt(input_sections: str) -> str:
    """
    Format the recipe list discovery prompt with input sections.

    Args:
        input_sections: Combined markdown link sections from chapters

    Returns:
        Formatted prompt
    """
    library = PromptLibrary()
    return library.format_prompt(
        PromptType.RECIPE_LIST_DISCOVERY,
        version="1.1.0",  # Use latest version
        input_sections=input_sections,
    )


def format_chapter_extraction_prompt(
    chapter_markdown: str,
    expected_titles: Optional[List[str]] = None,
) -> str:
    """
    Format the chapter extraction prompt.

    Args:
        chapter_markdown: Markdown content of the chapter
        expected_titles: Optional list of expected recipe titles

    Returns:
        Formatted prompt
    """
    library = PromptLibrary()

    expected_section = ""
    if expected_titles:
        expected_section = create_expected_titles_section(expected_titles)

    return library.format_prompt(
        PromptType.CHAPTER_EXTRACTION,
        version="1.1.0",  # Use latest version
        chapter_markdown=chapter_markdown,
        expected_titles_section=expected_section,
    )


# =============================================================================
# MAIN - Demo Usage
# =============================================================================


if __name__ == "__main__":
    # Initialize library
    library = PromptLibrary()

    # Example 1: Get latest prompt
    latest_extraction = library.get_prompt(
        PromptType.CHAPTER_EXTRACTION,
        model="gpt-5-nano",
    )
    print(f"Latest extraction prompt: v{latest_extraction.version}")

    # Example 2: Format a prompt with variables
    formatted = library.format_prompt(
        PromptType.CHAPTER_EXTRACTION,
        chapter_markdown="# Chocolate Cake\n\nIngredients:\n- 200g flour...",
        expected_titles_section="",
    )
    print(f"\nFormatted prompt length: {len(formatted)} chars")

    # Example 3: Update performance metrics
    metrics = PerformanceMetrics(
        total_recipes=125,
        expected_recipes=125,
        correctly_extracted=120,
        missing_recipes=5,
        extra_recipes=0,
        title_modifications=0,
    )
    metrics.calculate()

    library.update_performance(
        PromptType.CHAPTER_EXTRACTION,
        version="1.1.0",
        metrics=metrics,
    )

    print(f"\nPerformance: {metrics.accuracy_percent:.1f}% accuracy")

    # Example 4: Get best performing prompt
    best = library.get_best_prompt(
        PromptType.CHAPTER_EXTRACTION,
        model="gpt-5-nano",
    )
    if best and best.performance:
        print(f"\nBest prompt: v{best.version} ({best.performance.accuracy_percent:.1f}%)")

    # Example 5: List all versions
    versions = library.list_versions(PromptType.RECIPE_LIST_DISCOVERY)
    print(f"\nRecipe list discovery versions: {[v.version for v in versions]}")

    # Save library
    library.save()
    print(f"\nLibrary saved to: {library.library_path}")
