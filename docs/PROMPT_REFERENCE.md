# Prompt Reference Guide

Quick reference for all prompts in the library. Copy-paste ready.

---

## 1. Recipe List Discovery Prompt (v1.1.0)

**Model:** gpt-5-mini | **Temperature:** 0.0

**Purpose:** Extract unique recipe titles from table of contents/index sections

**Input Variables:**
- `{input_sections}` - Combined markdown link sections from all chapters

**The Prompt:**

```
You are a precise recipe title extractor for cookbook digitization.

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
```

---

## 2. Chapter Extraction Prompt (v1.1.0)

**Model:** gpt-5-nano | **Temperature:** 0.0

**Purpose:** Extract complete recipes from a single chapter with exact titles

**Input Variables:**
- `{chapter_markdown}` - Markdown content of the chapter
- `{expected_titles_section}` - Optional section listing expected recipes (or empty string)

**The Prompt:**

```
You are a precise recipe extractor for cookbook digitization. Your goal is 100% accuracy.

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
```

---

## 3. Expected Titles Section Template

**Purpose:** Optional section to guide chapter extraction when recipe list is known

**Input Variables:**
- `{title_list}` - Formatted list of expected titles (e.g., "- Title 1\n- Title 2")

**Template:**

```
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
```

---

## 4. Prompt Improvement Meta-Prompt (v1.0.0)

**Model:** gpt-5-mini | **Temperature:** 0.3

**Purpose:** Analyze validation failures and suggest specific prompt improvements

**Input Variables:**
- `{current_prompt}` - The prompt being analyzed
- `{total_extracted}` - Number of recipes extracted
- `{expected_count}` - Number of recipes expected
- `{correct_count}` - Number correctly matched
- `{missing_list}` - List of missing recipe titles
- `{extra_list}` - List of extra recipe titles
- `{modified_list}` - List of title modifications

**Output Format:** Structured JSON with improvement suggestions

---

## Usage Examples

### Python Code

```python
from prompt_library import (
    format_recipe_list_prompt,
    format_chapter_extraction_prompt,
)

# Recipe list discovery
input_sections = """
[Chocolate Cake](#page12)
[Vanilla Ice Cream](#page34)
[Strawberry Tart](#page56)
"""

prompt1 = format_recipe_list_prompt(input_sections)

# Chapter extraction (no expected titles)
chapter_md = """
# Chocolate Cake

Ingredients:
- 200g flour
- 100g sugar
...
"""

prompt2 = format_chapter_extraction_prompt(chapter_md)

# Chapter extraction (with expected titles)
expected = ["Chocolate Cake", "Vanilla Ice Cream"]
prompt3 = format_chapter_extraction_prompt(chapter_md, expected)
```

### CLI

```bash
# Validate extracted results
python validate_prompts.py \
    --ground-truth examples/output/recipe-lists/jerusalem-recipe-list.txt \
    --extracted extracted_titles.txt \
    --prompt-version 1.1.0 \
    --verbose \
    --update-library
```

---

## Quick Reference: Constitutional Principles

These core principles guide all prompts:

1. **"Extract exactly what exists. Never infer, never label, never enhance."**
   - No added commentary or categories
   - No "helpful" labels like "(partial)" or "(continued)"

2. **"The title in the source text is sacred."**
   - Character-by-character exact match
   - Preserve ALL punctuation and capitalization

3. **"When in doubt, SKIP the recipe."**
   - Better false negative than false positive
   - Only extract when 100% certain it's complete and correct

4. **"Leave metadata blank rather than guess."**
   - No placeholders like "N/A" or "Unknown"
   - Extract only explicitly stated information

---

## Common Failure Modes & Prevention

| Failure Mode | Prevention Strategy |
|--------------|---------------------|
| Adding "(partial)" labels | Constitutional principle + negative examples |
| Changing & to "and" | Explicit punctuation preservation examples |
| Extracting recipe teasers | Definition of "complete" + cross-reference rules |
| Extracting continuations | Exclusion rules + continuation detection |
| Standardizing spelling | "yoghurt" vs "yogurt" example |
| Guessing cook times | Metadata handling rules |

---

## Version History

| Version | Type | Changes |
|---------|------|---------|
| 1.0.0 | Recipe List | Initial version with basic rules |
| 1.1.0 | Recipe List | Added constitutional AI, quality checks, more examples |
| 1.0.0 | Chapter Extract | Initial version with completeness rules |
| 1.1.0 | Chapter Extract | Enhanced examples, exclusion rules, quality checks |
| 1.0.0 | Meta-Prompt | Initial version for failure analysis |

---

**End of Prompt Reference**
