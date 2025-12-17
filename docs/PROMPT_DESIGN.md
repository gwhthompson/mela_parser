# Prompt Design Documentation

## Overview

This document explains the design choices, techniques, and rationale behind the prompt library for chapter-based recipe extraction. The goal is **100% accuracy** in matching expected recipe lists.

## Design Philosophy

### Constitutional AI Principles

The prompts are built on a core constitutional principle:

> **"Extract exactly what exists. Never infer, never label, never enhance."**

This principle is explicitly stated in the prompts to guide the model's behavior and prevent common failure modes like:
- Adding labels: "(partial)", "(duplicate)", "(continued)"
- Modifying titles: changing "yoghurt" to "yogurt", "&" to "and"
- Inferring metadata: guessing cooking times, adding categories

### Key Techniques Applied

1. **XML Tag Structure** - Clear boundaries for different prompt sections
2. **Explicit Negative Constraints** - Tell the model what NOT to do
3. **Few-Shot Examples** - Show correct and incorrect behaviors
4. **Constitutional Self-Correction** - Embed principles that guide decision-making
5. **Quality Checks** - Pre-submission verification steps
6. **Fail-Safe Defaults** - "When uncertain, skip rather than guess"

---

## Prompt 1: Recipe List Discovery

**Model:** `gpt-5-mini`
**Temperature:** `0.0`
**Purpose:** Extract clean, unique recipe titles from table of contents/index sections

### The Prompt (v1.1.0)

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

### Design Rationale

#### 1. XML Tag Structure
- `<objective>`, `<input>`, `<extraction_rules>`, `<examples>`, `<output_format>`, `<quality_check>`
- **Why:** Clear visual boundaries help the model distinguish between instructions, examples, and data
- **Impact:** Reduces confusion between "what to do" and "what to process"

#### 2. Four-Tier Rule System
1. **INCLUDE rules** - Positive guidance (what to keep)
2. **EXCLUDE rules** - Negative guidance (what to remove)
3. **FORBIDDEN MODIFICATIONS** - Explicit don'ts for title accuracy
4. **CONSTITUTIONAL PRINCIPLE** - Core philosophy statement

**Why:** Layered rules catch different types of errors. The constitutional principle acts as a tiebreaker when rules are ambiguous.

#### 3. Dual Examples (Correct + Incorrect)
Each example shows:
- Input text
- Correct output
- Incorrect output with explanations of what's wrong

**Why:** Showing what NOT to do is as important as showing what to do. Models learn from contrast.

#### 4. Quality Checklist
Pre-submission verification steps as checkboxes.

**Why:** Encourages self-reflection before returning results. Acts as a final safety net.

#### 5. Explicit Punctuation Preservation
Dedicates an entire example to punctuation: apostrophes, ampersands, parentheses, commas.

**Why:** Common failure mode is "standardizing" punctuation. This example forces attention to exact preservation.

---

## Prompt 2: Chapter Extraction

**Model:** `gpt-5-nano`
**Temperature:** `0.0`
**Purpose:** Extract complete recipes from a single chapter with exact titles

### The Prompt (v1.1.0)

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

### Design Rationale

#### 1. Precise "Complete Recipe" Definition
Three mandatory components with concrete examples and minimum requirements:
- Title (not a header, not a continuation)
- At least 2 ingredients with measurements
- At least 2 instruction steps

**Why:** Ambiguity in "complete" leads to extraction of partial recipes. Specific minimums create clear boundaries.

#### 2. Five-Category Exclusion Rules
Categorizes all the things NOT to extract:
1. Incomplete recipes
2. Cross-references
3. Recipe continuations
4. Non-recipes
5. Variations/mentions

**Why:** Models need explicit negative examples. Categorization makes it easy to match edge cases to rules.

#### 3. Constitutional Title Principle
> "The title in the source text is sacred. Extract it exactly, or don't extract at all."

**Why:** Title accuracy is the #1 failure mode. This principle creates a bright line: exact or nothing.

#### 4. Four Detailed Examples
- Example 1: Complete recipe (correct extraction)
- Example 2: Cross-reference (skip, don't extract)
- Example 3: Continuation (skip, don't extract)
- Example 4: Exact title preservation with annotations

**Why:** Covers the most common failure modes with concrete "do this, not that" guidance.

#### 5. Quality Checks as Checklist
Seven verification steps before finalizing extraction.

**Why:** Creates a mental pause before submission. Acts as final accuracy gate.

#### 6. Failure Modes Section
Explicitly lists common mistakes and reminds: "When in doubt, SKIP the recipe."

**Why:** Fail-safe default. Better to miss a recipe (false negative) than extract it wrong (false positive).

---

## Prompt 3: Prompt Improvement Meta-Prompt

**Model:** `gpt-5-mini`
**Temperature:** `0.3` (slightly higher for creative suggestions)
**Purpose:** Analyze validation failures and suggest prompt improvements

### Design Rationale

This meta-prompt uses chain-of-thought reasoning to:

1. **Analyze failure patterns** - Categorize errors into missing, extra, and modified
2. **Identify root causes** - Why did the errors occur?
3. **Suggest specific improvements** - Not vague advice, but exact text changes
4. **Provide new examples** - Based on observed failure modes
5. **Estimate impact** - Expected improvement in accuracy

**Output Format:** Structured JSON for easy parsing and application.

**Use Case:** Automatic prompt iteration based on validation results. Feed in diffs, get back concrete improvements.

---

## PromptLibrary Class

### Features

1. **Version Management**
   - Semantic versioning (1.0.0, 1.1.0, 2.0.0)
   - Track multiple versions per prompt type
   - Get latest or specific version

2. **Performance Tracking**
   - `PerformanceMetrics` dataclass with accuracy, precision, recall
   - Update metrics after validation runs
   - Compare prompt versions

3. **Best Prompt Selection**
   - `get_best_prompt()` - Returns highest performing version by metric
   - Supports filtering by model
   - Enables automatic rollback if new version regresses

4. **A/B Testing Support**
   - Load multiple versions
   - Run experiments with different prompts
   - Compare results statistically

5. **Persistence**
   - Save/load from JSON (`prompt_versions.json`)
   - Track timestamp, description, notes
   - Full version history

### Usage Examples

```python
from prompt_library import PromptLibrary, PromptType, PerformanceMetrics

# Initialize library
library = PromptLibrary()

# Get latest chapter extraction prompt
prompt = library.get_prompt(
    PromptType.CHAPTER_EXTRACTION,
    model="gpt-5-nano"
)

# Format with variables
formatted = library.format_prompt(
    PromptType.CHAPTER_EXTRACTION,
    chapter_markdown=chapter_content,
    expected_titles_section=create_expected_titles_section(titles)
)

# Update performance after validation
metrics = PerformanceMetrics(
    total_recipes=125,
    expected_recipes=125,
    correctly_extracted=123,
    missing_recipes=2,
    extra_recipes=0,
    title_modifications=0
)
metrics.calculate()  # Computes accuracy, precision, recall

library.update_performance(
    PromptType.CHAPTER_EXTRACTION,
    version="1.1.0",
    metrics=metrics
)

# Get best performing prompt
best = library.get_best_prompt(
    PromptType.CHAPTER_EXTRACTION,
    model="gpt-5-nano",
    metric="accuracy_percent"
)
print(f"Best: v{best.version} ({best.performance.accuracy_percent:.1f}%)")

# List all versions
versions = library.list_versions(PromptType.RECIPE_LIST_DISCOVERY)
for v in versions:
    print(f"v{v.version}: {v.description}")

# Save changes
library.save()
```

---

## Advanced Techniques Summary

### 1. Chain-of-Thought Prompting
Used implicitly through quality checks and verification steps. The model "thinks through" the checks before finalizing.

### 2. Few-Shot Learning
Multiple examples per prompt showing correct and incorrect behaviors. Contrasting examples teach faster than positive examples alone.

### 3. Constitutional AI
Core principles embedded in prompts:
- "Extract exactly what exists"
- "The title is sacred"
- "When in doubt, skip"

These act as tiebreakers and guide decision-making in ambiguous cases.

### 4. Structured Input/Output
- XML tags for clear section boundaries
- JSON schema for structured extraction
- Checklists for verification

### 5. Explicit Negative Constraints
Every prompt has extensive "DO NOT" sections:
- Never add labels
- Never modify titles
- Never guess metadata

Telling the model what NOT to do is as important as what to do.

### 6. Fail-Safe Defaults
"When uncertain, skip rather than guess."

Prefer false negatives (missing a recipe) over false positives (wrong extraction).

### 7. Temperature Optimization
- `0.0` for extraction tasks (deterministic, consistent)
- `0.3` for meta-prompt (creative improvement suggestions)

### 8. Model-Specific Optimization
- `gpt-5-mini` for recipe list cleaning (higher intelligence for deduplication)
- `gpt-5-nano` for chapter extraction (cost-effective for bulk processing)

---

## Validation Strategy

### Test Against Ground Truth

1. **Load ground truth lists** (create your own for your test EPUBs)

2. **Run extraction with prompts**

3. **Calculate metrics:**
   - **Accuracy:** Correctly extracted / Expected
   - **Precision:** Correctly extracted / Total extracted
   - **Recall:** Correctly extracted / Expected
   - **Exact match:** Title matches character-by-character

4. **Analyze failures:**
   - Missing recipes → Too conservative? Incomplete in source?
   - Extra recipes → Extracting teasers/continuations?
   - Modified titles → Punctuation standardization? Added labels?

5. **Iterate:**
   - Use meta-prompt to suggest improvements
   - Create new version with enhanced rules/examples
   - Re-validate and compare metrics

### Success Criteria

- **100% accuracy:** All expected recipes extracted with exact titles
- **Zero false positives:** No extra recipes extracted
- **Zero title modifications:** Every title matches character-by-character

---

## Integration with Existing Code

### Update `main_chapters.py`

```python
from prompt_library import (
    PromptLibrary,
    PromptType,
    format_recipe_list_prompt,
    format_chapter_extraction_prompt,
    PerformanceMetrics,
)

# In discover_recipe_list():
def discover_recipe_list(chapters: List[Tuple[str, str]]) -> Optional[List[str]]:
    # ... collect link sections ...

    # Use prompt from library
    prompt = format_recipe_list_prompt(combined)

    # ... make API call ...

# In extract_from_chapter():
def extract_from_chapter(
    chapter_md: str,
    chapter_name: str,
    expected_titles: Optional[List[str]] = None,
    model: str = "gpt-5-nano"
) -> List[MelaRecipe]:
    # Use prompt from library
    prompt = format_chapter_extraction_prompt(chapter_md, expected_titles)

    # ... make API call ...

# After extraction, update performance:
library = PromptLibrary()
metrics = PerformanceMetrics(...)
library.update_performance(PromptType.CHAPTER_EXTRACTION, "1.1.0", metrics)
```

---

## Future Improvements

### A/B Testing Framework

Run multiple prompt versions in parallel and compare:

```python
def ab_test_prompts(chapters, versions=["1.0.0", "1.1.0"]):
    results = {}
    for version in versions:
        extracted = extract_with_version(chapters, version)
        metrics = validate_against_ground_truth(extracted)
        results[version] = metrics

    return results
```

### Automatic Prompt Evolution

1. Run extraction with current best prompt
2. Validate against ground truth
3. If accuracy < 100%, use meta-prompt to suggest improvements
4. Create new version with suggestions
5. Re-validate
6. If improved, promote new version to "best"
7. Repeat until 100% accuracy achieved

### Multi-Model Ensemble

Use multiple models and prompts, then merge results:

```python
def ensemble_extraction(chapter_md, expected_titles):
    results = []

    # Extract with gpt-5-nano v1.1.0
    results.append(extract(chapter_md, model="gpt-5-nano", version="1.1.0"))

    # Extract with gpt-5-mini v1.0.0
    results.append(extract(chapter_md, model="gpt-5-mini", version="1.0.0"))

    # Merge with voting or intersection
    final = merge_results(results, strategy="intersection")
    return final
```

---

## Conclusion

This prompt library implements production-grade prompt engineering for 100% accurate recipe extraction:

- **Constitutional AI** guides model behavior with core principles
- **Explicit constraints** prevent common failure modes (label additions, title modifications)
- **Comprehensive examples** teach through contrast (correct vs. incorrect)
- **Quality checks** create verification gates before submission
- **Fail-safe defaults** prefer false negatives over false positives
- **Version management** enables iteration, A/B testing, and rollback
- **Performance tracking** quantifies improvement and identifies best prompts

The system is designed to achieve and maintain 100% accuracy through systematic prompt optimization and continuous validation.
