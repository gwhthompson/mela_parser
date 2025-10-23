# Integration Example: Using Prompt Library

This document shows how to integrate `prompt_library.py` into the existing `main_chapters.py` workflow.

## Quick Start

### 1. Update `discover_recipe_list()` function

**Before:**
```python
def discover_recipe_list(chapters: List[Tuple[str, str]]) -> Optional[List[str]]:
    client = OpenAI()

    # Collect link sections
    all_link_sections = []
    for chapter_name, chapter_md in chapters:
        link_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links = re.findall(link_pattern, chapter_md)
        if len(links) > 5:
            all_link_sections.append("\n".join(links))

    if not all_link_sections:
        return None

    combined = "\n\n".join(all_link_sections)

    # Hardcoded prompt
    prompt = f"""Extract the unique list of recipe titles from these potential recipe lists.

Remove:
- Section headers (Contents, Index, About, etc.)
- Page numbers
- Duplicates

Keep:
- Actual recipe titles EXACTLY as written
- One entry per unique recipe

<potential_lists>
{combined}
</potential_lists>"""

    # ... rest of function ...
```

**After:**
```python
from prompt_library import format_recipe_list_prompt

def discover_recipe_list(chapters: List[Tuple[str, str]]) -> Optional[List[str]]:
    client = OpenAI()

    # Collect link sections (unchanged)
    all_link_sections = []
    for chapter_name, chapter_md in chapters:
        link_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links = re.findall(link_pattern, chapter_md)
        if len(links) > 5:
            all_link_sections.append("\n".join(links))

    if not all_link_sections:
        return None

    combined = "\n\n".join(all_link_sections)

    # Use prompt from library (automatically uses latest version)
    prompt = format_recipe_list_prompt(combined)

    # ... rest of function unchanged ...
```

### 2. Update `extract_from_chapter()` function

**Before:**
```python
def extract_from_chapter(
    chapter_md: str,
    chapter_name: str,
    expected_titles: Optional[List[str]] = None,
    model: str = "gpt-5-nano"
) -> List[MelaRecipe]:
    client = OpenAI()

    # Build targeted prompt
    if expected_titles:
        likely_here = [t for t in expected_titles if t.lower() in chapter_md.lower()]
        if not likely_here:
            return []

        expected_list = "\n".join(f"- {title}" for title in likely_here)
        prompt = f"""Extract ONLY these specific recipes from this chapter.
Use the EXACT titles listed below.

Expected recipes:
{expected_list}

<chapter>
{chapter_md}
</chapter>"""
    else:
        prompt = f"""Extract ALL complete recipes from this chapter.
Copy titles EXACTLY as they appear in the text.
Do NOT add commentary or labels.

<chapter>
{chapter_md}
</chapter>"""

    # ... rest of function ...
```

**After:**
```python
from prompt_library import format_chapter_extraction_prompt

def extract_from_chapter(
    chapter_md: str,
    chapter_name: str,
    expected_titles: Optional[List[str]] = None,
    model: str = "gpt-5-nano"
) -> List[MelaRecipe]:
    client = OpenAI()

    # Find likely recipes in this chapter (if expected_titles provided)
    if expected_titles:
        likely_here = [t for t in expected_titles if t.lower() in chapter_md.lower()]
        if not likely_here:
            return []
        chapter_expected = likely_here
    else:
        chapter_expected = None

    # Use prompt from library (automatically uses latest version)
    prompt = format_chapter_extraction_prompt(chapter_md, chapter_expected)

    # ... rest of function unchanged ...
```

### 3. Add Performance Tracking

Add this at the end of `main()` to track prompt performance:

```python
from prompt_library import (
    PromptLibrary,
    PromptType,
    PerformanceMetrics,
)

def main():
    # ... existing code ...

    # After extraction is complete, calculate performance
    if expected_titles:
        # Compare extracted vs expected
        extracted_titles = set(r.title for r in unique_recipes)
        expected_set = set(expected_titles)

        correct = extracted_titles & expected_set
        missing = expected_set - extracted_titles
        extra = extracted_titles - expected_set

        # Create metrics
        metrics = PerformanceMetrics(
            total_recipes=len(extracted_titles),
            expected_recipes=len(expected_titles),
            correctly_extracted=len(correct),
            missing_recipes=len(missing),
            extra_recipes=len(extra),
            title_modifications=0,  # Would need fuzzy matching to detect
        )
        metrics.calculate()

        # Log performance
        logging.info("\nPrompt Performance Metrics:")
        logging.info(f"  Accuracy:  {metrics.accuracy_percent:.1f}%")
        logging.info(f"  Precision: {metrics.precision_percent:.1f}%")
        logging.info(f"  Recall:    {metrics.recall_percent:.1f}%")

        # Update library
        library = PromptLibrary()
        library.update_performance(
            PromptType.CHAPTER_EXTRACTION,
            version="1.1.0",  # Current version being used
            metrics=metrics,
        )
        logging.info(f"Updated prompt library with metrics")

        # If accuracy < 100%, log improvement suggestions
        if metrics.accuracy_percent < 100.0 or metrics.extra_recipes > 0:
            logging.warning("\nValidation failed. Run validation script for improvement suggestions:")
            logging.warning(f"  python validate_prompts.py --ground-truth path/to/expected.txt --extracted extracted_titles.txt")
```

## Advanced: Version Comparison

Test multiple prompt versions and pick the best:

```python
from prompt_library import PromptLibrary, PromptType

def extract_with_version_comparison(
    chapter_md: str,
    expected_titles: Optional[List[str]] = None,
) -> List[MelaRecipe]:
    """Extract recipes using multiple prompt versions, return best result."""
    library = PromptLibrary()
    client = OpenAI()

    # Get all available versions
    versions = library.list_versions(PromptType.CHAPTER_EXTRACTION, model="gpt-5-nano")

    results = {}

    for version in versions:
        # Get prompt for this version
        prompt_version = library.get_prompt(
            PromptType.CHAPTER_EXTRACTION,
            version=version.version,
        )

        # Format prompt
        prompt_text = library.format_prompt(
            PromptType.CHAPTER_EXTRACTION,
            version=version.version,
            chapter_markdown=chapter_md,
            expected_titles_section=create_expected_titles_section(expected_titles or []),
        )

        # Extract with this version
        response = client.responses.parse(
            model="gpt-5-nano",
            input=[EasyInputMessageParam(role="user", content=prompt_text)],
            text_format=CookbookRecipes,
        )

        results[version.version] = response.output_parsed.recipes

    # If we have expected titles, pick version with best accuracy
    if expected_titles:
        expected_set = set(expected_titles)
        best_version = None
        best_score = 0

        for version, recipes in results.items():
            extracted = set(r.title for r in recipes)
            correct = len(extracted & expected_set)
            score = correct / len(expected_set)

            if score > best_score:
                best_score = score
                best_version = version

        logging.info(f"Best version: {best_version} ({best_score*100:.1f}% accuracy)")
        return results[best_version]
    else:
        # No ground truth, return latest version
        latest = max(versions, key=lambda v: v.version)
        return results[latest.version]
```

## Validation Workflow

### Step 1: Extract recipes

```bash
python main_chapters.py examples/jerusalem.epub --output-dir output/jerusalem
```

### Step 2: Save extracted titles

```python
# Add to main_chapters.py after extraction:
with open("extracted_titles.txt", "w") as f:
    for recipe in unique_recipes:
        f.write(f"{recipe.title}\n")
```

### Step 3: Validate against ground truth

```bash
python validate_prompts.py \
    --ground-truth examples/output/recipe-lists/jerusalem-recipe-list.txt \
    --extracted extracted_titles.txt \
    --prompt-version 1.1.0 \
    --verbose \
    --update-library
```

### Step 4: Review validation report

```
================================================================================
VALIDATION REPORT
================================================================================

Overall Metrics:
  Expected recipes:      125
  Extracted recipes:     125
  Correctly extracted:   125
  Missing:               0
  Extra:                 0
  Title modifications:   0

Accuracy Metrics:
  Accuracy:   100.00%
  Precision:  100.00%
  Recall:     100.00%

                   🎉 SUCCESS! 100% ACCURACY ACHIEVED!

================================================================================
```

### Step 5: If not 100%, iterate

If validation fails:

1. Review the detailed breakdown of missing/extra recipes
2. Read the improvement suggestions
3. Manually check sample failures
4. Update prompt based on observed patterns
5. Create new version in `prompt_library.py`
6. Re-run extraction with new version
7. Validate again

## Example: Creating a New Prompt Version

Based on validation failures:

```python
from prompt_library import PromptLibrary, PromptVersion, PromptType

# Create improved prompt based on failures
CHAPTER_EXTRACTION_PROMPT_V1_2 = """
You are a precise recipe extractor for cookbook digitization. Your goal is 100% accuracy.

<objective>
Extract complete recipes from this cookbook chapter.
Use EXACT titles. Never add labels or modify text.
Skip anything incomplete or ambiguous.
</objective>

<!-- ... rest of prompt with improvements ... -->

<!-- NEW: Based on validation, add example for common failure mode -->
<examples>
<!-- ... existing examples ... -->

EXAMPLE 5 - Recipe teaser (DO NOT EXTRACT):
<source>
The Perfect Roast Chicken

This is my go-to recipe for Sunday dinner. The secret is in the temperature
and timing - check out the full recipe on page 123 for all the details.
</source>

WRONG: Extracting this as a recipe

Why WRONG:
- No ingredient list provided
- No cooking instructions
- Explicitly references "full recipe on page 123"
- This is a teaser/preview, not a complete recipe

CORRECT: Skip this entirely, don't extract.
</examples>
"""

# Add to library
library = PromptLibrary()
library.add_prompt(PromptVersion(
    version="1.2.0",
    prompt_type=PromptType.CHAPTER_EXTRACTION,
    prompt_text=CHAPTER_EXTRACTION_PROMPT_V1_2,
    model="gpt-5-nano",
    description="Added example for recipe teasers based on validation failures",
    temperature=0.0,
    notes="Addresses issue where recipe previews were being extracted as complete recipes"
))

library.save()
```

## Automated Improvement Loop

For automatic prompt iteration:

```python
from prompt_library import PromptLibrary, PromptType
from openai import OpenAI

def auto_improve_prompt(
    current_version: str,
    validation_results: Dict[str, Any],
) -> str:
    """Use meta-prompt to suggest improvements, create new version."""
    library = PromptLibrary()
    client = OpenAI()

    # Get improvement prompt
    improvement_prompt_version = library.get_prompt(PromptType.PROMPT_IMPROVEMENT)

    # Format with validation results
    current_prompt = library.get_prompt(
        PromptType.CHAPTER_EXTRACTION,
        version=current_version
    ).prompt_text

    improvement_prompt = improvement_prompt_version.prompt_text.format(
        current_prompt=current_prompt,
        total_extracted=validation_results["total_extracted"],
        expected_count=validation_results["expected_count"],
        correct_count=validation_results["correct_count"],
        missing_list="\n".join(validation_results["missing"]),
        extra_list="\n".join(validation_results["extra"]),
        modified_list="\n".join(validation_results["modifications"]),
    )

    # Get improvement suggestions
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": improvement_prompt}],
        temperature=0.3,
    )

    suggestions = response.choices[0].message.content

    # Parse suggestions and create new prompt version
    # (Implementation would parse JSON and apply changes)

    print("Improvement suggestions:")
    print(suggestions)

    return suggestions
```

## Summary

To integrate prompt library:

1. **Import helper functions:**
   ```python
   from prompt_library import (
       format_recipe_list_prompt,
       format_chapter_extraction_prompt,
       PromptLibrary,
       PromptType,
       PerformanceMetrics,
   )
   ```

2. **Replace hardcoded prompts with library calls:**
   ```python
   prompt = format_recipe_list_prompt(input_sections)
   prompt = format_chapter_extraction_prompt(chapter_md, expected_titles)
   ```

3. **Track performance after extraction:**
   ```python
   library.update_performance(PromptType.CHAPTER_EXTRACTION, "1.1.0", metrics)
   ```

4. **Validate and iterate:**
   ```bash
   python validate_prompts.py --ground-truth ... --extracted ... --update-library
   ```

5. **Create new versions based on failures:**
   ```python
   library.add_prompt(PromptVersion(...))
   library.save()
   ```

This enables systematic prompt optimization toward 100% accuracy!
