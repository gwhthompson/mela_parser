# Critical Bug Fix: TOC Chapter Extraction Issue

## Problem Statement
All 125 recipes were extracted from Chapter 2 (the Table of Contents / Recipe List) but marked as incomplete and skipped during save, resulting in 0 recipes written to disk.

### Symptoms
- Extraction log: "Chapter 2 (pages/listofrecipes.html): extracted 125 recipes"
- Writing phase: "Skipping save; incomplete recipe: [all 125 recipe names]"
- Result: 0 recipes in output

## Root Cause Analysis

### Issue 1: Incorrect Chapter Targeting
The extraction pipeline was using a matching algorithm that compared recipe titles against chapter content:
```python
likely_here = [t for t in expected_titles if t.lower() in chapter_md.lower()]
```

This created a critical problem:
- **Phase 2 (Discovery)** correctly identified 125 recipes from the TOC chapter by extracting recipe titles from links
- **Phase 3 (Extraction)** tried to extract these 125 recipes from the TOC chapter itself
- Chapter 2 contains all recipe names as links (just titles), not full recipes with ingredients/instructions
- The LLM extracted title-only "recipes" from the TOC

### Issue 2: Incomplete Recipe Validation
The `recipe.write_recipe()` method correctly rejected these as incomplete:
```python
if (not recipe_dict.get("title", "").strip()
    or not recipe_dict.get("ingredients", "").strip()
    or not recipe_dict.get("instructions", "").strip()):
    print(f"Skipping save; incomplete recipe: {recipe_dict.get('title', 'UNKNOWN')}")
    return ""
```

TOC-extracted recipes had:
- ✓ Title (from TOC)
- ✗ Ingredients (missing)
- ✗ Instructions (missing)

## Solution Implemented

### Fix 1: Table of Contents Detection (New)
Added `_is_table_of_contents()` method to detect and skip TOC chapters before extraction:

**Detection criteria:**
1. Filename patterns: 'listofrecipes', 'contents', 'toc', 'index', 'table of contents'
2. Link density: > 50% of content lines are links AND >20 links total
3. Lack of recipe content: No ingredients/instructions patterns found, but many links exist

**Result:** TOC chapters are now skipped entirely during extraction phase

### Fix 2: Complete Recipe Filtering (New)
Added `_is_complete_recipe()` method to validate recipes before deduplication:

**Validation checks:**
- Title: non-empty string
- Ingredients: non-empty string
- Instructions: non-empty string

**Result:** Any incomplete recipes that slip through extraction are filtered out before deduplication

### Fix 3: Extraction Pipeline Refactoring
Modified the extraction pipeline in `extract_recipes()`:

**Before:**
1. Phase 1: Convert chapters
2. Phase 2: Discover recipe list
3. Phase 3: Extract from ALL chapters
4. Phase 4: Deduplication

**After:**
1. Phase 1: Convert chapters
2. Phase 2: Discover recipe list
3. Phase 3a: Filter out TOC chapters
4. Phase 3b: Extract from content chapters only
5. Phase 4: Filter incomplete recipes
6. Phase 5: Deduplication

## Code Changes

### File: `/Volumes/george/Developer/mela_parser/main_chapters_v2.py`

#### Added Method 1: TOC Detection
```python
def _is_table_of_contents(self, chapter_name: str, chapter_md: str) -> bool:
    """Detect if a chapter is a Table of Contents / Recipe List."""
    # Checks filename patterns, link density, and recipe content indicators
    # Returns True if TOC-like chapter detected
```

#### Added Method 2: Complete Recipe Validation
```python
def _is_complete_recipe(self, recipe: MelaRecipe) -> bool:
    """Check if recipe has minimum required fields (title, ingredients, instructions)."""
    # Validates recipe completeness
    # Returns True only if all required fields are non-empty
```

#### Modified Method: Extract Recipes Pipeline
```python
async def extract_recipes(...):
    # Added Phase 3a: TOC chapter filtering
    content_chapters = [ch for ch in chapters if not self._is_table_of_contents(...)]

    # Phase 3b: Extract only from content chapters
    # Phase 4: Filter incomplete recipes
    complete_recipes = [r for r in all_recipes if self._is_complete_recipe(r)]

    # Phase 5: Deduplication on complete recipes only
```

## Impact

### What Was Fixed
- TOC/recipe list chapters are now skipped during extraction (not processed at all)
- Any incomplete recipes that are extracted are filtered out before deduplication
- The validation properly rejects recipes without full content

### Expected Results
- Only actual recipe chapters with ingredients and instructions will be extracted
- All extracted recipes will have complete data
- No incomplete recipes will be written to disk
- Recipe count will match actual complete recipes found in the book

### Testing
To verify the fix works:
1. Run extraction on the Jerusalem cookbook
2. Verify Chapter 2 (listofrecipes.html) is skipped with message: "Skipping TOC chapter: pages/listofrecipes.html"
3. Verify Phase 4 shows filtering of incomplete recipes (if any slip through)
4. Verify final recipe count matches expected complete recipes
5. Verify all written recipes have ingredients and instructions

## Prevention
This issue is now prevented by:
1. **Proactive TOC detection** - identifies and skips TOC chapters before attempting extraction
2. **Multi-layer validation** - incomplete recipes are caught both in pipeline and write logic
3. **Better logging** - logs now clearly indicate when TOC chapters are skipped
4. **Safer extraction** - only attempts to extract from chapters with actual recipe content
