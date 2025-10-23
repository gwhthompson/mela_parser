# Testing Guide for Chapter-Based Recipe Extraction

## Overview

Comprehensive test suite for the chapter-based recipe extraction system in `main_chapters_v2.py`.

## Test File

**Location**: `test_recipe_lists.py`

**Ground Truth**: Recipe lists in `examples/output/recipe-lists/`
- `jerusalem-recipe-list.txt` (125 recipes)
- `a-modern-way-to-eat-recipe-list.txt` (142 recipes)
- `completely-perfect-recipe-list.txt` (122 recipes)
- `simple-recipe-list.txt` (140 recipes)

## Prerequisites

Install test dependencies:

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
pip install pytest pytest-asyncio
```

## Running Tests

### Run All Tests

```bash
pytest test_recipe_lists.py -v
```

### Run Specific Test Classes

```bash
# Ground truth validation (fast, no API calls)
pytest test_recipe_lists.py::TestAllCookbooks -v

# Prompt library tests (fast, no API calls)
pytest test_recipe_lists.py::TestPromptLibrary -v

# Chapter extraction tests (requires API access)
pytest test_recipe_lists.py::TestChapterExtraction -v

# Recipe list discovery tests (requires API access)
pytest test_recipe_lists.py::TestRecipeListDiscovery -v

# Iterative refinement tests (mocked, fast)
pytest test_recipe_lists.py::TestIterativeRefinement -v

# End-to-end integration tests (requires API access)
pytest test_recipe_lists.py::TestEndToEndExtraction -v
```

### Run Individual Tests

```bash
# Test exact recipe counts
pytest test_recipe_lists.py::TestChapterExtraction::test_jerusalem_chapter_count -v
pytest test_recipe_lists.py::TestChapterExtraction::test_modern_way_chapter_count -v
pytest test_recipe_lists.py::TestChapterExtraction::test_completely_perfect_chapter_count -v

# Test deduplication
pytest test_recipe_lists.py::TestChapterExtraction::test_no_duplicate_recipes -v

# Test title matching
pytest test_recipe_lists.py::TestChapterExtraction::test_exact_title_matching -v

# Test discovery
pytest test_recipe_lists.py::TestRecipeListDiscovery::test_discovers_correct_count -v
pytest test_recipe_lists.py::TestRecipeListDiscovery::test_list_cleaning -v

# Test end-to-end
pytest test_recipe_lists.py::TestEndToEndExtraction::test_end_to_end_extraction -v
```

### Run Tests with Output

```bash
# Show print statements
pytest test_recipe_lists.py -v -s

# Show detailed output
pytest test_recipe_lists.py -v -s --tb=short
```

### Run Fast Tests Only (No API Calls)

```bash
# Ground truth validation + prompt library tests
pytest test_recipe_lists.py::TestAllCookbooks test_recipe_lists.py::TestPromptLibrary -v
```

## Test Coverage

### 1. TestChapterExtraction

Tests chapter-based extraction against ground truth recipe lists.

- **test_jerusalem_chapter_count**: Validates exactly 125 recipes extracted
- **test_modern_way_chapter_count**: Validates exactly 142 recipes extracted
- **test_completely_perfect_chapter_count**: Validates exactly 122 recipes extracted
- **test_no_duplicate_recipes**: Ensures chapter boundaries prevent duplicates (<20% duplicate rate)
- **test_exact_title_matching**: Validates ≥90% exact title matches

### 2. TestRecipeListDiscovery

Tests recipe list discovery from EPUB structure.

- **test_discovers_correct_count**: Recipe list discovery finds expected number (within 10% tolerance)
- **test_list_cleaning**: gpt-5-mini removes section headers properly
- **test_handles_no_list_found**: Graceful handling when no TOC/Index exists

### 3. TestIterativeRefinement

Tests iterative prompt refinement process.

- **test_iteration_improves_accuracy**: Each iteration increases match rate
- **test_max_iterations_respected**: Pipeline stops at max iterations
- **test_saves_iteration_history**: Iteration history saved to JSON

### 4. TestEndToEndExtraction

Integration tests for complete extraction pipeline.

- **test_end_to_end_extraction**: Full pipeline on simple.epub validates all phases
- **test_validation_report_structure**: ValidationReport serialization and structure

### 5. TestPromptLibrary

Tests for prompt library and versioning.

- **test_default_prompts_have_placeholders**: Required placeholders present
- **test_prompt_serialization**: JSON serialization/deserialization
- **test_prompt_version_increments**: Version tracking across iterations

### 6. TestAllCookbooks

Parametrized tests across all cookbooks.

- **test_ground_truth_files_valid**: Validates ground truth files exist and have correct counts

## Test Results Interpretation

### Passing Tests

All tests should pass if:
- Chapter-based extraction achieves exact recipe counts
- Title matching is ≥90% accurate
- Duplicate rate is <20%
- Recipe list discovery works correctly
- Iteration system improves accuracy

### Common Failures

**Count Mismatch**
```
AssertionError: Expected exactly 125 unique recipes, got 132. Difference: 7
```
- **Cause**: Over-extraction or duplicate recipes not removed
- **Fix**: Check deduplication logic, verify chapter boundaries

**Low Match Rate**
```
AssertionError: Title match rate too low: 85.2%. Expected at least 90%
```
- **Cause**: Title extraction not exact, fuzzy matching issues
- **Fix**: Review extraction prompts, check for title variations

**High Duplicate Rate**
```
AssertionError: Duplicate rate too high: 25.3%. Chapter-based extraction should have <20%
```
- **Cause**: Recipes spanning multiple chapters or incomplete chapter boundaries
- **Fix**: Verify chapter-based extraction prevents overlaps

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest test_recipe_lists.py::TestAllCookbooks -v
    pytest test_recipe_lists.py::TestPromptLibrary -v
```

## Manual Testing

For manual validation:

```bash
# Run extraction on simple.epub
python main_chapters_v2.py examples/input/simple.epub --model gpt-5-nano

# Compare output count with ground truth
wc -l examples/output/recipe-lists/simple-recipe-list.txt
# Should be 140 recipes

# Check extraction output
ls -la output/simple-chapters-v2/
```

## Debugging Failed Tests

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Test Outputs

```bash
# Run single test with full output
pytest test_recipe_lists.py::TestEndToEndExtraction::test_end_to_end_extraction -v -s --tb=long
```

### Check Ground Truth Files

```bash
# Verify ground truth integrity
pytest test_recipe_lists.py::TestAllCookbooks::test_ground_truth_files_valid -v
```

## Performance Notes

- **Fast tests** (ground truth validation, prompt library): ~1-2 seconds
- **API-based tests** (chapter extraction, discovery): ~30-60 seconds per cookbook
- **End-to-end tests**: ~2-5 minutes per cookbook

To minimize API costs during development, run fast tests first:

```bash
pytest test_recipe_lists.py::TestAllCookbooks test_recipe_lists.py::TestPromptLibrary -v
```

## Future Enhancements

1. **Fuzzy title matching**: Add tests for normalized title comparison
2. **Performance benchmarks**: Track extraction time across iterations
3. **Ingredient validation**: Verify ingredient extraction accuracy
4. **Instruction validation**: Check instruction completeness
5. **Image association**: Test recipe-image linking

## Related Files

- **Implementation**: `main_chapters_v2.py`
- **Ground Truth**: `examples/output/recipe-lists/*.txt`
- **Test EPUBs**: `examples/input/*.epub`
- **Configuration**: `pyproject.toml` (pytest settings)
