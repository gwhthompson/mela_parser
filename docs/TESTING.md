# Testing Guide

## Overview

Test suite for recipe extraction from EPUB cookbooks.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_simple_chapters.py -v
```

## Test Structure

Tests automatically discover EPUB files in `examples/input/` and run against them.

If no EPUBs are present, tests are skipped with a helpful message.

### Test Classes

**TestExtraction** - Basic extraction functionality:
- `test_extraction_completes` - Extraction exits cleanly
- `test_extracts_recipes` - At least one recipe found
- `test_writes_recipes` - Recipe files written
- `test_output_files_created` - Output directory populated

**TestRecipeQuality** - Recipe filtering:
- `test_no_component_recipes` - No standalone sauce/marinade recipes

### Parametrized Tests

`test_each_epub_extracts` runs against every EPUB in `examples/input/`.

## Adding Test EPUBs

1. Place your EPUB files in `examples/input/`
2. Run tests: `uv run pytest -v`

Tests will run against all available EPUBs.

## CI Configuration

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: uv run pytest -v
```

Note: CI tests will skip if no EPUBs are available.

## Related Files

- **Tests**: `tests/test_simple_chapters.py`
- **Test EPUBs**: `examples/input/*.epub` (not included, add your own)
