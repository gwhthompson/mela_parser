# Examples

This directory contains example EPUB cookbook files and their extracted recipe outputs for testing and demonstration purposes.

## Directory Structure

```
examples/
├── input/          # Sample EPUB cookbook files
└── output/         # Extracted recipes and recipe lists
    └── recipe-lists/  # Expected recipe counts for validation
```

## Sample Books

The following cookbooks are used for testing:

- **Jerusalem** - Expected: 125 recipes
- **Modern Way to Eat** - Expected: 142 recipes
- **Completely Perfect** - Expected: 122 recipes
- **Simple** - Expected: 140 recipes

## Usage

To test recipe extraction with these examples:

```bash
# Basic extraction
uv run python scripts/main_simple_chapters.py examples/input/jerusalem.epub

# With different approaches (experimental)
uv run python scripts/main_chapters_v2.py examples/input/jerusalem.epub
uv run python scripts/main_overlap.py examples/input/jerusalem.epub
```

## Output Format

Extracted recipes are saved in Mela format (`.melarecipe` JSON files) in the `output/` directory at the project root.

Each recipe contains:
- Title
- Ingredients (grouped if applicable)
- Instructions
- Yield/servings
- Prep/cook/total time
- Categories
- Images (base64 encoded)

## Recipe Lists

The `output/recipe-lists/` directory contains text files with expected recipe titles for validation testing. These are used by the test suite to verify extraction accuracy.
