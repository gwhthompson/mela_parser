# Examples

This directory is where you place your own EPUB cookbook files for testing.

## Directory Structure

```
examples/
├── input/          # Place your EPUB cookbooks here
└── output/         # Extracted recipes (gitignored)
```

## Usage

1. Place your EPUB cookbook files in `examples/input/`
2. Run the parser:

```bash
# Basic extraction
mela-parse examples/input/your-cookbook.epub

# Or with uv run
uv run mela-parse examples/input/your-cookbook.epub
```

## Output Format

Extracted recipes are saved in Mela format (`.melarecipe` JSON files) in the `output/` directory.

Each recipe contains:
- Title
- Ingredients (grouped if applicable)
- Instructions
- Yield/servings
- Prep/cook/total time
- Categories
- Images (base64 encoded)

## Note

EPUB files are gitignored and not included in this repository. You must provide your own cookbook files for testing.
