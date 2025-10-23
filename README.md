# Mela Recipe Parser

Extract recipes from EPUB cookbooks to [Mela](https://mela.recipes) format.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 📚 **EPUB Parsing** - Extract recipes from EPUB cookbook files
- 🤖 **AI-Powered** - Uses OpenAI GPT models for intelligent recipe extraction
- 📝 **Structured Output** - Exports to Mela-compatible JSON format
- 🎯 **Accurate Extraction** - Chapter-based processing for precise recipe boundaries
- 🧪 **Well-Tested** - Comprehensive test suite with validation against expected outputs

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mela_parser

# Install with uv (recommended)
uv sync --all-extras

# Or install in development mode
uv pip install -e ".[dev]"
```

## Quick Start

```python
from mela_parser import RecipeParser, RecipeProcessor

# Parse a recipe from markdown
recipe_text = """
# Pasta Carbonara
SERVES 4
400g spaghetti
200g pancetta
...
"""

parser = RecipeParser(recipe_text)
recipe = parser.parse()
print(recipe.title)  # "Pasta Carbonara"
```

### Extract from EPUB

```bash
# Using the main extraction script
uv run python scripts/main_simple_chapters.py path/to/cookbook.epub

# Output will be in ./output/ directory
```

## Project Structure

```
mela_parser/
├── src/mela_parser/        # Core package
│   ├── parse.py            # Recipe parsing with OpenAI
│   ├── recipe.py           # Recipe processing and EPUB handling
│   ├── converter.py        # EPUB to Markdown conversion
│   ├── chapter_extractor.py # Chapter-based extraction pipeline
│   ├── prompt_library.py   # Prompt templates
│   └── image_processor.py  # Image extraction and processing
├── tests/                  # Test suite
├── scripts/                # Experimental extraction scripts
├── examples/              # Sample cookbooks and outputs
├── docs/                  # Documentation
└── Makefile              # Common development tasks
```

## Development

```bash
# Install development dependencies
make dev

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Clean generated files
make clean
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Pipeline v2 Guide](docs/testing/PIPELINE_V2_GUIDE.md)
- [Chapter Extractor](docs/CHAPTER_EXTRACTOR_README.md)
- [Testing Guide](docs/TESTING.md)
- [Prompt Design](docs/PROMPT_DESIGN.md)

## Testing

The project includes comprehensive tests for recipe extraction accuracy:

```bash
# Run all tests
uv run pytest

# Run with coverage
make test-cov

# Run specific test file
uv run pytest tests/test_extraction.py -v
```

Test targets verify extraction accuracy against known cookbooks:
- Jerusalem: 125 recipes
- Modern Way to Eat: 142 recipes
- Completely Perfect: 122 recipes
- Simple: 140 recipes

## API Reference

### RecipeParser

Parse individual recipes from markdown text:

```python
from mela_parser import RecipeParser

parser = RecipeParser(recipe_text, model="gpt-5-nano")
recipe = parser.parse()
```

### CookbookParser

Extract multiple recipes from cookbook markdown:

```python
from mela_parser import CookbookParser

parser = CookbookParser(model="gpt-5-mini")
result = parser.parse_cookbook(markdown_content, book_title)
print(f"Extracted {len(result.recipes)} recipes")
```

### RecipeProcessor

Process EPUB files and extract recipes:

```python
from mela_parser import RecipeProcessor

processor = RecipeProcessor("cookbook.epub")
recipe_dict = processor.extract_recipe("chapter1.html#recipe1")
filepath = processor.write_recipe(recipe_dict, output_dir="output")
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional
OPENAI_MODEL=gpt-5-nano  # Default model for parsing
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MarkItDown](https://github.com/microsoft/markitdown) for EPUB conversion
- Uses [OpenAI API](https://openai.com) for intelligent recipe extraction
- Designed for [Mela](https://mela.recipes) recipe manager

## Current Status

🚧 **In Active Development** - Multiple extraction approaches are being tested and refined. See `scripts/` directory for experimental implementations.

**Latest Approach**: Chapter-based extraction with GPT-5-nano for optimal accuracy and cost.
