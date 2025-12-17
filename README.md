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
# Install as a UV tool (recommended)
uv tool install git+https://github.com/yourusername/mela_parser.git

# Or install from local directory
cd mela_parser
uv tool install .

# For development
uv sync --all-extras
```

## Quick Start

```bash
# Extract recipes from an EPUB cookbook
mela-parse path/to/cookbook.epub

# With options
mela-parse cookbook.epub --model gpt-5-mini --output-dir my_recipes
```

### Options

- `--model`: Choose OpenAI model (`gpt-5-nano` [default] or `gpt-5-mini`)
- `--output-dir`: Output directory (default: `output`)
- `--no-images`: Skip image extraction

## Project Structure

```
mela_parser/
├── src/mela_parser/        # Core package
│   ├── cli.py              # CLI entry point
│   ├── parse.py            # Recipe parsing with OpenAI
│   ├── recipe.py           # Recipe processing and EPUB handling
│   ├── converter.py        # EPUB to Markdown conversion
│   ├── chapter_extractor.py # Chapter-based extraction pipeline
│   ├── validator.py        # Recipe quality validation
│   └── image_processor.py  # Image extraction and processing
├── tests/                  # Test suite
├── examples/              # Sample cookbooks and outputs
├── docs/                  # Documentation
└── Makefile              # Common development tasks
```

## Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run CLI locally (without installing)
uv run mela-parse examples/input/simple.epub
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
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

## How It Works

1. **Chapter Conversion**: EPUB chapters are converted to clean markdown
2. **Parallel Extraction**: Extracts recipes from all chapters simultaneously using async processing
3. **Deduplication**: Removes duplicate recipes based on title
4. **Export**: Saves recipes in Mela-compatible `.melarecipes` format

## Requirements

- Python 3.13+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- UV package manager (for installation)

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

✅ **Production Ready** - Chapter-based extraction with async parallel processing for optimal speed and accuracy.
