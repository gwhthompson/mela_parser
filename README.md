# Mela Recipe Parser

A tool for extracting recipes from EPUB books and converting them to the Mela recipe format.

## Features

- Parse EPUB files to extract recipe content
- Convert recipe text to structured Mela recipe format
- Extract and process images from EPUB files
- Generate individual .melarecipe files and a combined .melarecipes archive

## Usage

```bash
uv run main.py path/to/cookbook.epub
```

The tool will:
1. Extract recipes from the EPUB file
2. Parse each recipe into a structured format
3. Save individual recipes as .melarecipe files
4. Create a combined .melarecipes archive

## Output

- Individual recipes: `output/<book-slug>/<recipe-slug>.melarecipe`
- Combined archive: `output/<book-slug>.melarecipes`

## Development

### Requirements

- Python 3.13+
- Dependencies listed in pyproject.toml

## License

MIT
