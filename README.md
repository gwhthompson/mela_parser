# Mela Recipe Parser

Extract recipes from EPUB cookbooks to Mela format.

## Usage

```bash
uv run python main_chapters.py path/to/cookbook.epub
```

## Current Status

**Goal**: Extract exactly 125 recipes from Jerusalem matching official list
**Solution**: Chapter-based extraction
**Status**: `main_chapters.py` implemented, debugging extraction

## The Approach

1. **Discover recipe list** (scan book for links/TOC)
2. **Split by chapters** (EPUB natural structure, no overlap)
3. **Extract per chapter** (MarkItDown + GPT-5-nano, guided by list)
4. **Validate** against expected counts

## Test Targets

- Jerusalem: 125 recipes
- Modern Way: 142 recipes
- Completely Perfect: 122 recipes
- Simple: 140 recipes

Lists in: `examples/output/recipe-lists/`

## Next

Fix `main_chapters.py` to extract exactly 125 recipes from Jerusalem.
