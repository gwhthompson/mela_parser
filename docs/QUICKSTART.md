# Quick Start Guide - Chapter Extractor

Get up and running with the new async chapter-based extraction in 5 minutes.

## Prerequisites

```bash
# Ensure dependencies are installed
uv sync
# or
pip install ebooklib markitdown openai pydantic

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Usage Option 1: Command Line (Recommended)

The fastest way to extract recipes:

```bash
# Basic usage
python main_async_chapters.py cookbook.epub

# That's it! Your recipes will be in:
# output/{book-name}-async-chapters.melarecipes
```

### Common Options

```bash
# Use larger model for better accuracy
python main_async_chapters.py cookbook.epub --model gpt-5-mini

# Process 10 chapters at once (faster)
python main_async_chapters.py cookbook.epub --max-concurrent 10

# Custom output directory
python main_async_chapters.py cookbook.epub --output-dir my_recipes

# See detailed progress
python main_async_chapters.py cookbook.epub --verbose
```

## Usage Option 2: Python API

For integration into your own scripts:

```python
import asyncio
from chapter_extractor import process_epub_chapters

async def extract():
    # One-line extraction
    recipes, diff = await process_epub_chapters("cookbook.epub")

    print(f"Extracted {len(recipes)} recipes")

    if diff:
        print(f"Match rate: {diff.match_rate:.1%}")
        print(f"Missing: {len(diff.missing_titles)} recipes")

    return recipes

# Run it
recipes = asyncio.run(extract())
```

## Usage Option 3: Manual Control

For advanced customization:

```python
import asyncio
from chapter_extractor import (
    ChapterProcessor,
    RecipeListDiscoverer,
    AsyncChapterExtractor,
    ValidationEngine,
)

async def extract_manual():
    # Step 1: Split into chapters
    processor = ChapterProcessor("cookbook.epub")
    chapters = await processor.split_into_chapters()

    # Step 2: Find recipe list
    discoverer = RecipeListDiscoverer()
    expected_titles = await discoverer.discover_from_chapters(chapters)

    # Step 3: Extract recipes (parallel)
    extractor = AsyncChapterExtractor(model="gpt-5-nano")
    results = await extractor.extract_from_chapters(
        chapters,
        expected_titles=expected_titles,
        max_concurrent=5
    )

    # Step 4: Collect unique recipes
    recipes = []
    seen = set()
    for result in results:
        for recipe in result.recipes:
            if recipe.title not in seen:
                seen.add(recipe.title)
                recipes.append(recipe)

    # Step 5: Validate
    validator = ValidationEngine()
    diff = validator.create_diff(expected_titles, recipes)
    print(validator.generate_report(diff))

    return recipes

recipes = asyncio.run(extract_manual())
```

## Output

After running, you'll get:

```
output/
└── {book-name}-async-chapters/
    ├── chocolate-cake.melarecipe
    ├── apple-pie.melarecipe
    ├── banana-bread.melarecipe
    └── ...

output/{book-name}-async-chapters.melarecipes  ← Import this into Mela
```

## Understanding the Output

### Console Output

```
================================================================================
Processing: The Great Cookbook
================================================================================

PHASE 1: Converting EPUB chapters to markdown
Found 42 chapters

PHASE 2: Discovering recipe list
✓ Discovered 127 recipes in cookbook index

PHASE 3: Extracting recipes from chapters (async)
Extraction completed in 45.3s
Total recipes extracted: 129

PHASE 4: Deduplication
Total extracted: 129
Unique recipes: 127

PHASE 5: Validation
Match rate: 98.4% (125/127 recipes)
Missing: 2 recipes
Extra: 0 recipes

PHASE 6: Writing recipes to disk
Recipes written: 127

✓ Output: output/the-great-cookbook-async-chapters.melarecipes
```

### Validation Report

```
================================================================================
VALIDATION REPORT
================================================================================

Expected recipes: 127
Extracted recipes: 127
Exact matches: 125
Match rate: 98.4%

✓ EXACT MATCHES (125):
  ✓ Chocolate Cake
  ✓ Apple Pie
  ✓ Banana Bread
  ...

✗ MISSING RECIPES (2):
  ✗ Secret Family Recipe (may be in different section)
  ✗ Grandma's Special Cookies (may need manual extraction)
```

## Troubleshooting

### Problem: No recipes extracted

```bash
# Solution 1: Try larger model
python main_async_chapters.py cookbook.epub --model gpt-5-mini

# Solution 2: Disable recipe list (blind extraction)
python main_async_chapters.py cookbook.epub --no-recipe-list
```

### Problem: API rate limits

```bash
# Solution: Reduce concurrency
python main_async_chapters.py cookbook.epub --max-concurrent 2
```

### Problem: Low match rate (< 80%)

```bash
# Solution 1: Check the validation report in logs
cat async_chapters_*.log | grep "VALIDATION REPORT" -A 50

# Solution 2: Recipe titles might differ - check missing vs extra
# Solution 3: Try with gpt-5-mini for better title matching
python main_async_chapters.py cookbook.epub --model gpt-5-mini
```

### Problem: Too slow

```bash
# Solution: Increase concurrency
python main_async_chapters.py cookbook.epub --max-concurrent 10

# Note: This increases API costs but speeds up processing
```

## Performance Guide

### Small Cookbooks (< 50 recipes)

```bash
# Go fast!
python main_async_chapters.py cookbook.epub \
  --model gpt-5-nano \
  --max-concurrent 10
```

### Medium Cookbooks (50-200 recipes)

```bash
# Balanced
python main_async_chapters.py cookbook.epub \
  --model gpt-5-nano \
  --max-concurrent 5
```

### Large Cookbooks (> 200 recipes)

```bash
# Prioritize accuracy
python main_async_chapters.py cookbook.epub \
  --model gpt-5-mini \
  --max-concurrent 5 \
  --max-retries 5
```

## Comparison with Old Approach

### Old Way (main_overlap.py)

```bash
python main_overlap.py cookbook.epub
# Problems:
# - Duplicates common
# - Slow (sequential)
# - No validation
# - Manual deduplication needed
```

### New Way (main_async_chapters.py)

```bash
python main_async_chapters.py cookbook.epub
# Benefits:
# ✓ No duplicates (chapter boundaries)
# ✓ Fast (parallel async)
# ✓ Built-in validation
# ✓ Detailed metrics
```

## Next Steps

1. **Try it on your cookbook:**
   ```bash
   python main_async_chapters.py your_cookbook.epub
   ```

2. **Review the output:**
   - Check the validation report
   - Verify match rate is > 80%
   - Import `.melarecipes` into Mela app

3. **Tune if needed:**
   - Adjust `--max-concurrent` based on your needs
   - Try `--model gpt-5-mini` for better accuracy
   - Use `--verbose` to debug issues

4. **Read full docs:**
   - See `CHAPTER_EXTRACTOR_README.md` for complete API documentation
   - See `IMPLEMENTATION_SUMMARY.md` for technical details
   - See `test_chapter_extractor.py` for more examples

## Help

```bash
# Get full help
python main_async_chapters.py --help

# Example output:
usage: main_async_chapters.py [-h] [--model {gpt-5-nano,gpt-5-mini}]
                              [--output-dir OUTPUT_DIR]
                              [--max-concurrent MAX_CONCURRENT]
                              [--max-retries MAX_RETRIES]
                              [--no-recipe-list] [--verbose]
                              epub_path

positional arguments:
  epub_path             Path to EPUB cookbook file

options:
  -h, --help            show this help message and exit
  --model {gpt-5-nano,gpt-5-mini}
                        OpenAI model to use for extraction
  --output-dir OUTPUT_DIR
                        Output directory for recipes
  --max-concurrent MAX_CONCURRENT
                        Maximum concurrent chapter extractions
  --max-retries MAX_RETRIES
                        Maximum retry attempts per chapter
  --no-recipe-list      Skip recipe list discovery phase
  --verbose             Enable verbose (DEBUG) logging
```

## Quick Reference

| Task | Command |
|------|---------|
| Basic extraction | `python main_async_chapters.py book.epub` |
| Better accuracy | `python main_async_chapters.py book.epub --model gpt-5-mini` |
| Faster processing | `python main_async_chapters.py book.epub --max-concurrent 10` |
| Debug mode | `python main_async_chapters.py book.epub --verbose` |
| Skip validation | `python main_async_chapters.py book.epub --no-recipe-list` |
| Custom output | `python main_async_chapters.py book.epub --output-dir my_recipes` |

## Support

For issues or questions:

1. Check the logs: `cat async_chapters_*.log`
2. Review the validation report in the logs
3. See troubleshooting section above
4. Read full documentation in `CHAPTER_EXTRACTOR_README.md`

---

**Happy extracting! 🍳📚**
