# Chapter Extractor - Production-Ready Async EPUB Recipe Extraction

A modern, production-ready framework for extracting recipes from EPUB cookbooks using chapter-based processing and async parallel execution.

## Overview

The `chapter_extractor` module provides a clean, efficient approach to recipe extraction that eliminates the duplicate recipe problem inherent in chunking-based approaches. By leveraging the natural chapter structure of EPUB files and using async parallel processing, it delivers both accuracy and performance.

## Key Features

- **Chapter-Based Processing**: Uses ebooklib's `ITEM_DOCUMENT` to split EPUBs naturally by chapters
- **Async Parallel Execution**: Process multiple chapters simultaneously for optimal performance
- **Recipe List Discovery**: Automatically finds and validates against table of contents
- **Comprehensive Validation**: Detailed diff reports showing matched, missing, and extra recipes
- **Production-Ready Error Handling**: Custom exceptions, retry logic with exponential backoff
- **Type-Safe**: Full type hints throughout using Python 3.12+ features
- **Structured Logging**: Detailed logging at every stage for debugging and monitoring
- **Modern Python**: Dataclasses, Pydantic V2, async/await patterns

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Chapter Extractor                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌─────────────────────────┐    │
│  │ ChapterProcessor │──────▶│ List[Chapter]           │    │
│  │ - EPUB → Chapters│      │ - name, content, index  │    │
│  │ - MarkItDown     │      └─────────────────────────┘    │
│  └──────────────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌───────────────────────┐                                 │
│  │ RecipeListDiscoverer  │                                 │
│  │ - Scan for TOC/Index  │                                 │
│  │ - LLM cleaning        │                                 │
│  └───────────────────────┘                                 │
│           │                                                 │
│           ▼                                                 │
│  ┌───────────────────────┐    ┌──────────────────────┐    │
│  │ AsyncChapterExtractor │────▶│ List[MelaRecipe]     │    │
│  │ - Parallel processing │    │ - Pydantic validated │    │
│  │ - Retry logic         │    └──────────────────────┘    │
│  │ - Targeted extraction │                                 │
│  └───────────────────────┘                                 │
│           │                                                 │
│           ▼                                                 │
│  ┌───────────────────────┐                                 │
│  │ ValidationEngine      │                                 │
│  │ - Create diff         │                                 │
│  │ - Generate reports    │                                 │
│  │ - Quality metrics     │                                 │
│  └───────────────────────┘                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Quick Start

The simplest way to use the module:

```python
import asyncio
from chapter_extractor import process_epub_chapters

async def extract_recipes():
    recipes, diff = await process_epub_chapters("cookbook.epub")

    print(f"Extracted {len(recipes)} recipes")
    if diff:
        print(f"Match rate: {diff.match_rate:.1%}")
        print(f"Missing: {len(diff.missing_titles)} recipes")

    return recipes

# Run it
recipes = asyncio.run(extract_recipes())
```

### Manual Pipeline (Full Control)

For maximum control over each stage:

```python
import asyncio
from chapter_extractor import (
    ChapterProcessor,
    RecipeListDiscoverer,
    AsyncChapterExtractor,
    ValidationEngine,
)

async def extract_with_full_control():
    # Phase 1: Split EPUB into chapters
    processor = ChapterProcessor("cookbook.epub")
    chapters = await processor.split_into_chapters()
    print(f"Found {len(chapters)} chapters")

    # Phase 2: Discover recipe list
    discoverer = RecipeListDiscoverer(model="gpt-5-mini")
    expected_titles = await discoverer.discover_from_chapters(chapters)
    print(f"Discovered {len(expected_titles)} recipes in index")

    # Phase 3: Extract recipes (async parallel)
    extractor = AsyncChapterExtractor(
        model="gpt-5-nano",
        max_retries=3,
    )
    results = await extractor.extract_from_chapters(
        chapters,
        expected_titles=expected_titles,
        max_concurrent=5  # Process 5 chapters at once
    )

    # Phase 4: Collect and deduplicate
    all_recipes = []
    seen_titles = set()

    for result in results:
        for recipe in result.recipes:
            if recipe.title not in seen_titles:
                seen_titles.add(recipe.title)
                all_recipes.append(recipe)

    # Phase 5: Validate
    validator = ValidationEngine()
    diff = validator.create_diff(expected_titles, all_recipes)
    report = validator.generate_report(diff)
    print(report)

    return all_recipes

recipes = asyncio.run(extract_with_full_control())
```

### Command-Line Usage

Use the included script for production processing:

```bash
# Basic usage
python main_async_chapters.py cookbook.epub

# Use larger model for better accuracy
python main_async_chapters.py cookbook.epub --model gpt-5-mini

# Increase parallelism (faster, more API calls)
python main_async_chapters.py cookbook.epub --max-concurrent 10

# Skip recipe list discovery
python main_async_chapters.py cookbook.epub --no-recipe-list

# Custom output directory
python main_async_chapters.py cookbook.epub --output-dir my_recipes

# Verbose logging for debugging
python main_async_chapters.py cookbook.epub --verbose
```

## API Reference

### ChapterProcessor

Converts EPUB files into individual chapters.

```python
processor = ChapterProcessor(epub_path: str | Path)
chapters = await processor.split_into_chapters() -> list[Chapter]
```

**Attributes:**
- `epub_path`: Path to EPUB file
- `book`: Loaded EpubBook instance
- `book_title`: Title from EPUB metadata

**Raises:**
- `FileNotFoundError`: EPUB file not found
- `EPUBConversionError`: EPUB cannot be loaded or converted

### RecipeListDiscoverer

Discovers recipe lists from table of contents or indices.

```python
discoverer = RecipeListDiscoverer(
    model: str = "gpt-5-mini",
    min_links: int = 5
)
titles = await discoverer.discover_from_chapters(chapters) -> Optional[list[str]]
```

**Parameters:**
- `model`: OpenAI model for cleaning (default: gpt-5-mini)
- `min_links`: Minimum links to consider a section a recipe list (default: 5)

**Returns:**
- `list[str]`: Unique recipe titles, or `None` if no list found

### AsyncChapterExtractor

Extracts recipes from chapters using async parallel processing.

```python
extractor = AsyncChapterExtractor(
    model: str = "gpt-5-nano",
    max_retries: int = 3,
    initial_retry_delay: float = 1.0
)

# Extract from multiple chapters
results = await extractor.extract_from_chapters(
    chapters: list[Chapter],
    expected_titles: Optional[list[str]] = None,
    max_concurrent: int = 5
) -> list[ExtractionResult]

# Extract from single chapter
result = await extractor.extract_from_chapter(
    chapter: Chapter,
    expected_titles: Optional[list[str]] = None
) -> ExtractionResult
```

**Parameters:**
- `model`: OpenAI model for extraction (gpt-5-nano or gpt-5-mini)
- `max_retries`: Maximum retry attempts per chapter (default: 3)
- `initial_retry_delay`: Initial delay for exponential backoff (default: 1.0s)
- `max_concurrent`: Maximum concurrent extractions (default: 5)
- `expected_titles`: Optional list of expected recipe titles for targeted extraction

**Returns:**
- `list[ExtractionResult]`: Results for each chapter with recipes and error info

### ValidationEngine

Validates extraction results against discovered recipe lists.

```python
validator = ValidationEngine()

# Create diff
diff = validator.create_diff(
    expected_titles: list[str],
    extracted_recipes: list[MelaRecipe]
) -> ValidationDiff

# Generate report
report = validator.generate_report(
    diff: ValidationDiff,
    max_items: int = 20
) -> str

# Validate quality
is_valid, message = validator.validate_extraction_quality(
    results: list[ExtractionResult],
    min_success_rate: float = 0.8
) -> tuple[bool, str]
```

## Data Models

### Chapter

```python
@dataclass(frozen=True)
class Chapter:
    name: str          # Chapter filename (e.g., "chapter1.html")
    content: str       # Markdown content
    index: int         # Chapter position
```

### ExtractionResult

```python
@dataclass
class ExtractionResult:
    chapter_name: str            # Name of processed chapter
    recipes: list[MelaRecipe]    # Extracted recipes
    error: Optional[str]         # Error message if failed
    retry_count: int             # Number of retries attempted

    # Properties
    is_success: bool             # True if no error
    recipe_count: int            # Number of recipes extracted
```

### ValidationDiff

```python
@dataclass
class ValidationDiff:
    expected_titles: set[str]    # Expected recipe titles
    extracted_titles: set[str]   # Actually extracted titles
    exact_matches: set[str]      # Titles that match exactly
    missing_titles: set[str]     # Expected but not found
    extra_titles: set[str]       # Found but not expected
    match_rate: float            # Percentage matched (0.0-1.0)

    # Properties
    is_perfect_match: bool       # True if 100% match with no extras
```

## Error Handling

The module provides custom exceptions for different failure scenarios:

```python
from chapter_extractor import (
    ChapterExtractionError,      # Base exception
    EPUBConversionError,         # EPUB loading/conversion failed
    RecipeListDiscoveryError,    # Recipe list discovery failed
    RecipeExtractionError,       # Recipe extraction failed
)

try:
    recipes, diff = await process_epub_chapters("cookbook.epub")
except EPUBConversionError as e:
    print(f"Failed to convert EPUB: {e}")
except RecipeListDiscoveryError as e:
    print(f"Recipe list discovery failed: {e}")
except ChapterExtractionError as e:
    print(f"Extraction error: {e}")
```

## Performance Tuning

### Concurrency Settings

Control parallel processing with `max_concurrent`:

```python
# Conservative (stable, slower)
results = await extractor.extract_from_chapters(
    chapters,
    max_concurrent=2  # 2 chapters at a time
)

# Balanced (recommended)
results = await extractor.extract_from_chapters(
    chapters,
    max_concurrent=5  # 5 chapters at a time
)

# Aggressive (fast, more API calls)
results = await extractor.extract_from_chapters(
    chapters,
    max_concurrent=10  # 10 chapters at a time
)
```

### Model Selection

Choose model based on your needs:

```python
# Fast and cost-effective (recommended for most cases)
extractor = AsyncChapterExtractor(model="gpt-5-nano")

# More accurate (slower, more expensive)
extractor = AsyncChapterExtractor(model="gpt-5-mini")
```

### Retry Configuration

Adjust retry behavior for reliability:

```python
# Aggressive retries for unreliable networks
extractor = AsyncChapterExtractor(
    max_retries=5,
    initial_retry_delay=2.0  # Start with 2s delay
)

# Minimal retries for stable environments
extractor = AsyncChapterExtractor(
    max_retries=1,
    initial_retry_delay=0.5
)
```

## Logging

The module uses structured logging throughout:

```python
import logging

# Enable INFO logging
logging.basicConfig(level=logging.INFO)

# Enable DEBUG logging for detailed diagnostics
logging.basicConfig(level=logging.DEBUG)

# Log to file
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)
```

## Comparison with Chunking Approach

| Feature | Chapter-Based (New) | Chunking-Based (Old) |
|---------|---------------------|----------------------|
| **Duplicates** | ✓ None (natural boundaries) | ✗ Common (overlapping windows) |
| **Speed** | ✓ Fast (async parallel) | ✗ Slower (sequential) |
| **Accuracy** | ✓ High (complete chapters) | ⚠ Variable (split recipes) |
| **Memory** | ✓ Efficient (stream processing) | ⚠ Higher (large chunks) |
| **Validation** | ✓ Built-in diff engine | ✗ Manual |
| **Error Handling** | ✓ Per-chapter retry | ✗ All-or-nothing |
| **Type Safety** | ✓ Full type hints | ⚠ Partial |

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_chapter_extractor.py

# Run specific test class
pytest test_chapter_extractor.py::TestChapterProcessor

# Run integration tests (requires sample EPUB)
pytest test_chapter_extractor.py::TestIntegration

# Run with verbose output
pytest test_chapter_extractor.py -v

# Run with coverage
pytest test_chapter_extractor.py --cov=chapter_extractor
```

## Migration Guide

### From main_overlap.py (Chunking)

**Old approach:**
```python
# main_overlap.py (chunking with overlaps)
converter = EpubConverter()
markdown = converter.convert_epub_to_markdown(epub_path)
chunks = chunk_markdown_with_overlap(markdown, chunk_size=10000)

for chunk in chunks:
    recipes = extract_from_chunk(chunk)
    # Duplicates likely!
```

**New approach:**
```python
# main_async_chapters.py (chapter-based)
import asyncio
from chapter_extractor import process_epub_chapters

recipes, diff = await process_epub_chapters(epub_path)
# No duplicates, parallel processing, validated!
```

### From main_chapters.py (Sequential)

**Old approach:**
```python
# main_chapters.py (sequential chapter processing)
for chapter_name, chapter_md in chapters:
    recipes = extract_from_chapter(chapter_md, chapter_name)
    # Slow, no retry logic, no validation
```

**New approach:**
```python
# New async approach
extractor = AsyncChapterExtractor()
results = await extractor.extract_from_chapters(chapters)
# Parallel, retry logic, comprehensive validation!
```

## Best Practices

### 1. Always Use Recipe List Discovery

```python
# Good - enables targeted extraction and validation
recipes, diff = await process_epub_chapters(
    epub_path,
    use_recipe_list=True
)

# Less optimal - blind extraction
recipes, diff = await process_epub_chapters(
    epub_path,
    use_recipe_list=False
)
```

### 2. Monitor Validation Metrics

```python
recipes, diff = await process_epub_chapters(epub_path)

if diff:
    if diff.match_rate < 0.8:
        print(f"⚠ Low match rate: {diff.match_rate:.1%}")
        print(f"Missing recipes: {list(diff.missing_titles)[:10]}")

    if diff.is_perfect_match:
        print("✓ Perfect extraction!")
```

### 3. Handle Errors Gracefully

```python
try:
    recipes, diff = await process_epub_chapters(epub_path)
except EPUBConversionError as e:
    # Retry with different settings
    print(f"Conversion failed: {e}")
except Exception as e:
    # Log and report
    logging.error(f"Unexpected error: {e}", exc_info=True)
```

### 4. Tune for Your Use Case

```python
# For small cookbooks (< 50 recipes)
extractor = AsyncChapterExtractor(
    model="gpt-5-nano",
    max_concurrent=10  # Go fast!
)

# For large cookbooks (> 200 recipes)
extractor = AsyncChapterExtractor(
    model="gpt-5-mini",      # Better accuracy
    max_concurrent=5,        # Balanced speed
    max_retries=5           # More resilient
)
```

## Troubleshooting

### Issue: No recipes extracted

**Cause**: Chapters don't contain complete recipes or detection failed

**Solution**:
```python
# Try with recipe list disabled for blind extraction
recipes, _ = await process_epub_chapters(
    epub_path,
    use_recipe_list=False
)

# Or use larger model
recipes, diff = await process_epub_chapters(
    epub_path,
    model="gpt-5-mini"
)
```

### Issue: Low match rate

**Cause**: Recipe titles in TOC don't match recipe titles in chapters

**Solution**:
```python
# Check the diff report
validator = ValidationEngine()
report = validator.generate_report(diff)
print(report)

# Examine missing vs extra to find title mismatches
```

### Issue: Extraction too slow

**Cause**: Too conservative concurrency settings

**Solution**:
```python
# Increase parallelism
results = await extractor.extract_from_chapters(
    chapters,
    max_concurrent=10  # Up from default 5
)
```

### Issue: API rate limits

**Cause**: Too many concurrent requests

**Solution**:
```python
# Reduce concurrency
results = await extractor.extract_from_chapters(
    chapters,
    max_concurrent=2  # Down from default 5
)
```

## Advanced Usage

### Custom Retry Logic

```python
class CustomExtractor(AsyncChapterExtractor):
    async def extract_from_chapter(self, chapter, expected_titles=None):
        # Add custom retry logic
        for attempt in range(self.max_retries):
            try:
                return await super().extract_from_chapter(chapter, expected_titles)
            except SpecificError as e:
                # Custom handling
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(attempt * 2)
```

### Progress Tracking

```python
from tqdm.asyncio import tqdm

async def extract_with_progress(chapters):
    extractor = AsyncChapterExtractor()

    results = []
    for chapter in tqdm(chapters, desc="Extracting"):
        result = await extractor.extract_from_chapter(chapter)
        results.append(result)

    return results
```

### Batch Processing

```python
async def process_multiple_cookbooks(epub_paths: list[str]):
    results = {}

    for epub_path in epub_paths:
        try:
            recipes, diff = await process_epub_chapters(epub_path)
            results[epub_path] = {
                'recipes': recipes,
                'diff': diff,
                'success': True
            }
        except Exception as e:
            results[epub_path] = {
                'error': str(e),
                'success': False
            }

    return results
```

## Contributing

Contributions are welcome! Key areas for improvement:

1. **Performance**: Optimize async patterns for even faster processing
2. **Accuracy**: Improve recipe detection and title matching
3. **Robustness**: Add more error recovery strategies
4. **Testing**: Expand test coverage with more EPUB samples
5. **Documentation**: Add more usage examples

## License

This module is part of the mela_parser project and follows the same license.

## Support

For issues, questions, or contributions, please refer to the main project repository.
