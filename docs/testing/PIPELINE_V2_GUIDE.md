# LLM-Powered Recipe Extraction Pipeline v2 - Complete Guide

## Overview

`main_chapters_v2.py` is a production-ready recipe extraction pipeline that achieves 100% match rates through **automatic prompt iteration**. It uses LLM-driven analysis to identify and fix extraction gaps iteratively.

## Architecture

### Core Components

1. **ChapterProcessor**: Converts EPUB chapters to markdown using MarkItDown
2. **RecipeListDiscoverer**: Discovers comprehensive recipe lists from book structure (gpt-5-mini)
3. **ChapterExtractor**: Extracts recipes from chapters in parallel with async/await (gpt-5-nano)
4. **ExtractionPipeline**: Orchestrates the entire extraction and iteration workflow
5. **Validation & Gap Analysis**: Compares results and identifies failure patterns using LLM analysis

### Data Flow

```
EPUB → Chapters → Recipe List Discovery → Parallel Extraction → Validation → Gap Analysis → Prompt Improvement → Retry
                                                                      ↓
                                                                  100% Match?
                                                                      ↓
                                                              Write to Disk
```

## Key Features

### 1. Automatic Prompt Iteration

The pipeline doesn't just extract once - it **learns and improves**:

- **Iteration 1**: Extract with default prompts
- **Validate**: Compare against discovered recipe list
- **Analyze**: LLM analyzes WHY recipes were missed or duplicated
- **Improve**: LLM rewrites prompts to fix identified issues
- **Retry**: Re-extract with improved prompts
- **Repeat**: Until 100% match or max iterations

### 2. No Ground Truth Cheating

- **Discovery Phase**: Uses only the EPUB's own table of contents/recipe lists
- **Validation**: Compares extracted recipes against discovered list
- **Ground Truth**: Only used in final tests, never during extraction

### 3. Production-Ready Error Handling

- Custom exceptions: `ExtractionError`, `ValidationError`, `PromptOptimizationError`
- Exponential backoff retry for API calls (3 attempts per chapter)
- Graceful degradation: Failed chapters don't stop the pipeline
- Comprehensive logging at every phase

### 4. Parallel Processing

- Processes chapters concurrently with `asyncio.gather()`
- Configurable concurrency limit (default: 5 concurrent chapters)
- Semaphore-based rate limiting to avoid API throttling

### 5. Comprehensive Iteration Tracking

Every iteration saves:
- Match percentage
- Missing/extra titles
- Prompt version used
- Execution time
- Full validation report
- Prompt change reasoning

## Usage

### Basic Usage (Iterative Mode)

```bash
.venv/bin/python main_chapters_v2.py examples/input/jerusalem.epub \
  --model gpt-5-nano \
  --max-iterations 10 \
  --output-dir output/jerusalem-v2
```

This will:
1. Extract recipes using default prompts
2. Validate against discovered list
3. If not 100%, analyze gaps and improve prompts
4. Retry with new prompts
5. Repeat until 100% match or 10 iterations
6. Save all recipes + iteration history

### Single-Pass Mode (No Iteration)

```bash
.venv/bin/python main_chapters_v2.py examples/input/cookbook.epub \
  --skip-iteration \
  --output-dir output/cookbook
```

Use when you have pre-tuned prompts or want a quick test.

### Using Custom Prompts

```bash
.venv/bin/python main_chapters_v2.py examples/input/cookbook.epub \
  --prompt-library output/final_prompts.json \
  --skip-iteration
```

Load prompts from a previous successful run.

### Verbose Mode

```bash
.venv/bin/python main_chapters_v2.py examples/input/cookbook.epub \
  --verbose \
  --max-iterations 5
```

Enables DEBUG-level logging for detailed diagnostics.

## Output Files

### 1. Recipes Archive

```
output/
  └── {book-slug}-chapters-v2/
      ├── recipe-1.melarecipe
      ├── recipe-2.melarecipe
      └── ...
  └── {book-slug}-chapters-v2.melarecipes  # ZIP archive
```

### 2. Iteration Snapshots

```
output/
  └── iteration_1.json
  └── iteration_2.json
  └── ...
```

Each contains:
- Iteration metrics
- Full validation report
- Prompts used for that iteration

### 3. Final Outputs

```
output/
  └── final_prompts.json       # Winning prompts (locked)
  └── iteration_history.json   # Complete iteration log
  └── extraction_pipeline.log  # Full execution log
```

## Prompt Library Structure

```json
{
  "discovery_prompt": "Template for discovering recipe lists...",
  "extraction_prompt": "Template for extracting from chapters...",
  "version": 3,
  "iteration_history": [
    {
      "version": 2,
      "changes": "Fixed issue with multi-line titles...",
      "confidence": 0.85
    }
  ],
  "locked": false
}
```

### Placeholders

Discovery prompt must include:
- `{combined_lists}`: Where concatenated recipe list sections go

Extraction prompt must include:
- `{expected_list}`: Where expected recipe titles go
- `{chapter_md}`: Where chapter markdown content goes

## Gap Analysis

The LLM analyzer examines:

1. **Missing Recipe Patterns**
   - Are they in a specific format? (e.g., "Recipe: Title" vs "Title")
   - Are they in special sections? (e.g., appendix, index)
   - Do they use different capitalization?
   - Are they split across pages?

2. **False Positive Patterns**
   - What's being extracted that shouldn't be?
   - Recipe lists vs. actual recipes
   - Section headers vs. recipe titles
   - Cross-references vs. full recipes

3. **Prompt Weaknesses**
   - Too vague?
   - Missing edge case handling?
   - Wrong delimiters?
   - Unclear instructions?

## Error Handling

### ExtractionError

Raised when:
- EPUB conversion fails
- Chapter processing fails completely
- Unable to read EPUB file

**Recovery**: Check EPUB file integrity, ensure MarkItDown dependencies installed

### ValidationError

Raised when:
- No discovered recipe list found
- Validation comparison fails

**Recovery**: Ensure cookbook has a table of contents or recipe index

### PromptOptimizationError

Raised when:
- Gap analysis API call fails
- Prompt rewriting fails
- Unable to parse LLM suggestions

**Recovery**: Check API credentials, reduce max iterations, use --skip-iteration

## Performance Tuning

### Concurrency

Adjust parallel chapter processing:

```python
pipeline = ExtractionPipeline(max_concurrent_chapters=10)  # Default: 5
```

Higher values = faster but more API load.

### Model Selection

```bash
--model gpt-5-nano   # Faster, cheaper, good for structured recipes
--model gpt-5-mini   # Slower, more accurate, better for complex layouts
```

### Iteration Limits

```bash
--max-iterations 3   # Quick iteration
--max-iterations 10  # Thorough iteration (default)
--max-iterations 20  # Exhaustive iteration
```

## Debugging

### Enable Verbose Logging

```bash
.venv/bin/python main_chapters_v2.py book.epub --verbose
```

Shows:
- Chapter-by-chapter extraction details
- API call traces
- Prompt text before/after improvements
- Recipe-level validation

### Examine Iteration Snapshots

```bash
cat output/iteration_1.json | jq '.validation.missing_titles'
```

See exactly which recipes were missed in each iteration.

### Review Prompt Evolution

```bash
cat output/final_prompts.json | jq '.iteration_history'
```

Understand how prompts evolved to fix issues.

## Best Practices

### 1. Start with Iterative Mode

Always run with iteration enabled first - it finds edge cases you didn't know existed.

### 2. Save Winning Prompts

Once you achieve 100% on a cookbook style (e.g., modern cookbooks vs. vintage):

```bash
cp output/final_prompts.json prompts/modern_cookbooks.json
```

Reuse for similar books with `--prompt-library`.

### 3. Monitor Iteration Trends

If match rate plateaus (e.g., stuck at 95% for 3 iterations):
- Manually inspect missing recipes
- Check if they're truly recipes or false positives in ground truth
- Adjust max iterations or accept current result

### 4. Batch Processing

For multiple books:

```bash
for book in examples/input/*.epub; do
  .venv/bin/python main_chapters_v2.py "$book" \
    --max-iterations 10 \
    --output-dir "output/$(basename $book .epub)"
done
```

### 5. Cost Optimization

- Use `gpt-5-nano` for extraction (cheaper)
- Reserve `gpt-5-mini` for discovery and gap analysis (more expensive but only runs once per iteration)
- Set `--max-iterations 5` for production to cap costs

## Integration with Existing Pipeline

### Use with Image Processing

```python
from image_processor import ImageProcessor
from main_chapters_v2 import ExtractionPipeline

# After extraction
pipeline = ExtractionPipeline()
results, chapters, discovered = await pipeline.extract_recipes(epub_path, prompts)

# Add images
image_processor = ImageProcessor(book, model="gpt-5-nano")
for recipe in results.recipes:
    # Extract images for recipe...
```

### Use with Recipe Processor

Already integrated! Pipeline automatically uses `RecipeProcessor` for writing recipes to disk.

### Use in Tests

```python
from main_chapters_v2 import ExtractionPipeline
import asyncio

async def test_extraction():
    pipeline = ExtractionPipeline()
    results, _, discovered = await pipeline.extract_recipes(
        "test.epub",
        PromptLibrary.default(),
        model="gpt-5-nano"
    )

    assert len(results.recipes) > 0
    assert results.unique_count == len(results.recipes)

asyncio.run(test_extraction())
```

## Troubleshooting

### Issue: No recipes discovered

**Symptom**: "No recipe list sections found"

**Cause**: Book doesn't have a table of contents with recipe links

**Solution**: Use `--skip-recipe-list` flag (falls back to extracting all recipes found)

### Issue: Stuck at same match rate

**Symptom**: Iterations 5-8 all show 92% match

**Cause**: LLM unable to improve prompts further

**Solution**:
1. Manually inspect `iteration_5.json` to see missing titles
2. Check if missing items are edge cases or data issues
3. Consider accepting 92% if missing items aren't true recipes

### Issue: Extraction very slow

**Symptom**: 30+ minutes for 50 chapters

**Cause**: Sequential processing or low concurrency

**Solution**: Increase `max_concurrent_chapters` to 10-15

### Issue: API rate limit errors

**Symptom**: "Rate limit exceeded" in logs

**Cause**: Too many concurrent requests

**Solution**: Decrease `max_concurrent_chapters` to 3

## Advanced: Custom Gap Analysis

Extend gap analysis with domain-specific rules:

```python
class CustomPipeline(ExtractionPipeline):
    async def analyze_gaps(self, validation_report, chapters, prompts):
        # Call parent
        improvements = await super().analyze_gaps(validation_report, chapters, prompts)

        # Add custom analysis
        if any("continued" in title.lower() for title in validation_report.missing_titles):
            improvements.suggested_extraction_changes += (
                "\n\nAdd rule: Skip recipes with '(continued)' suffix as they're "
                "multi-page splits of previous recipe."
            )

        return improvements
```

## Metrics & Monitoring

The pipeline tracks:

- **Extraction time**: Total time for chapter processing
- **Match rate**: Percentage of discovered recipes successfully extracted
- **Iterations needed**: How many attempts to reach 100%
- **Prompt version**: Which prompt version achieved best results
- **Duplicate rate**: How many duplicates were filtered
- **Chapter yield**: Which chapters contained recipes

Use `iteration_history.json` for analysis:

```python
import json
import pandas as pd

with open("output/iteration_history.json") as f:
    history = json.load(f)

df = pd.DataFrame(history)
print(df[["iteration", "match_percentage", "missing", "extra"]])
```

## Future Enhancements

1. **Multi-model consensus**: Run with both gpt-5-nano and gpt-5-mini, merge results
2. **Smart chapter filtering**: Skip non-recipe chapters (dedupe, introduction) early
3. **Streaming extraction**: Start writing recipes before all chapters complete
4. **Prompt A/B testing**: Try multiple prompt variants per iteration, pick best
5. **Transfer learning**: Use prompts from similar cookbooks as starting point

## Support

For issues or questions:
1. Check logs: `extraction_pipeline.log`
2. Review iteration snapshots: `output/iteration_*.json`
3. Enable verbose mode: `--verbose`
4. Check API credentials and quota
