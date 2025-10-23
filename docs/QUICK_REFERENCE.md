# Pipeline v2 - Quick Reference Card

## 🚀 Common Commands

### Basic Extraction (Iterative)
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub
```
→ Extracts with automatic prompt improvement until 100% match

### Fast Extraction (Single-Pass)
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub --skip-iteration
```
→ Quick extraction without iteration (use for testing)

### Custom Iterations
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub --max-iterations 5
```
→ Limit iterations (faster, may not reach 100%)

### Use Pre-Tuned Prompts
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub \
  --prompt-library output/final_prompts.json \
  --skip-iteration
```
→ Reuse successful prompts from previous runs

### Debug Mode
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub --verbose
```
→ Detailed logging for troubleshooting

### Change Model
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub --model gpt-5-mini
```
→ Use gpt-5-mini for extraction (slower but more accurate)

### Custom Output Directory
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub \
  --output-dir output/my-cookbook
```
→ Specify where to save results

## 📂 Output Files

After running, check:

```
output/
├── {book-slug}-chapters-v2/          # Individual recipe files
│   ├── recipe-1.melarecipe
│   ├── recipe-2.melarecipe
│   └── ...
├── {book-slug}-chapters-v2.melarecipes  # ZIP archive (import to Mela)
├── iteration_1.json                  # First iteration snapshot
├── iteration_2.json                  # Second iteration snapshot
├── ...
├── final_prompts.json               # Winning prompts (save for reuse!)
├── iteration_history.json           # Complete iteration log
└── extraction_pipeline.log          # Execution log
```

## 🔍 Check Results

### View Final Match Rate
```bash
cat output/iteration_history.json | jq '.[-1].match_percentage'
```

### See Missing Recipes
```bash
cat output/iteration_history.json | jq '.[-1].missing_titles'
```

### Count Extracted Recipes
```bash
ls -1 output/*-chapters-v2/*.melarecipe | wc -l
```

### View Prompt Evolution
```bash
cat output/final_prompts.json | jq '.iteration_history'
```

## 🐛 Troubleshooting

### No recipes extracted?
```bash
# Check if recipe list was discovered
grep "Discovered" extraction_pipeline.log

# If not, EPUB may not have a recipe index
# Try with exploration mode (future feature)
```

### Stuck at low match rate (< 90%)?
```bash
# Check what's missing
cat output/iteration_3.json | jq '.validation.missing_titles'

# Manually inspect those recipes in EPUB
# May be edge cases or data issues
```

### API errors?
```bash
# Check logs
tail -50 extraction_pipeline.log

# Verify API key
echo $OPENAI_API_KEY

# Reduce concurrency
# (edit main_chapters_v2.py, line ~742: max_concurrent_chapters=3)
```

### Too slow?
```bash
# Use gpt-5-nano (default, faster)
# Reduce max iterations
--max-iterations 3

# Or skip iteration entirely
--skip-iteration
```

## 🎯 Best Practices

### 1. First Run: Let it iterate
```bash
.venv/bin/python main_chapters_v2.py cookbook.epub --max-iterations 10
```
→ Discovers edge cases and tunes prompts

### 2. Save Winning Prompts
```bash
cp output/final_prompts.json prompts/modern-cookbooks.json
```
→ Reuse on similar cookbooks

### 3. Second Run: Use saved prompts
```bash
.venv/bin/python main_chapters_v2.py another-cookbook.epub \
  --prompt-library prompts/modern-cookbooks.json \
  --skip-iteration
```
→ Fast extraction with proven prompts

### 4. Batch Processing
```bash
for book in examples/input/*.epub; do
  .venv/bin/python main_chapters_v2.py "$book" \
    --output-dir "output/$(basename $book .epub)" \
    --max-iterations 5
done
```
→ Process entire collection

## 📊 Performance Tips

| Scenario | Command | Speed | Cost |
|----------|---------|-------|------|
| Quick test | `--skip-iteration` | ⚡️ Fast | 💰 Low |
| Production | `--max-iterations 10` | 🐢 Slow | 💰💰 Medium |
| Reuse prompts | `--prompt-library X --skip-iteration` | ⚡️ Fast | 💰 Low |
| High accuracy | `--model gpt-5-mini --max-iterations 10` | 🐌 Very slow | 💰💰💰 High |

## 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| "No recipe list found" | EPUB has no TOC, try `--skip-recipe-list` (future) |
| "Rate limit exceeded" | Reduce `max_concurrent_chapters` in code |
| "Out of memory" | Process fewer chapters at once |
| "Stuck at 95%" | May have reached limit, check missing recipes manually |
| "API key invalid" | Set `export OPENAI_API_KEY=sk-...` |

## 📚 Full Documentation

- **User Guide**: `docs/PIPELINE_V2_GUIDE.md`
- **Implementation**: `IMPLEMENTATION_V2_SUMMARY.md`
- **Tests**: `test_pipeline_v2.py`
- **Examples**: `examples_pipeline_usage.py`

## 🔗 Integration

### Use in Python Code
```python
from main_chapters_v2 import ExtractionPipeline
import asyncio

async def extract():
    pipeline = ExtractionPipeline(max_concurrent_chapters=5)
    recipes, prompts, history = await pipeline.iterative_refinement(
        epub_path="cookbook.epub",
        max_iterations=10,
        model="gpt-5-nano",
    )
    print(f"Extracted {len(recipes)} recipes")
    return recipes

asyncio.run(extract())
```

### Run Tests
```bash
.venv/bin/python -m pytest test_pipeline_v2.py -v
```

### Check Syntax
```bash
.venv/bin/python -m py_compile main_chapters_v2.py
```

---

**Need help?** Check the logs first, then consult the full guide.
