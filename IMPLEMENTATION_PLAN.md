# Clean Implementation Plan - Chapter-Based Extraction

## Current Status

**Branch**: `feature/markitdown-single-pass`
**Problem**: 157 recipes extracted vs 125 expected (32 duplicates, wrong images, title modifications)
**Solution**: Chapter-based extraction with recipe list discovery

---

## What's Been Tried (For Reference Only)

1. ❌ Single-pass MarkItDown (0-19 recipes, failed)
2. ❌ Delimiter markers (40% success, failed)
3. ❌ Overlapping chunks 50% (96% success, but 32 duplicates)
4. ❌ Overlapping chunks 75% (100% success, still duplicates)
5. → **Chapter-based** (implementing now)

**All experimental files kept in branch for reference, won't be used.**

---

## THE SOLUTION: Chapter-Based Extraction

### Architecture

```
Pass 0: Discover recipe list from book
  ↓
Pass 1: Split EPUB by chapters (MarkItDown per chapter)
  ↓
Pass 2: Extract from each chapter (guided by list, temperature=0)
  ↓
Pass 3: Simple deduplication (minimal needed)
  ↓
Pass 4: Validate against expected list
```

---

## Implementation (Clean, Focused)

### File: `main_chapters.py` (Production)

**What it does**:
1. Converts each EPUB chapter to markdown (MarkItDown)
2. Discovers recipe list by finding link patterns
3. Extracts from each chapter using GPT-5-nano
4. Validates against recipe list

**Why it's better**:
- Natural boundaries (chapters don't split recipes)
- No overlap needed (no duplicates!)
- Recipe list guides extraction (exact titles)
- Cheaper & faster

---

## Testing Plan (TDD)

### Test File: `test_recipe_lists.py`

**Ground truth**: `examples/output/recipe-lists/*.txt`
- Jerusalem: 125 recipes
- Modern Way: 142 recipes
- Completely Perfect: 122 recipes
- Simple: 140 recipes

**Tests**:
1. `test_exact_count()` - Must extract exact number
2. `test_no_duplicates()` - No duplicate recipes
3. `test_titles_match()` - Titles match list exactly
4. `test_all_books()` - All 4 books pass

---

## Next Steps (Clean & Focused)

1. **Fix `main_chapters.py`** - Currently extracts 0 recipes
   - Debug why extraction failing
   - Ensure temperature settings correct
   - Validate recipe list discovery works

2. **Run on Jerusalem**
   - Should get exactly 125 recipes
   - Validate against recipe list
   - Check for duplicates (should be 0)

3. **Test on all 4 books**
   - Validate counts match
   - Commit final solution

4. **Clean up branch**
   - Keep only: main_chapters.py, tests, docs
   - Archive experimental files
   - Merge to master

---

## Files to Keep (Production)

- `main_chapters.py` - Chapter-based extraction ⭐
- `test_recipe_lists.py` - TDD validation
- `converter.py` - MarkItDown helpers
- `parse.py` - Parsers
- `recipe.py` - Recipe processing
- `image_processor.py` - Image handling
- `pyproject.toml` - Dependencies

## Files to Archive (Experiments)

- main_v2.py - Single-pass (failed)
- main_delimiter.py - Markers (failed)
- main_overlap.py - Overlapping (works but duplicates)
- main_parallel.py - Parallel overlap (works but duplicates)
- test_extraction.py - Model comparison
- compare_results.py - Analysis tool

---

## Success Criteria

✅ Jerusalem: Exactly 125 recipes
✅ No duplicates
✅ Titles match list exactly
✅ All 4 books pass tests
✅ Clean, maintainable code

---

## Current Focus

**ONLY working on**: `main_chapters.py`
**Test against**: Jerusalem (125 recipes)
**Goal**: Get it working correctly, then expand to other books

**Status**: Debugging chapter extraction (currently gets 0 recipes)
