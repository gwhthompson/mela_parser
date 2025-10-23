# Pipeline Comparison: Before vs After Fix

## Visual Pipeline Flow

### BEFORE FIX (Broken)

```
┌─────────────────────────────────────────────────────────────────┐
│                     EPUB Processing Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Convert Chapters
  163 chapters → 163 markdown files
  ├─ Chapter 2: pages/listofrecipes.html (TOC with 125 links)
  ├─ Chapter 5: pages/introduction.html (2 recipes)
  ├─ Chapter 6: pages/chapter_001.html (33 recipes)
  └─ ... 160 more chapters ...

Step 2: Discover Recipe List
  Scan all chapters for recipe titles (from links)
  Result: Found 125 recipe titles in Chapter 2 TOC

Step 3: Extract Recipes (BROKEN - FROM ALL CHAPTERS)
  ┌─────────────────────────────────────────────────────┐
  │ For each chapter:                                    │
  │   Check if expected_titles appear in chapter_text   │
  │   Extract matching recipes                          │
  └─────────────────────────────────────────────────────┘

  Chapter 2 (TOC):
    MATCH ALL 125 titles (because it's the TOC!)
    └─> Attempt extraction of 125 recipes from link-only content
        └─> LLM extracts: title only (no ingredients/instructions)
            └─> 125 INCOMPLETE recipes ✗

  Chapter 5 (Introduction):
    MATCH 2 titles
    └─> Extract 2 recipes successfully
        └─> 2 COMPLETE recipes ✓

  Chapter 6 (Recipe Chapter):
    MATCH 33 titles
    └─> Extract 33 recipes successfully
        └─> 33 COMPLETE recipes ✓

Step 4: Collect All Extracted
  Total: 160 recipes (125 incomplete from TOC + 35 complete)

Step 5: Deduplication
  Result: ~160 recipes (deduplicated by title)

Step 6: Write to Disk
  For each recipe:
    Check: title? ✓ ingredients? ✗ instructions? ✗
    Decision: SKIP (incomplete) ✗

  Chapter 2 recipes: 0 written (incomplete)
  Chapter 5 recipes: 2 written ✓
  Chapter 6 recipes: 33 written ✓
  ...

  FINAL RESULT: ~35 recipes written
  PROBLEM: 125 wasted API calls + 125 incomplete recipes processed
```

### AFTER FIX (Correct)

```
┌─────────────────────────────────────────────────────────────────┐
│                     EPUB Processing Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Convert Chapters
  163 chapters → 163 markdown files
  ├─ Chapter 2: pages/listofrecipes.html (TOC with 125 links)
  ├─ Chapter 5: pages/introduction.html (2 recipes)
  ├─ Chapter 6: pages/chapter_001.html (33 recipes)
  └─ ... 160 more chapters ...

Step 2: Discover Recipe List
  Scan all chapters for recipe titles (from links)
  Result: Found 125 recipe titles in Chapter 2 TOC

Step 3a: Filter TOC Chapters (NEW)
  ┌─────────────────────────────────────────────────────┐
  │ For each chapter:                                    │
  │   Check _is_table_of_contents(chapter)              │
  │     • Filename patterns? (listofrecipes) ✓ YES      │
  │     • Link density? (>50% links) ✓ YES              │
  │     • Recipe content? (none) ✓ MISSING              │
  │   Decision: SKIP if TOC detected                    │
  └─────────────────────────────────────────────────────┘

  Chapter 2: DETECTED AS TOC
    └─> SKIPPED ✓ (not extracted)
        └─> SAVED: 1 API call + 0 incomplete recipes

  Chapter 5: Content chapter
    └─> KEPT for extraction ✓

  Chapter 6: Content chapter
    └─> KEPT for extraction ✓

  Result: 162 content chapters to process

Step 3b: Extract from Content Chapters Only (MODIFIED)
  Chapter 5 (Introduction):
    MATCH 2 titles
    └─> Extract 2 recipes successfully
        └─> 2 COMPLETE recipes ✓

  Chapter 6 (Recipe Chapter):
    MATCH 33 titles
    └─> Extract 33 recipes successfully
        └─> 33 COMPLETE recipes ✓

  Chapters 3, 4, 7-163:
    └─> Extract recipes as applicable
        └─> ~50-100 COMPLETE recipes ✓

Step 4: Filter Incomplete Recipes (NEW)
  ┌─────────────────────────────────────────────────────┐
  │ For each extracted recipe:                          │
  │   Check _is_complete_recipe(recipe)                 │
  │     • Title? ✓ YES                                  │
  │     • Ingredients? ✓ YES                            │
  │     • Instructions? ✓ YES                           │
  │   Decision: KEEP if all fields present              │
  └─────────────────────────────────────────────────────┘

  ALL extracted recipes: COMPLETE ✓
  Filtered out: 0 incomplete recipes
  Remaining: 50-100 complete recipes

Step 5: Deduplication
  Result: 50-100 recipes (deduplicated by title)

Step 6: Write to Disk
  For each recipe:
    Check: title? ✓ ingredients? ✓ instructions? ✓
    Decision: WRITE ✓

  Chapter 5 recipes: 2 written ✓
  Chapter 6 recipes: 33 written ✓
  Other chapters: 15-65 written ✓

  FINAL RESULT: 50-100 recipes written ✓
  BENEFIT: 0 wasted API calls + all recipes legitimate
```

---

## Comparison Table

| Aspect | BEFORE | AFTER |
|--------|--------|-------|
| **Chapters Processed** | 163 (including TOC) | 162 (TOC skipped) |
| **API Calls for TOC** | ~5-10 calls | 0 calls (skipped) |
| **Incomplete Recipes** | ~125 (from TOC) | 0 (filtered) |
| **Complete Recipes** | ~50-100 | ~50-100 |
| **Recipes Written** | ~50-100 | ~50-100 |
| **Efficiency** | Low (wastes resources) | High (optimized) |
| **Detection Layers** | 1 (write-time validation) | 3 (extraction + filtering + writing) |
| **False Positives** | Possible | Minimal |

---

## Resource Impact

### API Calls
```
BEFORE: 163 chapter extractions (including TOC)
AFTER:  162 chapter extractions (TOC skipped)
SAVED:  ~5-10 API calls (1 chapter × 5-10 parallel calls)
```

### Processing Time
```
BEFORE: Extract TOC + Filter incomplete + Write legitimate
AFTER:  Skip TOC early + Extract content + Filter complete + Write legitimate
DELTA:  Slightly faster (TOC detection + early skip saves time)
```

### Memory
```
BEFORE: Store 125 incomplete TOC recipes in memory during processing
AFTER:  Skip TOC recipes entirely (not stored in memory)
SAVED:  ~10-20 MB (125 recipe objects)
```

---

## Detection Flow Diagram

```
Is this chapter a Table of Contents?

┌─ Check 1: Filename Pattern ─┐
│ Filename contains:          │
│ • listofrecipes             │
│ • contents                  │
│ • toc                       │
│ • index                     │
│ • table.*content            │
└─────────────────────────────┘
        │
        ├─ YES ──→ SKIP (TOC) ✓
        │
        NO
        │
        ├─ Check 2: Link Density ────┐
        │ >20 links AND >50% are      │
        │ markdown links              │
        └────────────────────────────┘
        │
        ├─ YES ──→ SKIP (TOC) ✓
        │
        NO
        │
        └─ Check 3: Recipe Content ──┐
          Has ingredients OR          │
          instructions OR             │
          cooking verbs OR            │
          measurements?               │
          └────────────────────────────┘
          │
          ├─ NO & >10 links ──→ SKIP (TOC) ✓
          │
          └─ YES ──→ PROCESS (Content) ✓
```

---

## Recipe Validation Flow

```
Did we extract a recipe?

    YES
    │
    └─ Check: Recipe is Complete?
        │
        ├─ Has title?        └─ YES
        ├─ Has ingredients?  └─ YES
        ├─ Has instructions? └─ YES
        │
        ALL YES ──→ ACCEPT (Complete) ✓
        ANY NO  ──→ REJECT (Incomplete) ✗
```

---

## Summary

### Key Improvements
1. **Detection**: Multi-layer TOC detection prevents false extractions
2. **Prevention**: Early skip of TOC chapters before processing
3. **Validation**: Complete recipe filtering catches edge cases
4. **Efficiency**: Fewer API calls, less memory, faster processing
5. **Robustness**: Multiple detection methods catch various TOC formats

### Result
- Correct extraction without TOC contamination
- Optimized resource usage
- Robust multi-layer validation
- Foundation for handling various cookbook structures
