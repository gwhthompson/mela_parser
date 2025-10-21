# PROJECT STATUS - Single Source of Truth

## GOAL
Extract exactly 125 recipes from Jerusalem EPUB matching the official recipe list.

---

## THE SOLUTION
**Chapter-based extraction** using EPUB's natural structure

---

## WHY PREVIOUS APPROACHES FAILED

**Overlapping chunks** (current):
- Extracts 157 recipes (32 duplicates)
- Same recipe extracted 2-3x from overlap
- Model adds "(partial)", "(duplicate)" to titles
- All images wrong (heuristic failed)

**Root cause**: Arbitrary chunks ignore book structure, create artificial duplicates.

---

## THE FINAL APPROACH

### Phase 0: Discover Recipe List
- Scan book for link patterns `[Recipe](link.html)`
- Extract from index/TOC/recipe list sections
- Use GPT-5-mini to clean and deduplicate
- Result: Official list of 125 exact titles

### Phase 1: Split by Chapters
- Use EPUB's native chapter structure (163 chapters)
- Convert each chapter with MarkItDown
- Natural boundaries = no recipe splits
- No overlap needed = no duplicates!

### Phase 2: Extract from Chapters
- For each chapter, extract recipes
- Guided by discovered list (know exact titles)
- Temperature=0 (deterministic, no modifications)
- GPT-5-nano for cost efficiency

### Phase 3: Validate
- Compare extracted vs expected (125)
- Report missing/extra
- Ensure exact title matching

---

## CURRENT STATUS

**File**: `main_chapters.py`
**Issue**: Extracting 0 recipes (debugging needed)
**Next**: Fix extraction, test on Jerusalem

**Target**: Exactly 125 recipes matching `examples/output/recipe-lists/jerusalem-recipe-list.txt`

---

## IMPLEMENTATION CHECKLIST

- [x] Chapter splitter with MarkItDown
- [x] Recipe list discovery
- [x] Guided extraction logic
- [ ] Fix: Currently extracts 0 (debug needed)
- [ ] Test: Get 125 recipes from Jerusalem
- [ ] Validate: Match official list
- [ ] Images: AI-powered unique assignment
- [ ] Test: All 4 books

---

## FILES

**Production** (keep):
- `main_chapters.py` - Final solution
- `test_recipe_lists.py` - TDD tests
- Core modules (parse.py, recipe.py, converter.py)

**Experimental** (archive):
- main_v2.py, main_overlap.py, main_delimiter.py, main_parallel.py
- Various test logs and outputs

---

## SUCCESS =
Jerusalem extracts exactly 125 recipes with correct titles and images.
