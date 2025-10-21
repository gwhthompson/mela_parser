# YOUR FOCUSED IMPLEMENTATION PLAN

## GOAL
Extract exactly 125 recipes from Jerusalem (and correct counts from all books) with zero duplicates and correct images.

---

## THE SOLUTION (Your Idea)

**Chapter-based extraction** with recipe list discovery

### Why This Works
- EPUB chapters = natural boundaries (recipes never split)
- No overlap needed = zero duplicates
- Recipe list from book = know exact titles
- Guided extraction = no modifications

---

## IMPLEMENTATION STEPS

### Step 1: Fix `main_chapters.py` (Currently Broken)
**Issue**: Extracts 0 recipes
**Fix**: Debug and get basic extraction working

### Step 2: Test on Jerusalem
**Target**: Exactly 125 recipes
**Validate**: Against `examples/output/recipe-lists/jerusalem-recipe-list.txt`

### Step 3: Add Image Processing
**Strategy**: One image per recipe (AI verification)
**Method**: Extract all book images, assign best match to each recipe

### Step 4: Test All 4 Books
- Jerusalem: 125
- Modern Way: 142
- Completely Perfect: 122
- Simple: 140

### Step 5: Clean Up & Merge
**Keep**: main_chapters.py
**Archive**: All experimental files (main_v2, main_overlap, etc.)
**Merge**: To master

---

## KEY REQUIREMENTS (From You)

1. ✅ Use EPUB chapter structure (not arbitrary chunks)
2. ✅ No overlap (chapters are boundaries)
3. ✅ Temperature=0 (deterministic)
4. ✅ Extract recipe list first (guide extraction)
5. ✅ MarkItDown per chapter (clean conversion)
6. ✅ Favor first occurrence (if any dups)
7. ✅ One image per recipe (unique assignment)
8. ✅ No title modifications (exact from book)
9. ✅ Match recipe list counts exactly

---

## CURRENT FOCUS

**File**: `main_chapters.py`
**Task**: Debug why it's extracting 0 recipes
**Test**: Jerusalem only (get to 125)
**Timeline**: Get it working today

Once working on Jerusalem → extend to other books → done.

---

## CLEANUP TASKS

- ✅ Killed all running tests
- ✅ Deleted log files
- ✅ Cleaned output directory
- Next: Get main_chapters.py working
- Then: Archive experimental files

---

## SUCCESS =
Jerusalem extracts exactly 125 recipes matching the list with correct images.
