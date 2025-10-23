# Mela Parser - Project Status & Roadmap

**Date**: 2025-10-22
**Status**: Restructured, boundary-based extraction in progress

---

## Current State

### ✅ Completed: Project Modernization

**Restructured to modern Python standards:**
```
mela_parser/
├── src/mela_parser/          # Core package (was: root dir mess)
├── tests/                    # Test suite (was: mixed with source)
├── scripts/                  # Experimental scripts (preserved)
├── docs/                     # Organized documentation
├── examples/                 # Test cookbooks & outputs
├── debug/                    # EPUB structure inspection tool
├── pyproject.toml            # Modern build system
├── Makefile                  # Dev task automation
└── LICENSE                   # MIT license
```

**Benefits:**
- Clean imports: `from mela_parser import RecipeParser`
- Pip installable: `uv sync`
- Proper testing: `make test`
- Professional structure following PEP 518/621

### ✅ Completed: Debug Analysis

**Created comprehensive EPUB inspection tool** (`debug/debug.py`):
- Analyzes EPUB structure (TOC, spine, navigation, links)
- Identified that recipe list pages exist with exact recipe→chapter mappings
- Discovered different TOC formats across books

**Key findings:**
- Jerusalem: Dedicated `listofrecipes.html` with 125 links → chapters
- Modern Way: `part0016.html` with 143 links using #rec fragments
- Completely Perfect: `nav.xhtml` with 609 links (includes page numbers, sections)
- Simple: `nav.xhtml` with 337 links
- Saffron Tales: `nav.html` with 106 links
- Planted: `part0010.html` with 402 links (uses "category: recipe" format)

### ✅ Completed: Structural Recipe Link Extraction

**Implemented** (`src/mela_parser/extractors/structured_list.py`):
- `find_recipe_list_pages()` - Uses scoring (link density, target uniqueness)
- `extract_all_links_from_book()` - Parses HTML links with BeautifulSoup
- `apply_structural_filters()` - Removes page numbers, duplicates only
- `validate_with_llm()` - Single API call validation (partially working)

**Results on all 6 books:**
| Book | Expected | Extracted | Status |
|------|----------|-----------|--------|
| Jerusalem | 125 | 125 | ✅ 100% recall, 0 extras |
| Modern Way | 142 | 143 | ✅ 100% recall, 1 extra |
| Completely Perfect | 122 | 141 | ✅ 100% recall, 19 extras |
| Simple | 140 | 158 | ✅ 100% recall, 18 extras |
| Saffron Tales | 81 | 106 | ✅ 100% recall, 25 extras |
| Planted | 200 | 339 | ⚠️ Different TOC structure |

**Key achievement**: **100% recall** - we find ALL expected recipes!

---

## ⚠️ Current Challenge: LLM Validation Instability

**Problem**: Getting LLM to filter extras to exact counts is unreliable
- Jerusalem: 125→113→125 (varies with prompt changes)
- Requires extensive prompt engineering
- Results not consistent across runs

**Decision**: Stop fighting for exact LLM counts

**Why**: The extras (1-25 per book) will be naturally filtered when we try to extract their content:
- Link to "Introduction" → no ingredients → skipped
- Link to "1Breakfast" section → no ingredients → skipped
- Only real recipes with ingredients/instructions survive

---

## 🎯 New Focus: Boundary-Based Content Extraction

### The Critical Missing Piece

We have the recipe boundaries (hrefs + fragments), but we haven't implemented **extracting content AT those boundaries**.

### What We Need to Build

**RecipeContentExtractor** (new class):

```python
class RecipeContentExtractor:
    """Extract recipe content using href boundaries from EPUB"""

    def extract_at_boundary(
        self,
        book: EpubBook,
        current: RecipeBoundary,
        next_boundary: RecipeBoundary = None
    ) -> str:
        """
        Extract content from current boundary to next boundary.

        For whole-file recipes (Jerusalem):
          - Extract entire chapter_001u.html file
          - Convert to markdown
          - That's the complete recipe

        For fragment-based recipes (Modern Way):
          - Extract from <element id="rec59">
          - To <element id="rec60"> (or end of file)
          - Convert just that section to markdown
          - That's the complete recipe

        Returns: Markdown content for LLM parsing
        """
```

**Why boundaries matter:**
- Using next boundary as endpoint prevents:
  - Getting whole chapter when recipe is just one section
  - Missing parts of recipe that span multiple elements
  - Including multiple recipes in one extraction
- Perfect content isolation for LLM parsing

### Implementation Plan

**Step 1: Implement `sort_by_spine_order()`**
- Uses EPUB spine to sort boundaries by book order
- Critical for determining "next boundary" correctly

**Step 2: Implement content extraction**
- `extract_whole_file()` - For Jerusalem style
- `extract_fragment_section()` - For Modern Way style
- `extract_at_boundary()` - Smart dispatch

**Step 3: End-to-end pipeline**
- Boundaries → Content → Parse → Validate → Save
- Test on Jerusalem first (simplest)
- Then Modern Way (fragments)

**Step 4: Generate new expected lists**
- Run successful extraction on all books
- Save results as ground truth for regression testing

---

## Roadmap

### Immediate (Next 2-4 hours)

- [ ] Implement `sort_by_spine_order()` with TDD
- [ ] Create `RecipeContentExtractor` class
- [ ] Implement whole-file extraction (Jerusalem case)
- [ ] Implement fragment extraction (Modern Way case)
- [ ] Test end-to-end on Jerusalem

### Short-term (Next session)

- [ ] Test on all 6 books
- [ ] Handle edge cases (overlapping fragments, missing boundaries)
- [ ] Generate new expected lists from extraction results
- [ ] Update all tests to use new baselines

### Future Enhancements

- [ ] Fallback for books without TOC (use spine chapters directly)
- [ ] Title normalization for category-prefixed TOCs (Planted)
- [ ] Parallel extraction with asyncio
- [ ] Cache extracted content

---

## Files to Focus On

**Active development:**
- `src/mela_parser/extractors/structured_list.py` - Boundary discovery (mostly done)
- `src/mela_parser/extractors/recipe_content.py` - Content extraction (TO IMPLEMENT)
- `tests/test_boundary_extraction.py` - New focused test suite (TO CREATE)

**Keep for reference:**
- `scripts/main_*.py` - Experimental approaches
- `debug/` - EPUB structure analysis
- `docs/` - Historical debugging notes

**Ignore:**
- Old test files (test_pipeline_v2.py, etc.) - from previous approaches

---

## Key Decisions

1. **Recipe links define boundaries** - not just validation metadata
2. **Overinclusive extraction is OK** - natural validation filters
3. **Exact LLM classification not critical** - focus on boundary correctness
4. **Expected lists should be generated** - not manually curated
5. **Test for approximate counts** - ±5 recipes is acceptable

---

## Questions Still Open

1. Should we regenerate ALL expected lists from new extraction?
2. How to handle Planted's "category: recipe" format?
3. Should we keep LLM validation or skip it entirely?
4. What's the minimum viable fallback for books without TOC?
