# Next Steps: Implementing Boundary-Based Extraction

## Where We Are

✅ **Project restructured** - Modern Python package layout
✅ **Debug analysis complete** - Understand all 6 cookbook structures
✅ **Structural extraction working** - 100% recall on recipe links
⚠️ **LLM validation flaky** - Don't rely on exact counts

## The New Strategy

**Stop**: Fighting LLM prompts for exact recipe counts
**Start**: Using recipe links as content boundaries

## What to Build Next

### 1. Sort Boundaries by Book Order

```python
def sort_by_spine_order(links: List[RecipeLink], spine: List) -> List[RecipeLink]:
    """
    Sort recipe links by EPUB spine order (reading order).

    Critical for boundary detection:
    - Recipe N's content = from its href to Recipe N+1's href
    - Must be in correct book order!
    """
```

**Why critical**: If boundaries are out of order, we'll extract wrong content sections.

### 2. Extract Content at Boundaries

```python
class RecipeContentExtractor:
    def extract_at_boundary(
        self,
        book: EpubBook,
        current_boundary: RecipeBoundary,
        next_boundary: RecipeBoundary = None
    ) -> str:
        """
        Two modes based on boundary type:

        Mode A - Whole file (Jerusalem):
          current: href="chapter_001u.html", fragment=None
          → Extract entire file, convert to markdown

        Mode B - Fragment section (Modern Way):
          current: href="part0005.html", fragment="rec59"
          next: href="part0005.html", fragment="rec60"
          → Extract from <id="rec59"> to <id="rec60">
          → Convert just that section to markdown

        Returns: Clean markdown for one recipe only
        """
```

**Why critical**: This is the ONLY way to get exact recipe content without overlap or missing parts.

### 3. End-to-End Pipeline

```python
def extract_cookbook(epub_path: str):
    book = epub.read_epub(epub_path)

    # Phase 1: Discover boundaries
    extractor = StructuredListExtractor()
    links = extractor.extract_all_links_from_book(book)
    filtered = extractor.apply_structural_filters(links)
    # Optional: validated = extractor.validate_with_llm(filtered.candidates)
    # OR: Just use filtered.candidates directly

    boundaries = filtered.candidates  # Accept we might have extras
    sorted_boundaries = extractor.sort_by_spine_order(boundaries, book.spine)

    # Phase 2: Extract content at each boundary
    content_extractor = RecipeContentExtractor()
    for i, boundary in enumerate(sorted_boundaries):
        next_b = sorted_boundaries[i+1] if i+1 < len(sorted_boundaries) else None

        markdown = content_extractor.extract_at_boundary(book, boundary, next_b)

        # Phase 3: Parse with LLM
        recipe = RecipeParser(markdown).parse()

        # Phase 4: Natural validation
        if recipe.ingredients and recipe.instructions:
            yield recipe
        else:
            # Boundary was false positive (TOC, section header, etc.)
            # Skip silently
            pass
```

**Expected results:**
- Jerusalem: ~125 recipes (some boundaries might be section headers, get filtered)
- Modern Way: ~142 recipes
- Natural variation ±5 recipes is acceptable

## Implementation Order (TDD)

### Session 1: Sorting & Basic Extraction

1. **RED**: Write test for spine sorting
2. **GREEN**: Implement `sort_by_spine_order()`
3. **RED**: Write test for whole-file extraction
4. **GREEN**: Implement whole-file mode
5. **TEST**: Run on Jerusalem

### Session 2: Fragment Extraction

6. **RED**: Write test for fragment extraction
7. **GREEN**: Implement fragment mode
8. **TEST**: Run on Modern Way

### Session 3: Integration

9. **RED**: Write end-to-end test
10. **GREEN**: Complete pipeline
11. **RUN**: Extract all 6 books
12. **GENERATE**: New expected lists from results

## Success Criteria

- [ ] Jerusalem extracts 120-130 recipes (±5 from 125)
- [ ] All extracted recipes have: title, ingredients, instructions
- [ ] No content overlap between consecutive recipes
- [ ] Boundaries correctly isolate single recipes
- [ ] Works on both whole-file and fragment-based EPUBs

## Files to Work On

**Next implementation:**
1. `src/mela_parser/extractors/structured_list.py` - Add `sort_by_spine_order()`
2. `src/mela_parser/extractors/recipe_content.py` - New file, main work here
3. `tests/test_boundary_extraction.py` - New focused test suite

**Supporting**:
4. `docs/BOUNDARY_EXTRACTION.md` - Approach documentation (done)
5. `PROJECT_STATUS.md` - This file
6. `NEXT_STEPS.md` - You are here

## What to Ignore (For Now)

- `scripts/main_*.py` - Keep for reference, not active development
- `tests/test_pipeline_v2.py`, `test_recipe_lists.py` - Old approaches
- Expected recipe lists - Will regenerate after successful extraction
- LLM validation exact counts - Natural validation will handle this

## Open Questions for Next Session

1. **LLM validation**: Keep it (liberal filtering) or skip entirely?
2. **Expected lists**: Regenerate all or keep existing as regression baselines?
3. **Fallback**: What to do for books without structured TOC?
4. **Title normalization**: Handle "category: recipe" format in extraction or post-process?
