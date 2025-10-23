# Boundary-Based Recipe Extraction Strategy

## Core Philosophy

**The recipe links ARE the boundaries** - not just metadata for validation.

## Two-Phase Approach

### Phase 1: Discover Recipe Boundaries from EPUB Structure

**Goal**: Extract list of (href, fragment) pairs that point to recipe locations

**Method**:
1. Find recipe list page (high link density, diverse targets)
2. Extract all links from that page
3. Minimal structural filtering (remove page numbers, duplicates only)
4. Optional LLM validation to remove obvious non-recipes
5. Sort by EPUB spine order for sequential boundary detection

**Output**: List of `RecipeBoundary` objects
```python
[
  RecipeBoundary(href="chapter_001.html", fragment=None, link_text="Hummus"),
  RecipeBoundary(href="chapter_002.html", fragment=None, link_text="Falafel"),
  RecipeBoundary(href="part0005_split_020.html", fragment="rec59", link_text="Pancakes"),
]
```

**Key insight**: Don't worry about exact counts. Overinclusive is OK because:
- Real recipes will have ingredients/instructions
- False positives (section headers, TOC) will fail at extraction phase

### Phase 2: Extract Content at Each Boundary

**Goal**: Get the exact recipe content using boundary information

**Method**:
```python
for i, boundary in enumerate(sorted_boundaries):
    next_boundary = sorted_boundaries[i+1] if i+1 < len(sorted_boundaries) else None

    # Determine extraction mode
    if boundary.fragment:
        # Mode A: Fragment-based (Modern Way, Simple, etc.)
        # Extract from <element id="rec59"> to <element id="rec60">
        content = extract_fragment_section(
            file=boundary.href,
            start_fragment=boundary.fragment,
            end_fragment=next_boundary.fragment if next_boundary else None
        )
    else:
        # Mode B: Whole file (Jerusalem style)
        # Entire file = one recipe
        content = extract_whole_file(boundary.href)

    # Convert to markdown
    markdown = html_to_markdown(content)

    # Parse with LLM
    recipe = RecipeParser(markdown).parse()

    # Natural validation
    if recipe.ingredients and recipe.instructions:
        yield recipe
    else:
        # Boundary pointed to non-recipe (section header, TOC, etc.)
        # Silently skip
        continue
```

## Why This Is Robust

**Handles all EPUB structures**:
- ✅ Dedicated recipe list pages (Jerusalem, Completely Perfect)
- ✅ Recipe lists with fragments (Modern Way, Simple)
- ✅ Category-prefixed TOCs (Planted: "almonds: almond butter")
- ✅ Books without structured TOC (fallback: use spine chapters)

**Natural validation**:
- No fighting LLM prompts for exact classification
- Recipes self-validate by having ingredients/instructions
- Over-extraction in Phase 1 is harmless (filtered in Phase 2)

**Perfect boundaries**:
- Using href + fragment gives exact content sections
- Next recipe's boundary = current recipe's endpoint
- No overlap, no missing content

## Test Strategy

### Phase 1 Tests (Boundary Discovery)
```python
def test_jerusalem_finds_125_boundaries():
    boundaries = discover_boundaries("jerusalem.epub")
    assert 120 <= len(boundaries) <= 130  # Approximate, not exact
    assert all(b.href for b in boundaries)

def test_modern_way_boundaries_have_fragments():
    boundaries = discover_boundaries("a-modern-way-to-eat.epub")
    assert most_have_fragments(boundaries)  # Should have #rec59, etc.
```

### Phase 2 Tests (Content Extraction)
```python
def test_extract_whole_file_jerusalem():
    boundary = RecipeBoundary(href="chapter_001u.html", fragment=None)
    content = extract_at_boundary(book, boundary, next_boundary=None)
    assert "A'ja" in content
    assert "bread fritters" in content

def test_extract_fragment_modern_way():
    b1 = RecipeBoundary(href="part0005.html", fragment="rec59")
    b2 = RecipeBoundary(href="part0005.html", fragment="rec60")
    content = extract_at_boundary(book, b1, b2)
    assert len(content) < 5000  # Just one recipe
```

### End-to-End Tests
```python
def test_jerusalem_end_to_end():
    recipes = extract_all_recipes("jerusalem.epub")
    assert 120 <= len(recipes) <= 130
    assert all(r.ingredients for r in recipes)
    assert all(r.instructions for r in recipes)
```

## Implementation Order

1. ✅ StructuredListExtractor (done, needs cleanup)
2. ❌ `sort_by_spine_order()` - NEXT
3. ❌ RecipeContentExtractor - CRITICAL
4. ❌ End-to-end pipeline
5. ❌ Generate expected lists from extraction

## Success Criteria

- Extract 120-130 recipes from Jerusalem (±5 from 125)
- Extract 140-150 from Modern Way (±5 from 142)
- All recipes have: title, ingredients, instructions
- No content overlap between consecutive recipes
- Works on books WITH and WITHOUT structured TOC
