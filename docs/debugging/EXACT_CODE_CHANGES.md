# Exact Code Changes - Before and After

## File: `/Volumes/george/Developer/mela_parser/main_chapters_v2.py`

---

## Change 1: Added TOC Detection Method

### Location: Lines 405-470 (NEW)
```python
def _is_table_of_contents(self, chapter_name: str, chapter_md: str) -> bool:
    """
    Detect if a chapter is a Table of Contents / Recipe List.

    A TOC chapter typically has:
    - High density of links (many links relative to text)
    - Short text content overall
    - Filenames like 'listofrecipes', 'contents', 'toc', 'index'
    - Few actual recipe sections (no ingredients/instructions blocks)

    Args:
        chapter_name: Chapter filename
        chapter_md: Chapter markdown content

    Returns:
        True if this appears to be a TOC chapter
    """
    # Check filename patterns
    toc_patterns = [
        r'list.*recipe',
        r'contents?',
        r'toc',
        r'index',
        r'table.*content',
    ]
    for pattern in toc_patterns:
        if re.search(pattern, chapter_name, re.IGNORECASE):
            logging.debug(f"Detected TOC chapter by filename pattern: {chapter_name}")
            return True

    # Check link density
    link_pattern = r'\[([^\]]+)\]\([^)]+\)'
    links = re.findall(link_pattern, chapter_md)
    link_count = len(links)

    # Count estimated text lines (rough heuristic)
    lines = [l for l in chapter_md.split('\n') if l.strip() and not l.startswith('[')]
    text_lines = len(lines)

    # If more than 50% of content lines are links, likely a TOC
    if link_count > 20 and link_count > text_lines * 0.5:
        logging.debug(
            f"Detected TOC chapter by link density: {chapter_name} "
            f"({link_count} links in {text_lines} text lines)"
        )
        return True

    # Check for ingredient/instruction patterns (recipe chapters have these)
    recipe_indicators = [
        r'ingredients?:',
        r'instructions?:',
        r'\d+\s*(cups?|tbsp|tsp|grams?|ml)',  # Measurements
        r'combine|mix|stir|blend|bake|cook|heat|season',  # Cooking verbs
    ]
    has_recipe_content = any(
        re.search(pattern, chapter_md, re.IGNORECASE)
        for pattern in recipe_indicators
    )

    if not has_recipe_content and link_count > 10:
        logging.debug(
            f"Detected TOC chapter by lack of recipe content: {chapter_name}"
        )
        return True

    return False
```

---

## Change 2: Added Recipe Completeness Validation Method

### Location: Lines 472-506 (NEW)
```python
def _is_complete_recipe(self, recipe: MelaRecipe) -> bool:
    """
    Check if a recipe has the minimum required fields to be valid.

    A complete recipe must have:
    - A non-empty title
    - A non-empty ingredients section
    - A non-empty instructions section

    Args:
        recipe: Recipe object to validate

    Returns:
        True if recipe is complete
    """
    # Convert to dict if needed
    if hasattr(recipe, 'model_dump'):
        recipe_dict = recipe.model_dump()
    elif hasattr(recipe, '__dict__'):
        recipe_dict = recipe.__dict__
    else:
        recipe_dict = dict(recipe) if isinstance(recipe, dict) else {}

    title = recipe_dict.get('title', '').strip() if isinstance(recipe_dict.get('title'), str) else ''
    ingredients = recipe_dict.get('ingredients', '').strip() if isinstance(recipe_dict.get('ingredients'), str) else ''
    instructions = recipe_dict.get('instructions', '').strip() if isinstance(recipe_dict.get('instructions'), str) else ''

    is_complete = bool(title and ingredients and instructions)
    if not is_complete:
        title_for_log = recipe_dict.get('title', 'UNKNOWN')
        logging.debug(
            f"Incomplete recipe filtered: {title_for_log} "
            f"(title={bool(title)}, ingredients={bool(ingredients)}, instructions={bool(instructions)})"
        )
    return is_complete
```

---

## Change 3: Refactored Extract Recipes Pipeline

### Location: Lines 539-620 (MODIFIED)

#### BEFORE:
```python
# Phase 3: Extract from each chapter in parallel
logging.info(f"Phase 3: Extracting recipes from {len(chapters)} chapters")

extractor = ChapterExtractor(self.async_client, model=model)
semaphore = asyncio.Semaphore(self.max_concurrent_chapters)

tasks = [
    extractor.extract_from_chapter(
        chapter_md,
        chapter_name,
        expected_titles,
        prompts.extraction_prompt,
        semaphore,
    )
    for chapter_name, chapter_md in chapters  # <-- ALL chapters, including TOC!
]

results = await asyncio.gather(*tasks, return_exceptions=True)

# Collect recipes and handle errors
all_recipes = []
chapters_with_recipes = 0

for i, result in enumerate(results):
    if isinstance(result, Exception):
        logging.error(f"Chapter {i} failed with exception: {result}")
        continue

    if result:
        all_recipes.extend(result)
        chapters_with_recipes += 1
        logging.info(f"Chapter {i} ({chapters[i][0]}): extracted {len(result)} recipes")

# Phase 4: Simple deduplication
logging.info("Phase 4: Deduplication")
seen = set()
unique_recipes = []
duplicates = 0

for recipe in all_recipes:  # <-- Including incomplete TOC recipes!
    if recipe.title not in seen:
        seen.add(recipe.title)
        unique_recipes.append(recipe)
    else:
        duplicates += 1
        logging.debug(f"Duplicate removed: {recipe.title}")
```

#### AFTER:
```python
# Phase 3: Filter out TOC chapters and extract from content chapters
logging.info(f"Phase 3a: Filtering Table of Contents chapters from {len(chapters)} chapters")
content_chapters = []
skipped_toc_chapters = []

for chapter_name, chapter_md in chapters:
    if self._is_table_of_contents(chapter_name, chapter_md):
        logging.info(f"Skipping TOC chapter: {chapter_name}")
        skipped_toc_chapters.append(chapter_name)
    else:
        content_chapters.append((chapter_name, chapter_md))

logging.info(
    f"Proceeding with extraction from {len(content_chapters)} content chapters "
    f"(skipped {len(skipped_toc_chapters)} TOC chapters)"
)

# Phase 3b: Extract recipes from content chapters in parallel
logging.info(f"Phase 3b: Extracting recipes from {len(content_chapters)} content chapters")

extractor = ChapterExtractor(self.async_client, model=model)
semaphore = asyncio.Semaphore(self.max_concurrent_chapters)

tasks = [
    extractor.extract_from_chapter(
        chapter_md,
        chapter_name,
        expected_titles,
        prompts.extraction_prompt,
        semaphore,
    )
    for chapter_name, chapter_md in content_chapters  # <-- Only content chapters!
]

results = await asyncio.gather(*tasks, return_exceptions=True)

# Collect recipes and handle errors
all_recipes = []
chapters_with_recipes = 0

for i, result in enumerate(results):
    if isinstance(result, Exception):
        logging.error(f"Content chapter {i} failed with exception: {result}")
        continue

    if result:
        all_recipes.extend(result)
        chapters_with_recipes += 1
        logging.info(f"Content chapter {i} ({content_chapters[i][0]}): extracted {len(result)} recipes")

# Phase 4: Filter incomplete recipes
logging.info("Phase 4: Filtering incomplete recipes")
incomplete_count = 0
complete_recipes = []

for recipe in all_recipes:
    if self._is_complete_recipe(recipe):
        complete_recipes.append(recipe)
    else:
        incomplete_count += 1
        title = recipe.title if hasattr(recipe, 'title') else dict(recipe).get('title', 'UNKNOWN')
        logging.info(f"Filtering incomplete recipe: {title}")

logging.info(
    f"Filtered {incomplete_count} incomplete recipes; "
    f"{len(complete_recipes)} complete recipes remain"
)

# Phase 5: Deduplication
logging.info("Phase 5: Deduplication")
seen = set()
unique_recipes = []
duplicates = 0

for recipe in complete_recipes:  # <-- Only complete recipes!
    recipe_title = recipe.title if hasattr(recipe, 'title') else dict(recipe).get('title', '')
    if recipe_title not in seen:
        seen.add(recipe_title)
        unique_recipes.append(recipe)
    else:
        duplicates += 1
        logging.debug(f"Duplicate removed: {recipe_title}")
```

---

## Summary of Changes

### Lines Modified: ~80 lines
### Lines Added: ~200 lines (new methods + pipeline changes)
### Functions Added: 2 new methods
### Phases Added: 1 new filtering phase (3a) + 1 new validation phase (4)

### Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| TOC Detection | None | Implemented with 3-level detection |
| TOC Handling | Extract recipes from TOC | Skip TOC entirely |
| Incomplete Recipe Handling | Pass through to write | Filter before deduplication |
| Validation Layers | 1 (at write time) | 3 (detection + filtering + write time) |
| API Calls Wasted | Yes (TOC extraction) | No (TOC skipped) |
| Final Recipe Count | 0 (incomplete filtered) | >0 (actual recipes) |

---

## Testing the Changes

### Verification Commands
```bash
# 1. Check syntax
python3 -m py_compile main_chapters_v2.py

# 2. Run with debug logging to see TOC detection
python3 main_chapters_v2.py --debug

# 3. Verify expected log messages
grep "Skipping TOC chapter" output.log
grep "Phase 3a" output.log
grep "Filtering incomplete" output.log
```

### Expected Log Output
```
[INFO] Phase 3a: Filtering Table of Contents chapters from 163 chapters
[INFO] Skipping TOC chapter: pages/listofrecipes.html
[INFO] Proceeding with extraction from 162 content chapters (skipped 1 TOC chapters)
[INFO] Phase 3b: Extracting recipes from 162 content chapters
[INFO] Phase 4: Filtering incomplete recipes
[INFO] Filtered 0 incomplete recipes; X complete recipes remain
[INFO] Phase 5: Deduplication
```

---

## Backward Compatibility

- ✓ No breaking changes to public API
- ✓ Existing code paths still work
- ✓ Only internal implementation changed
- ✓ Results are more accurate (false positives filtered)
- ✓ No new dependencies added

---

## File Location for Reference

**Main file with changes:** `/Volumes/george/Developer/mela_parser/main_chapters_v2.py`

**Line numbers for exact locations:**
- TOC Detection: 405-470
- Recipe Validation: 472-506
- Pipeline Refactor: 539-620
