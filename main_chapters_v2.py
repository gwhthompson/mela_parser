#!/usr/bin/env python3
"""
Production-ready LLM-powered recipe extraction pipeline with automatic prompt iteration.

This pipeline achieves 100% match rate by:
1. Discovering recipe lists from EPUB structure
2. Extracting recipes chapter-by-chapter in parallel
3. Validating against discovered list
4. Iteratively improving prompts via LLM analysis
5. Retrying until 100% match or max iterations reached
"""
import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import ebooklib
from ebooklib import epub
from markitdown import MarkItDown
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel

from parse import CookbookRecipes, MelaRecipe
from recipe import RecipeProcessor
from image_processor import ImageProcessor


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class ExtractionError(Exception):
    """Error during recipe extraction."""
    pass


class ValidationError(Exception):
    """Error during validation."""
    pass


class PromptOptimizationError(Exception):
    """Error during prompt optimization."""
    pass


# ============================================================================
# DATA MODELS
# ============================================================================


class PromptVersion(str, Enum):
    """Prompt version identifiers."""
    DISCOVERY_V1 = "discovery_v1"
    EXTRACTION_V1 = "extraction_v1"


@dataclass
class PromptLibrary:
    """Manages prompts with versioning."""

    discovery_prompt: str
    extraction_prompt: str
    version: int = 1
    iteration_history: List[Dict] = field(default_factory=list)
    locked: bool = False

    def to_dict(self) -> Dict:
        """Export to dictionary for JSON serialization."""
        return {
            "discovery_prompt": self.discovery_prompt,
            "extraction_prompt": self.extraction_prompt,
            "version": self.version,
            "iteration_history": self.iteration_history,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptLibrary":
        """Load from dictionary."""
        return cls(
            discovery_prompt=data["discovery_prompt"],
            extraction_prompt=data["extraction_prompt"],
            version=data.get("version", 1),
            iteration_history=data.get("iteration_history", []),
            locked=data.get("locked", False),
        )

    @classmethod
    def default(cls) -> "PromptLibrary":
        """Create default prompts."""
        discovery_prompt = """Extract the unique list of recipe titles from these potential recipe lists.

Remove:
- Section headers (Contents, Index, About, etc.)
- Page numbers
- Duplicates

Keep:
- Actual recipe titles EXACTLY as written
- One entry per unique recipe

<potential_lists>
{combined_lists}
</potential_lists>"""

        extraction_prompt = """Extract ONLY these specific recipes from this chapter.
Use the EXACT titles listed below.

Expected recipes:
{expected_list}

<chapter>
{chapter_md}
</chapter>"""

        return cls(
            discovery_prompt=discovery_prompt,
            extraction_prompt=extraction_prompt,
        )


@dataclass
class ExtractionResult:
    """Results from recipe extraction."""

    recipes: List[MelaRecipe]
    total_extracted: int
    unique_count: int
    duplicates_removed: int
    chapters_processed: int
    chapters_with_recipes: int
    extraction_time: float
    model_used: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Detailed validation results."""

    extracted_titles: Set[str]
    discovered_titles: Set[str]
    matched_titles: Set[str]
    missing_titles: Set[str]
    extra_titles: Set[str]
    match_percentage: float
    total_discovered: int
    total_extracted: int
    is_perfect_match: bool

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "extracted_titles": sorted(list(self.extracted_titles)),
            "discovered_titles": sorted(list(self.discovered_titles)),
            "matched_titles": sorted(list(self.matched_titles)),
            "missing_titles": sorted(list(self.missing_titles)),
            "extra_titles": sorted(list(self.extra_titles)),
            "match_percentage": self.match_percentage,
            "total_discovered": self.total_discovered,
            "total_extracted": self.total_extracted,
            "is_perfect_match": self.is_perfect_match,
        }


@dataclass
class PromptImprovements:
    """Suggested prompt improvements from LLM analysis."""

    analysis: str
    missing_recipe_patterns: List[str]
    false_positive_patterns: List[str]
    suggested_discovery_changes: Optional[str]
    suggested_extraction_changes: Optional[str]
    confidence: float
    reasoning: str


class RecipeList(BaseModel):
    """Schema for recipe list discovery."""
    titles: List[str]


class GapAnalysis(BaseModel):
    """Schema for gap analysis results."""
    analysis: str
    missing_patterns: List[str]
    false_positive_patterns: List[str]
    discovery_improvements: Optional[str] = None
    extraction_improvements: Optional[str] = None
    confidence: float
    reasoning: str


# ============================================================================
# CHAPTER PROCESSING
# ============================================================================


class ChapterProcessor:
    """Process EPUB chapters."""

    def __init__(self):
        self.md = MarkItDown()

    def convert_epub_by_chapters(self, epub_path: str) -> Tuple[epub.EpubBook, List[Tuple[str, str]]]:
        """
        Convert each EPUB chapter to markdown using MarkItDown.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Tuple of (EpubBook, list of (chapter_name, markdown_content))
        """
        try:
            book = epub.read_epub(epub_path, {"ignore_ncx": True})
            chapters = []

            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                html_content = item.get_content()
                result = self.md.convert_stream(BytesIO(html_content), file_extension=".html")
                markdown_content = result.text_content

                chapter_name = item.get_name()
                chapters.append((chapter_name, markdown_content))

            logging.info(f"Converted {len(chapters)} chapters to markdown")
            return book, chapters

        except Exception as e:
            raise ExtractionError(f"Failed to convert EPUB chapters: {e}") from e


# ============================================================================
# RECIPE LIST DISCOVERY
# ============================================================================


class RecipeListDiscoverer:
    """Discover recipe lists from cookbook structure."""

    def __init__(self, client: OpenAI, model: str = "gpt-5-mini"):
        self.client = client
        self.model = model

    def discover_recipe_list(
        self,
        chapters: List[Tuple[str, str]],
        prompt_template: str,
    ) -> Optional[List[str]]:
        """
        Discover recipe list by finding link patterns anywhere in the book.

        Args:
            chapters: List of (chapter_name, markdown_content)
            prompt_template: Prompt template with {combined_lists} placeholder

        Returns:
            List of recipe titles or None if not found
        """
        # Collect all potential recipe list sections
        all_link_sections = []

        for chapter_name, chapter_md in chapters:
            # Find sections with many markdown links
            link_pattern = r'\[([^\]]+)\]\([^)]+\)'
            links = re.findall(link_pattern, chapter_md)

            if len(links) > 5:  # Looks like a list
                all_link_sections.append("\n".join(links))

        if not all_link_sections:
            logging.info("No recipe list sections found")
            return None

        combined = "\n\n".join(all_link_sections)
        prompt = prompt_template.format(combined_lists=combined)

        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[EasyInputMessageParam(role="user", content=prompt)],
                text_format=RecipeList,
            )

            titles = response.output_parsed.titles
            logging.info(f"Discovered {len(titles)} recipes in book's recipe list")
            return titles

        except Exception as e:
            logging.warning(f"Failed to extract recipe list: {e}")
            return None


# ============================================================================
# CHAPTER EXTRACTION
# ============================================================================


class ChapterExtractor:
    """Extract recipes from individual chapters."""

    def __init__(self, async_client: AsyncOpenAI, model: str = "gpt-5-nano"):
        self.async_client = async_client
        self.model = model

    async def extract_from_chapter(
        self,
        chapter_md: str,
        chapter_name: str,
        expected_titles: Optional[List[str]],
        prompt_template: str,
        semaphore: asyncio.Semaphore,
    ) -> List[MelaRecipe]:
        """
        Extract recipes from a single chapter with exponential backoff retry.

        Args:
            chapter_md: Chapter markdown content
            chapter_name: Chapter identifier
            expected_titles: Expected recipe titles (if known)
            prompt_template: Prompt template
            semaphore: Concurrency control

        Returns:
            List of extracted recipes
        """
        async with semaphore:
            # Build targeted prompt
            if expected_titles:
                # Find which recipes might be in this chapter
                likely_here = [t for t in expected_titles if t.lower() in chapter_md.lower()]

                if not likely_here:
                    return []

                expected_list = "\n".join(f"- {title}" for title in likely_here)
                prompt = prompt_template.format(
                    expected_list=expected_list,
                    chapter_md=chapter_md,
                )
            else:
                # Fallback: extract all recipes found
                prompt = f"""Extract ALL complete recipes from this chapter.
Copy titles EXACTLY as they appear in the text.
Do NOT add commentary or labels.

<chapter>
{chapter_md}
</chapter>"""

            # Retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.async_client.responses.parse(
                        model=self.model,
                        input=[EasyInputMessageParam(role="user", content=prompt)],
                        text_format=CookbookRecipes,
                    )

                    return response.output_parsed.recipes

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for chapter {chapter_name} "
                            f"after error: {e}. Waiting {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"Failed to extract from chapter {chapter_name} after {max_retries} attempts: {e}")
                        return []


# ============================================================================
# EXTRACTION PIPELINE
# ============================================================================


class ExtractionPipeline:
    """Main extraction pipeline with validation and iteration."""

    def __init__(self, max_concurrent_chapters: int = 5):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.max_concurrent_chapters = max_concurrent_chapters
        self.chapter_processor = ChapterProcessor()

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

    async def extract_recipes(
        self,
        epub_path: str,
        prompts: PromptLibrary,
        model: str = "gpt-5-nano",
    ) -> Tuple[ExtractionResult, List[Tuple[str, str]], List[str]]:
        """
        Extract recipes from EPUB using chapter-based approach.

        Args:
            epub_path: Path to EPUB file
            prompts: Prompt library to use
            model: Model for extraction (gpt-5-nano or gpt-5-mini)

        Returns:
            Tuple of (ExtractionResult, chapters, discovered_recipe_list)
        """
        start_time = time.time()

        # Phase 1: Convert chapters
        logging.info("Phase 1: Converting EPUB chapters to markdown")
        book, chapters = self.chapter_processor.convert_epub_by_chapters(epub_path)

        # Phase 2: Discover recipe list
        logging.info("Phase 2: Discovering recipe list")
        discoverer = RecipeListDiscoverer(self.client, model="gpt-5-mini")
        expected_titles = discoverer.discover_recipe_list(chapters, prompts.discovery_prompt)

        if not expected_titles:
            logging.warning("No recipe list discovered - proceeding without expectations")

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
            for chapter_name, chapter_md in content_chapters
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

        for recipe in complete_recipes:
            recipe_title = recipe.title if hasattr(recipe, 'title') else dict(recipe).get('title', '')
            if recipe_title not in seen:
                seen.add(recipe_title)
                unique_recipes.append(recipe)
            else:
                duplicates += 1
                logging.debug(f"Duplicate removed: {recipe_title}")

        extraction_time = time.time() - start_time

        result = ExtractionResult(
            recipes=unique_recipes,
            total_extracted=len(all_recipes),
            unique_count=len(unique_recipes),
            duplicates_removed=duplicates,
            chapters_processed=len(chapters),
            chapters_with_recipes=chapters_with_recipes,
            extraction_time=extraction_time,
            model_used=model,
            metadata={
                "expected_count": len(expected_titles) if expected_titles else 0,
            }
        )

        return result, chapters, expected_titles or []

    def validate_extraction(
        self,
        results: ExtractionResult,
        discovered_list: List[str],
    ) -> ValidationReport:
        """
        Validate extraction against discovered recipe list.

        Args:
            results: Extraction results
            discovered_list: Discovered recipe titles

        Returns:
            ValidationReport with detailed diff
        """
        if not discovered_list:
            raise ValidationError("Cannot validate without discovered recipe list")

        extracted_titles = {r.title for r in results.recipes}
        discovered_titles = set(discovered_list)

        matched = extracted_titles & discovered_titles
        missing = discovered_titles - extracted_titles
        extra = extracted_titles - discovered_titles

        match_percentage = (len(matched) / len(discovered_titles) * 100) if discovered_titles else 0
        is_perfect = len(missing) == 0 and len(extra) == 0

        report = ValidationReport(
            extracted_titles=extracted_titles,
            discovered_titles=discovered_titles,
            matched_titles=matched,
            missing_titles=missing,
            extra_titles=extra,
            match_percentage=match_percentage,
            total_discovered=len(discovered_titles),
            total_extracted=len(extracted_titles),
            is_perfect_match=is_perfect,
        )

        # Log summary
        logging.info(f"Validation: {len(matched)}/{len(discovered_titles)} matched ({match_percentage:.1f}%)")
        if missing:
            logging.warning(f"Missing {len(missing)} recipes: {sorted(list(missing))[:10]}")
        if extra:
            logging.warning(f"Extra {len(extra)} recipes: {sorted(list(extra))[:10]}")

        return report

    async def analyze_gaps(
        self,
        validation_report: ValidationReport,
        chapters: List[Tuple[str, str]],
        prompts: PromptLibrary,
    ) -> PromptImprovements:
        """
        Use LLM to analyze WHY recipes were missed or duplicated.

        Args:
            validation_report: Validation results
            chapters: Original chapters for context
            prompts: Current prompts

        Returns:
            PromptImprovements with suggestions
        """
        # Build analysis prompt
        missing_samples = list(validation_report.missing_titles)[:5]
        extra_samples = list(validation_report.extra_titles)[:5]

        # Find chapters where missing recipes might be
        missing_context = []
        for title in missing_samples:
            for chapter_name, chapter_md in chapters:
                if title.lower() in chapter_md.lower():
                    # Extract snippet
                    snippet = self._extract_recipe_snippet(chapter_md, title)
                    missing_context.append(f"Recipe: {title}\nChapter: {chapter_name}\nSnippet:\n{snippet}\n")
                    break

        analysis_prompt = f"""You are an expert at analyzing LLM prompt failures for recipe extraction.

CURRENT SITUATION:
- Total recipes expected: {validation_report.total_discovered}
- Successfully extracted: {len(validation_report.matched_titles)}
- Missing: {len(validation_report.missing_titles)}
- Extra (false positives): {len(validation_report.extra_titles)}
- Match rate: {validation_report.match_percentage:.1f}%

CURRENT PROMPTS:
Discovery Prompt:
{prompts.discovery_prompt}

Extraction Prompt:
{prompts.extraction_prompt}

MISSING RECIPES (samples):
{chr(10).join(f"- {t}" for t in missing_samples)}

EXTRA RECIPES (samples):
{chr(10).join(f"- {t}" for t in extra_samples)}

CONTEXT FOR MISSING RECIPES:
{chr(10).join(missing_context[:3])}

TASK:
Analyze why the extraction failed and suggest specific prompt improvements.

Focus on:
1. Patterns in missing recipes (are they in a specific format? section? style?)
2. Patterns in false positives (what's being extracted that shouldn't be?)
3. How to improve the discovery prompt to find the correct recipe list
4. How to improve the extraction prompt to match titles exactly
5. Edge cases the current prompts don't handle

Be specific and actionable. Suggest exact wording changes."""

        try:
            response = await self.async_client.responses.parse(
                model="gpt-5-mini",
                input=[EasyInputMessageParam(role="user", content=analysis_prompt)],
                text_format=GapAnalysis,
            )

            analysis = response.output_parsed

            improvements = PromptImprovements(
                analysis=analysis.analysis,
                missing_recipe_patterns=analysis.missing_patterns,
                false_positive_patterns=analysis.false_positive_patterns,
                suggested_discovery_changes=analysis.discovery_improvements,
                suggested_extraction_changes=analysis.extraction_improvements,
                confidence=analysis.confidence,
                reasoning=analysis.reasoning,
            )

            logging.info(f"Gap analysis complete (confidence: {analysis.confidence:.2f})")
            logging.info(f"Analysis: {analysis.analysis[:200]}...")

            return improvements

        except Exception as e:
            raise PromptOptimizationError(f"Failed to analyze gaps: {e}") from e

    def _extract_recipe_snippet(self, chapter_md: str, title: str, context_chars: int = 500) -> str:
        """Extract snippet around recipe title."""
        try:
            idx = chapter_md.lower().index(title.lower())
            start = max(0, idx - context_chars // 2)
            end = min(len(chapter_md), idx + len(title) + context_chars // 2)
            return "..." + chapter_md[start:end] + "..."
        except ValueError:
            return "[Recipe title not found in chapter]"

    async def apply_prompt_improvements(
        self,
        current_prompts: PromptLibrary,
        improvements: PromptImprovements,
    ) -> PromptLibrary:
        """
        Apply suggested improvements to prompts.

        Args:
            current_prompts: Current prompt library
            improvements: Suggested improvements

        Returns:
            New PromptLibrary with updated prompts
        """
        # Use LLM to rewrite prompts based on suggestions
        rewrite_prompt = f"""You are rewriting prompts for recipe extraction based on analysis.

CURRENT DISCOVERY PROMPT:
{current_prompts.discovery_prompt}

CURRENT EXTRACTION PROMPT:
{current_prompts.extraction_prompt}

ANALYSIS AND SUGGESTIONS:
{improvements.analysis}

Discovery improvements needed:
{improvements.suggested_discovery_changes or "None"}

Extraction improvements needed:
{improvements.suggested_extraction_changes or "None"}

TASK:
Rewrite BOTH prompts to fix the identified issues.

Rules:
- Keep the same structure and placeholders ({{combined_lists}}, {{expected_list}}, {{chapter_md}})
- Make changes that directly address the patterns identified
- Be more specific where vagueness caused errors
- Add edge case handling
- Keep prompts concise but complete

Return the improved prompts."""

        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "user", "content": rewrite_prompt}
                ],
            )

            improved_text = response.choices[0].message.content

            # Parse out the two prompts (simple heuristic)
            # Look for sections marked with headers or code blocks
            discovery_section = self._extract_section(improved_text, "discovery", current_prompts.discovery_prompt)
            extraction_section = self._extract_section(improved_text, "extraction", current_prompts.extraction_prompt)

            new_prompts = PromptLibrary(
                discovery_prompt=discovery_section,
                extraction_prompt=extraction_section,
                version=current_prompts.version + 1,
                iteration_history=current_prompts.iteration_history + [
                    {
                        "version": current_prompts.version + 1,
                        "changes": improvements.analysis,
                        "confidence": improvements.confidence,
                    }
                ],
                locked=False,
            )

            logging.info(f"Prompts updated to version {new_prompts.version}")
            return new_prompts

        except Exception as e:
            raise PromptOptimizationError(f"Failed to apply improvements: {e}") from e

    def _extract_section(self, text: str, section_type: str, fallback: str) -> str:
        """Extract a prompt section from LLM response."""
        # Try to find code blocks or clear sections
        patterns = [
            rf"```\n(.*?{section_type}.*?)\n```",
            rf"{section_type}[^\n]*:\s*\n(.*?)(?=\n\n|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: return original
        logging.warning(f"Could not extract {section_type} section, using original")
        return fallback

    async def iterative_refinement(
        self,
        epub_path: str,
        max_iterations: int = 10,
        model: str = "gpt-5-nano",
        output_dir: Optional[str] = None,
    ) -> Tuple[List[MelaRecipe], PromptLibrary, List[Dict]]:
        """
        Main iteration loop for achieving 100% match rate.

        Args:
            epub_path: Path to EPUB file
            max_iterations: Maximum iteration attempts
            model: Model for extraction
            output_dir: Directory to save iteration history

        Returns:
            Tuple of (final_recipes, winning_prompts, iteration_history)
        """
        prompts = PromptLibrary.default()
        iteration_history = []

        for iteration in range(1, max_iterations + 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"ITERATION {iteration}/{max_iterations}")
            logging.info(f"{'='*80}\n")

            # Extract
            try:
                results, chapters, discovered_list = await self.extract_recipes(
                    epub_path, prompts, model
                )
            except Exception as e:
                logging.error(f"Extraction failed: {e}")
                raise ExtractionError(f"Iteration {iteration} extraction failed") from e

            # Validate
            if not discovered_list:
                logging.warning("No discovered list - cannot validate")
                # Return what we have
                return results.recipes, prompts, iteration_history

            try:
                validation = self.validate_extraction(results, discovered_list)
            except Exception as e:
                logging.error(f"Validation failed: {e}")
                raise ValidationError(f"Iteration {iteration} validation failed") from e

            # Record iteration
            iter_record = {
                "iteration": iteration,
                "match_percentage": validation.match_percentage,
                "matched": len(validation.matched_titles),
                "missing": len(validation.missing_titles),
                "extra": len(validation.extra_titles),
                "missing_titles": sorted(list(validation.missing_titles))[:10],
                "extra_titles": sorted(list(validation.extra_titles))[:10],
                "prompt_version": prompts.version,
                "extraction_time": results.extraction_time,
            }
            iteration_history.append(iter_record)

            # Save iteration snapshot
            if output_dir:
                snapshot_path = Path(output_dir) / f"iteration_{iteration}.json"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                with open(snapshot_path, "w") as f:
                    json.dump({
                        "iteration": iter_record,
                        "validation": validation.to_dict(),
                        "prompts": prompts.to_dict(),
                    }, f, indent=2)

            # Check if perfect
            if validation.is_perfect_match:
                logging.info(f"\n🎉 100% MATCH ACHIEVED in iteration {iteration}!")
                return results.recipes, prompts, iteration_history

            # If not perfect and not last iteration, improve
            if iteration < max_iterations:
                logging.info(f"\nAnalyzing gaps to improve prompts...")

                try:
                    improvements = await self.analyze_gaps(validation, chapters, prompts)
                    prompts = await self.apply_prompt_improvements(prompts, improvements)
                except Exception as e:
                    logging.error(f"Prompt improvement failed: {e}")
                    # Continue with current prompts
                    logging.info("Continuing with current prompts")
            else:
                logging.warning(f"\nMax iterations ({max_iterations}) reached without 100% match")

        # Return best attempt
        return results.recipes, prompts, iteration_history


# ============================================================================
# RECIPE OUTPUT
# ============================================================================


def write_recipes_to_disk(
    recipes: List[MelaRecipe],
    book: epub.EpubBook,
    epub_path: str,
    output_dir: str,
) -> Tuple[int, str]:
    """
    Write recipes to disk in Mela format.

    Args:
        recipes: List of recipes to write
        book: EpubBook for metadata
        epub_path: Path to original EPUB
        output_dir: Output directory

    Returns:
        Tuple of (written_count, archive_path)
    """
    book_title = book.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    out_dir = Path(output_dir) / f"{book_slug}-chapters-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(epub_path, book)
    written = 0

    for recipe in recipes:
        recipe_dict = processor._mela_recipe_to_object(recipe)
        recipe_dict["link"] = book_title
        recipe_dict["id"] = str(uuid.uuid4())
        recipe_dict["images"] = []

        if processor.write_recipe(recipe_dict, output_dir=str(out_dir)):
            written += 1

    # Create archive
    archive_zip = shutil.make_archive(base_name=str(out_dir), format="zip", root_dir=str(out_dir))
    archive_mela = archive_zip.replace(".zip", ".melarecipes")
    os.rename(archive_zip, archive_mela)

    return written, archive_mela


# ============================================================================
# CLI
# ============================================================================


def setup_logging(log_file: str = "extraction_pipeline.log", verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Production-ready LLM-powered recipe extraction with automatic prompt iteration"
    )
    parser.add_argument("epub_path", type=str, help="Path to EPUB file")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        choices=["gpt-5-nano", "gpt-5-mini"],
        help="Model for chapter extraction",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iteration attempts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-iteration",
        action="store_true",
        help="Skip iteration and use default prompts",
    )
    parser.add_argument(
        "--prompt-library",
        type=str,
        help="Path to prompt library JSON (for locked prompts)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        parser.error(f"File not found: {args.epub_path}")

    setup_logging(verbose=args.verbose)

    logging.info("="*80)
    logging.info("LLM-Powered Recipe Extraction Pipeline with Automatic Prompt Iteration")
    logging.info("="*80)
    logging.info(f"EPUB: {args.epub_path}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Max iterations: {args.max_iterations}")
    logging.info(f"Skip iteration: {args.skip_iteration}")
    logging.info("="*80)

    pipeline = ExtractionPipeline(max_concurrent_chapters=5)

    try:
        start_time = time.time()

        if args.skip_iteration:
            # Single-pass extraction
            logging.info("\nRunning single-pass extraction (no iteration)")

            prompts = PromptLibrary.default()
            if args.prompt_library:
                with open(args.prompt_library, "r") as f:
                    prompts = PromptLibrary.from_dict(json.load(f))

            results, chapters, discovered_list = await pipeline.extract_recipes(
                args.epub_path, prompts, args.model
            )

            if discovered_list:
                validation = pipeline.validate_extraction(results, discovered_list)
                logging.info(f"\nFinal match rate: {validation.match_percentage:.1f}%")

            final_recipes = results.recipes
            final_prompts = prompts
            iteration_history = []
        else:
            # Iterative refinement
            logging.info("\nRunning iterative refinement")

            final_recipes, final_prompts, iteration_history = await pipeline.iterative_refinement(
                args.epub_path,
                max_iterations=args.max_iterations,
                model=args.model,
                output_dir=args.output_dir,
            )

        # Write recipes
        logging.info("\n" + "="*80)
        logging.info("Writing recipes to disk")
        logging.info("="*80)

        book = epub.read_epub(args.epub_path, {"ignore_ncx": True})
        written_count, archive_path = write_recipes_to_disk(
            final_recipes, book, args.epub_path, args.output_dir
        )

        total_time = time.time() - start_time

        # Final summary
        logging.info("\n" + "="*80)
        logging.info("PIPELINE COMPLETE")
        logging.info("="*80)
        logging.info(f"Total recipes: {len(final_recipes)}")
        logging.info(f"Written to disk: {written_count}")
        logging.info(f"Archive: {archive_path}")
        logging.info(f"Total time: {total_time:.1f}s")

        if iteration_history:
            logging.info(f"\nIteration summary:")
            for record in iteration_history:
                logging.info(
                    f"  Iteration {record['iteration']}: "
                    f"{record['match_percentage']:.1f}% match "
                    f"({record['matched']}/{record['matched'] + record['missing']})"
                )

        # Save final prompts
        prompt_output = Path(args.output_dir) / "final_prompts.json"
        with open(prompt_output, "w") as f:
            json.dump(final_prompts.to_dict(), f, indent=2)
        logging.info(f"\nFinal prompts saved to: {prompt_output}")

        # Save full history
        history_output = Path(args.output_dir) / "iteration_history.json"
        with open(history_output, "w") as f:
            json.dump(iteration_history, f, indent=2)
        logging.info(f"Iteration history saved to: {history_output}")

    except ExtractionError as e:
        logging.error(f"Extraction error: {e}")
        return 1
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        return 1
    except PromptOptimizationError as e:
        logging.error(f"Prompt optimization error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0


def main():
    """Main entry point."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())
