"""
Structure-based recipe list extraction from EPUB files.

This module extracts recipe lists by analyzing EPUB structure (links, hrefs, patterns)
with optional lightweight LLM validation for ambiguous cases.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class RecipeLink:
    """A link to a recipe with its metadata"""
    title: str
    href: str
    fragment: Optional[str] = None  # e.g., "rec59" from "#rec59"
    source_page: str = ""  # Which page contained this link
    css_class: str = ""


@dataclass
class CandidateLink:
    """Same as RecipeLink but for candidates before validation"""
    title: str
    href: str
    fragment: Optional[str] = None
    source_page: str = ""
    css_class: str = ""


@dataclass
class FilteredLinks:
    """Result of structural filtering"""
    candidates: List[CandidateLink]  # Likely recipes
    excluded: List[CandidateLink]    # Filtered out (front matter, sections, etc.)


@dataclass
class ValidationResult:
    """Result of LLM validation"""
    recipes: List[RecipeLink]      # Confirmed recipes (with normalized titles)
    non_recipes: List[RecipeLink]  # Not recipes (e.g., notes, appendices)
    title_normalizations: dict = None  # Original -> Normalized title mappings


class StructuredListExtractor:
    """
    Extract recipe lists from EPUB structure using:
    1. Pattern matching (high-link-density pages, filename patterns)
    2. Structural filtering (chapter_* vs. cover*, about*, etc.)
    3. Optional LLM validation for ambiguous cases
    """

    def __init__(self):
        """Initialize extractor"""
        pass

    def find_recipe_list_pages(self, book: epub.EpubBook) -> List:
        """
        Find pages that contain recipe lists using ONLY structural analysis.

        Criteria (NO filename cheating):
        1. High link density (>100 links)
        2. Links point to diverse chapter files (not all same file)
        3. Link text looks like recipe titles (not "Chapter 1", "Next", etc.)
        4. Page has minimal body text (mostly links, not content)

        Returns:
            List of EpubItem objects that are recipe list pages
        """
        candidates_with_score = []

        # Get all document items
        documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

        for doc in documents:
            try:
                content = doc.get_content().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.find_all('a', href=True)

                if len(links) < 15:  # Lowered to catch chapter-level TOCs
                    continue

                # Analyze link characteristics
                link_targets = set()
                link_texts = []
                for link in links:
                    href = link.get('href', '')
                    if href and href != '#':
                        # Extract base filename (before #fragment)
                        base_href = href.split('#')[0]
                        if base_href:
                            link_targets.add(base_href)
                        link_texts.append(link.get_text(strip=True))

                # Calculate structural score
                score = 0

                # Score 1: Target uniqueness ratio (comprehensive lists have mostly unique targets)
                # This is KEY: canonical recipe lists have link_count ≈ unique_targets
                # Partial/duplicate lists have link_count >> unique_targets
                target_uniqueness = len(link_targets) / len(links) if len(links) > 0 else 0
                score += int(target_uniqueness * 300)  # Max 300 points - HIGHEST WEIGHT

                # Score 2: Absolute link count (more is better, up to a point)
                score += min(len(links), 150)  # Max 150 points

                # Score 3: Link diversity (many different targets = comprehensive)
                target_diversity = len(link_targets)
                score += min(target_diversity, 150)  # Max 150 points

                # Score 4: Link text quality (not navigation)
                nav_keywords = {'next', 'previous', 'back', 'chapter', 'section', 'part', 'home', 'contents'}
                non_nav_links = sum(1 for text in link_texts
                                   if text and text.lower() not in nav_keywords and len(text) > 5)
                score += min(non_nav_links, 100)  # Max 100 points

                candidates_with_score.append((score, target_uniqueness, len(links), doc))

            except Exception:
                continue

        # Sort by score (descending), then uniqueness ratio, then link count
        candidates_with_score.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        # Return ALL high-quality TOC pages (main TOC + chapter TOCs)
        # Filter: score > 200 to exclude low-quality link pages
        quality_threshold = 200
        toc_pages = [doc for score, _, _, doc in candidates_with_score if score >= quality_threshold]

        if not toc_pages:
            # Fallback: return top candidate if no pages meet threshold
            toc_pages = [candidates_with_score[0][3]] if candidates_with_score else []

        return toc_pages

    def extract_links_from_page(self, page) -> List[CandidateLink]:
        """
        Extract all links from a single page.

        Args:
            page: An EpubItem (HTML document) to extract links from

        Returns:
            List of CandidateLink objects
        """
        links = []
        try:
            content = page.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link['href']
                title = link.get_text(strip=True)
                css_class = link.get('class', [''])[0] if link.get('class') else ''

                # Parse fragment ID if present
                fragment = None
                if '#' in href:
                    href_parts = href.split('#')
                    href = href_parts[0]
                    fragment = href_parts[1] if len(href_parts) > 1 else None

                # Skip empty titles or self-references
                if not title or href == '#':
                    continue

                candidate = CandidateLink(
                    title=title,
                    href=href,
                    fragment=fragment,
                    source_page=page.file_name,
                    css_class=css_class
                )
                links.append(candidate)

        except Exception as e:
            logger.warning(f"Error extracting links from page {getattr(page, 'file_name', 'unknown')}: {e}")

        return links

    def get_page_content_safe(self, book: epub.EpubBook, href: str) -> Optional[str]:
        """
        Get page content, trying common path prefixes if needed.

        Args:
            book: EPUB book object
            href: Page href (may be relative)

        Returns:
            HTML content as string, or None if not found
        """
        # Try direct lookup
        item = book.get_item_with_href(href)

        # Try common prefixes
        if not item:
            for prefix in ['pages/', 'text/', 'xhtml/', 'OEBPS/', 'content/']:
                item = book.get_item_with_href(f'{prefix}{href}')
                if item:
                    break

        if item:
            return item.get_content().decode('utf-8', errors='ignore')
        return None

    def get_anchor_positions(self, book: epub.EpubBook, href: str) -> dict:
        """
        Find positions of all id= anchors in a page's HTML.

        Args:
            book: EPUB book object
            href: Page href

        Returns:
            Dict mapping fragment_id -> character position in HTML
        """
        html = self.get_page_content_safe(book, href)
        if not html:
            return {}

        positions = {}
        # Find all id="..." or id='...' attributes
        for match in re.finditer(r'id=["\']([^"\']+)["\']', html):
            fragment_id = match.group(1)
            positions[fragment_id] = match.start()

        return positions

    def are_fragments_close(self, book: epub.EpubBook, link1: CandidateLink, link2: CandidateLink,
                           threshold: int = 3000) -> bool:
        """
        Check if two links point to nearby locations in the same page.

        Args:
            book: EPUB book object
            link1, link2: Links to compare
            threshold: Character distance threshold (default: 3000 chars)

        Returns:
            True if links are to same page with nearby anchors
        """
        # Must be same page
        if link1.href != link2.href:
            return False

        # If no fragments, they're the same (both point to top of page)
        if not link1.fragment and not link2.fragment:
            return True

        # If only one has fragment, they're different
        if not link1.fragment or not link2.fragment:
            return False

        # Get anchor positions
        positions = self.get_anchor_positions(book, link1.href)
        pos1 = positions.get(link1.fragment)
        pos2 = positions.get(link2.fragment)

        # If can't find anchors, assume different
        if pos1 is None or pos2 is None:
            return False

        # Check if close together
        distance = abs(pos1 - pos2)
        return distance < threshold

    def extract_all_links_from_book(self, book: epub.EpubBook) -> List[CandidateLink]:
        """
        Extract all links from recipe list pages.

        Returns:
            List of all links with title, href, fragment, etc.
        """
        # Find the recipe list page(s)
        list_pages = self.find_recipe_list_pages(book)
        if not list_pages:
            return []

        all_links = []

        for page in list_pages:
            try:
                content = page.get_content().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.find_all('a', href=True)

                for link in links:
                    href = link['href']
                    title = link.get_text(strip=True)
                    css_class = link.get('class', [''])[0] if link.get('class') else ''

                    # Parse fragment ID if present
                    fragment = None
                    if '#' in href:
                        href_parts = href.split('#')
                        href = href_parts[0]
                        fragment = href_parts[1] if len(href_parts) > 1 else None

                    # Skip empty titles or self-references
                    if not title or href == '#':
                        continue

                    candidate = CandidateLink(
                        title=title,
                        href=href,
                        fragment=fragment,
                        source_page=page.file_name,
                        css_class=css_class
                    )
                    all_links.append(candidate)

            except Exception:
                # Skip if we can't parse
                continue

        return all_links

    def apply_structural_filters(self, links: List[CandidateLink], book: Optional[epub.EpubBook] = None) -> FilteredLinks:
        """
        Minimal structural filtering with proximity-based deduplication.

        Filters:
        1. Empty/too short titles
        2. Pure page numbers
        3. Numbered sections
        4. Proximity deduplication (if book provided)

        Returns:
            FilteredLinks with candidates and excluded
        """
        # Step 1: Filter obvious non-recipes
        initial_candidates = []
        excluded = []

        for link in links:
            should_exclude = False

            # Filter: obvious non-recipes only
            if len(link.title) < 3 or not link.href or link.href == '#':
                should_exclude = True
            elif re.match(r'^\d+$', link.title) or re.match(r'^[ivxlcdm]{2,}$', link.title.lower()):
                should_exclude = True
            elif re.match(r'^\d+[a-zA-Z]', link.title):
                should_exclude = True

            if should_exclude:
                excluded.append(link)
            else:
                initial_candidates.append(link)

        # Step 2: Proximity-based deduplication (if book provided)
        if not book:
            # No book - just dedupe by (href, fragment)
            candidates = []
            seen = set()
            for link in initial_candidates:
                key = (link.href, link.fragment)
                if key not in seen:
                    seen.add(key)
                    candidates.append(link)
                else:
                    excluded.append(link)
            return FilteredLinks(candidates=candidates, excluded=excluded)

        # With book: use proximity-based deduplication
        candidates = []
        for link in initial_candidates:
            # Check if this link is close to any already-kept link
            is_duplicate = False
            for kept_link in candidates:
                if self.are_fragments_close(book, link, kept_link, threshold=3000):
                    is_duplicate = True
                    excluded.append(link)
                    break

            if not is_duplicate:
                candidates.append(link)

        return FilteredLinks(candidates=candidates, excluded=excluded)

    def validate_with_llm(self, candidates: List[CandidateLink], model: str = "gpt-4o-mini") -> ValidationResult:
        """
        Validate candidates with single LLM API call using STRUCTURAL CONTEXT.

        Sends titles + structural metadata (href patterns, fragments, target analysis)
        to help LLM distinguish recipes from section headers/navigation.

        Args:
            candidates: List of candidate links to validate
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)

        Returns:
            ValidationResult with confirmed recipes and non-recipes
        """
        if not candidates:
            return ValidationResult(recipes=[], non_recipes=[])

        # Analyze structural patterns across all candidates
        href_targets = {}
        for link in candidates:
            href_targets[link.href] = href_targets.get(link.href, 0) + 1

        # Build rich structured list with metadata for LLM
        structured_list = []
        for i, link in enumerate(candidates, 1):
            # Count how many other links point to same target
            target_count = href_targets.get(link.href, 1)

            # Build metadata hints
            metadata = []

            # Add href hint
            href_filename = link.href.split('/')[-1]  # Just the filename part
            metadata.append(f"→{href_filename}")

            if link.fragment:
                metadata.append(f"#{link.fragment}")
            if target_count > 1:
                metadata.append(f"{target_count}x")  # Multiple links to this file

            meta_str = f" [{', '.join(metadata)}]"

            structured_list.append(f"{i}. {link.title}{meta_str}")

        title_list = "\n".join(structured_list)

        prompt = f"""Analyze cookbook TOC with {len(candidates)} entries.

Each entry: number, title, [→file, #fragment]

TASK 1 - Classify RECIPE vs NON-RECIPE:
✓ RECIPE: Any food/dish - "Hummus", "Apricot tart", "Caesar Salad", "tart, apricot"
✗ NON-RECIPE: Navigation - "Cover", "Index", "Introduction", generic categories alone

TASK 2 - Identify DUPLICATE entries (same recipe, multiple listings):
Look for patterns:
- Inverted titles: "apricot tart" (#frag1) + "tart, apricot" (#frag2) = DUPLICATES
- Same href, similar fragments (#page_44, #page_45) = likely DUPLICATES (page refs)
- Same href, different pattern (#rec01, #rec02) = DIFFERENT recipes
- When uncertain → treat as UNIQUE

{title_list}

Return JSON (indices are 1-based):
{{
  "unique_recipes": [recipe indices - ONE per actual recipe, excluding duplicates],
  "non_recipes": [navigation indices],
  "duplicate_groups": [[1,5], [2,7]]  // optional: groups of same recipe
}}

Note: If entries 1 and 5 are the same recipe, only include 1 in unique_recipes."""

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,  # Deterministic
            )

            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)

            # Handle multiple field name variations
            recipe_indices = set(result_data.get("unique_recipes",
                                result_data.get("unique_recipe_indices",
                                result_data.get("recipe_indices", []))))
            non_recipe_indices = set(result_data.get("non_recipes",
                                     result_data.get("non_recipe_indices", [])))

            recipes = []
            non_recipes = []

            for i, candidate in enumerate(candidates, start=1):
                # Convert CandidateLink to RecipeLink
                recipe_link = RecipeLink(
                    title=candidate.title,
                    href=candidate.href,
                    fragment=candidate.fragment,
                    source_page=candidate.source_page,
                    css_class=candidate.css_class
                )

                if i in recipe_indices:
                    recipes.append(recipe_link)
                elif i in non_recipe_indices:
                    non_recipes.append(recipe_link)
                else:
                    # If not explicitly categorized, default to recipe (permissive)
                    recipes.append(recipe_link)

            logger.info(f"LLM validated: {len(recipes)} recipes, {len(non_recipes)} non-recipes from {len(candidates)} candidates")

            return ValidationResult(recipes=recipes, non_recipes=non_recipes)

        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            # Fallback: treat all candidates as recipes
            recipes = [RecipeLink(
                title=c.title,
                href=c.href,
                fragment=c.fragment,
                source_page=c.source_page,
                css_class=c.css_class
            ) for c in candidates]
            return ValidationResult(recipes=recipes, non_recipes=[])

    def sort_by_book_order(self, links: List[RecipeLink], spine: List) -> List[RecipeLink]:
        """
        Sort recipes by book appearance order using EPUB spine.

        The spine defines the reading order of content.
        Maps each recipe's href to spine index and sorts accordingly.

        Returns:
            Recipes sorted by book order (not alphabetical)
        """
        # TODO: Implement in GREEN phase
        raise NotImplementedError("RED phase - test should fail")

    def extract_recipe_list(self, book: epub.EpubBook) -> List[RecipeLink]:
        """
        Main method: extract complete recipe list from EPUB.

        Pipeline:
        1. Find recipe list pages
        2. Extract all links
        3. Apply structural filters
        4. Validate with LLM (if needed)
        5. Return confirmed recipe links

        Returns:
            List of RecipeLink objects
        """
        # TODO: Implement in GREEN phase
        raise NotImplementedError("RED phase - test should fail")
