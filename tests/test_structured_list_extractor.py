"""
Tests for the structured list extractor, focusing on LLM validation of recipe extraction.
"""
import pytest
from pathlib import Path
from ebooklib import epub
from src.mela_parser.extractors.structured_list import (
    StructuredListExtractor,
    CandidateLink,
    RecipeLink
)


class TestLLMValidation:
    """Test LLM validation of recipe extraction from various cookbooks."""

    @pytest.fixture
    def extractor(self):
        """Create a StructuredListExtractor instance."""
        return StructuredListExtractor()

    def load_epub(self, filename: str) -> epub.EpubBook:
        """Load an EPUB file from the examples/input directory."""
        examples_dir = Path(__file__).parent.parent / "examples" / "input"
        epub_path = examples_dir / filename

        if not epub_path.exists():
            pytest.skip(f"EPUB file not found: {epub_path}")

        return epub.read_epub(str(epub_path))

    def create_candidates_from_book(self, book: epub.EpubBook) -> list[CandidateLink]:
        """Extract candidate links from a cookbook."""
        extractor = StructuredListExtractor()
        pages = extractor.find_recipe_list_pages(book)

        if not pages:
            # If no dedicated list pages, look at TOC or navigation
            # This is a fallback for books that structure differently
            import ebooklib
            from bs4 import BeautifulSoup

            all_links = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                if 'nav' in item.file_name.lower() or 'toc' in item.file_name.lower():
                    content = item.get_content().decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(content, 'html.parser')

                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        fragment = None

                        if '#' in href:
                            href_parts = href.split('#')
                            href = href_parts[0]
                            fragment = href_parts[1] if len(href_parts) > 1 else None

                        if text and href:
                            all_links.append(CandidateLink(
                                title=text,
                                href=href,
                                fragment=fragment,
                                source_page=item.file_name,
                                css_class=link.get('class', [''])[0] if link.get('class') else ''
                            ))

            return all_links

        # Extract links from identified list pages
        all_links = []
        for page in pages:
            links = extractor.extract_links_from_page(page)
            for link in links:
                all_links.append(CandidateLink(
                    title=link.title,
                    href=link.href,
                    fragment=link.fragment,
                    source_page=link.source_page,
                    css_class=link.css_class
                ))

        return all_links

    @pytest.mark.slow
    def test_jerusalem_llm_confirms_all_125(self, extractor):
        """
        Test Jerusalem cookbook - should identify 125 recipes.
        This cookbook has a clear structure with all recipes well-defined.
        """
        book = self.load_epub("jerusalem.epub")

        # Get all candidate links
        candidates = self.create_candidates_from_book(book)

        # Filter structural non-recipes
        filtered = extractor.apply_structural_filters(candidates)

        # Validate with LLM
        validated = extractor.validate_with_llm(filtered.candidates)

        # Jerusalem has exactly 125 recipes
        assert len(validated.recipes) == 125

        # Check some known recipes are present
        recipe_titles = {r.title for r in validated.recipes}
        # Using actual recipes from Jerusalem cookbook
        expected_keywords = {
            "hummus", "falafel", "tahini", "pistachio", "barley"
        }
        for keyword in expected_keywords:
            assert any(keyword.lower() in title.lower() for title in recipe_titles), \
                f"No recipe with keyword '{keyword}' found in extracted list"

    def test_modern_way_llm_confirms_142(self, extractor):
        """
        Test The Modern Way cookbook - should identify 142 recipes.
        This cookbook uses a different structure with categorized sections.
        """
        book = self.load_epub("a-modern-way-to-eat.epub")

        # Get all candidate links
        candidates = self.create_candidates_from_book(book)

        # Filter structural non-recipes
        filtered = extractor.apply_structural_filters(candidates)

        # Validate with LLM
        validated = extractor.validate_with_llm(filtered.candidates)

        # The Modern Way has exactly 142 recipes
        assert len(validated.recipes) == 142

    def test_completely_perfect_llm_identifies_122_recipes(self, extractor):
        """
        Test Completely Perfect cookbook - should identify 122 recipes.
        This cookbook might have a mixed structure with both recipes and technique sections.
        """
        book = self.load_epub("completely-perfect.epub")

        # Get all candidate links
        candidates = self.create_candidates_from_book(book)

        # Filter structural non-recipes
        filtered = extractor.apply_structural_filters(candidates)

        # Validate with LLM
        validated = extractor.validate_with_llm(filtered.candidates)

        # Completely Perfect has exactly 122 recipes
        assert len(validated.recipes) == 122

    def test_simple_llm_identifies_140_recipes(self, extractor):
        """
        Test Simple cookbook - should identify 140 recipes.
        Ottolenghi Simple has a well-structured format with clear recipe titles.
        """
        book = self.load_epub("simple.epub")

        # Get all candidate links
        candidates = self.create_candidates_from_book(book)

        # Filter structural non-recipes
        filtered = extractor.apply_structural_filters(candidates)

        # Validate with LLM
        validated = extractor.validate_with_llm(filtered.candidates)

        # Simple has exactly 140 recipes
        assert len(validated.recipes) == 140

        # Check some known Ottolenghi Simple recipes
        recipe_titles = {r.title for r in validated.recipes}
        expected_recipes = {
            "Roasted", "Grilled", "Salad", "Rice", "Chicken"
        }
        for keyword in expected_recipes:
            assert any(keyword.lower() in title.lower() for title in recipe_titles), \
                f"No recipes with keyword '{keyword}' found in extracted list"


@pytest.mark.slow
class TestStructuralExtraction:
    """Test pure structural extraction without LLM validation."""

    def test_high_link_density_detection(self):
        """Test detection of high-link-density pages (recipe lists)."""
        extractor = StructuredListExtractor()
        # Implementation would test the structural detection logic
        pass

    def test_filter_page_numbers(self):
        """Test filtering of page number links."""
        extractor = StructuredListExtractor()

        candidates = [
            CandidateLink("Hummus", "chapter1.html#rec1"),
            CandidateLink("15", "page15.html"),
            CandidateLink("234", "page234.html"),
            CandidateLink("Falafel", "chapter2.html#rec2"),
            CandidateLink("xiv", "frontmatter.html"),
        ]

        filtered = extractor.apply_structural_filters(candidates)

        # Should keep only actual recipe names
        assert len(filtered.candidates) == 2
        assert all(c.title in ["Hummus", "Falafel"] for c in filtered.candidates)
        assert len(filtered.excluded) == 3