"""Unit tests for mela_parser.converter module.

Tests EPUB to markdown conversion, token estimation, and chunking.
"""

from unittest.mock import MagicMock

import pytest

from mela_parser.converter import EpubConverter, convert_epub_by_chapters


class TestEpubConverterInit:
    """Tests for EpubConverter initialization."""

    def test_init_creates_markitdown(self) -> None:
        """EpubConverter creates MarkItDown instance."""
        converter = EpubConverter()
        assert converter.md is not None


class TestEstimateTokens:
    """Tests for EpubConverter.estimate_tokens method."""

    @pytest.fixture
    def converter(self) -> EpubConverter:
        """Create converter instance."""
        return EpubConverter()

    def test_empty_string(self, converter: EpubConverter) -> None:
        """Empty string returns 0 tokens."""
        assert converter.estimate_tokens("") == 0

    def test_short_text(self, converter: EpubConverter) -> None:
        """Short text estimates correctly (4 chars per token)."""
        # 12 chars = 3 tokens
        assert converter.estimate_tokens("Hello World!") == 3

    def test_exact_multiple(self, converter: EpubConverter) -> None:
        """Text that's exact multiple of 4."""
        # 400 chars = 100 tokens
        assert converter.estimate_tokens("A" * 400) == 100

    def test_not_exact_multiple(self, converter: EpubConverter) -> None:
        """Text that's not exact multiple of 4 (floors)."""
        # 401 chars = 100 tokens (floors)
        assert converter.estimate_tokens("A" * 401) == 100
        # 403 chars = 100 tokens
        assert converter.estimate_tokens("A" * 403) == 100

    def test_large_text(self, converter: EpubConverter) -> None:
        """Large text estimates correctly."""
        # 1,000,000 chars = 250,000 tokens
        text = "x" * 1_000_000
        assert converter.estimate_tokens(text) == 250_000


class TestNeedsChunking:
    """Tests for EpubConverter.needs_chunking method."""

    @pytest.fixture
    def converter(self) -> EpubConverter:
        """Create converter instance."""
        return EpubConverter()

    def test_small_content_no_chunking(self, converter: EpubConverter) -> None:
        """Small content doesn't need chunking."""
        # 100 chars = 25 tokens, well under 200k default
        small_text = "A" * 100
        assert converter.needs_chunking(small_text) is False

    def test_large_content_needs_chunking(self, converter: EpubConverter) -> None:
        """Large content needs chunking."""
        # 1,000,000 chars = 250,000 tokens, exceeds 200k
        large_text = "A" * 1_000_000
        assert converter.needs_chunking(large_text) is True

    def test_custom_max_tokens(self, converter: EpubConverter) -> None:
        """Custom max_tokens is respected."""
        # 1000 chars = 250 tokens
        text = "A" * 1000
        # Under 300 token limit
        assert converter.needs_chunking(text, max_tokens=300) is False
        # Over 200 token limit
        assert converter.needs_chunking(text, max_tokens=200) is True

    def test_boundary_exactly_at_limit(self, converter: EpubConverter) -> None:
        """Content exactly at limit doesn't need chunking."""
        # 800 chars = 200 tokens, exactly at limit
        text = "A" * 800
        assert converter.needs_chunking(text, max_tokens=200) is False


class TestChunkByHeadings:
    """Tests for EpubConverter.chunk_by_headings method."""

    @pytest.fixture
    def converter(self) -> EpubConverter:
        """Create converter instance."""
        return EpubConverter()

    def test_no_headings(self, converter: EpubConverter) -> None:
        """Content without headings returns single chunk."""
        content = "Just some plain text.\nNo headings here."
        chunks = converter.chunk_by_headings(content, level=1)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_level_1_headings(self, converter: EpubConverter) -> None:
        """Splits on level 1 headings."""
        content = "# Chapter 1\nFirst content\n# Chapter 2\nSecond content"
        chunks = converter.chunk_by_headings(content, level=1)
        assert len(chunks) == 2
        assert "# Chapter 1" in chunks[0]
        assert "# Chapter 2" in chunks[1]

    def test_level_2_headings(self, converter: EpubConverter) -> None:
        """Splits on level 2 headings."""
        content = "## Section A\nContent A\n## Section B\nContent B"
        chunks = converter.chunk_by_headings(content, level=2)
        assert len(chunks) == 2
        assert "## Section A" in chunks[0]
        assert "## Section B" in chunks[1]

    def test_preserves_heading_in_chunk(self, converter: EpubConverter) -> None:
        """Each chunk starts with its heading."""
        content = "# Chapter 1\nFirst\n# Chapter 2\nSecond"
        chunks = converter.chunk_by_headings(content, level=1)
        assert chunks[0].startswith("# Chapter 1")
        assert chunks[1].startswith("# Chapter 2")

    def test_ignores_deeper_headings(self, converter: EpubConverter) -> None:
        """Level 1 split ignores level 2+ headings."""
        content = "# Chapter\n## Section 1\nContent\n## Section 2\nMore"
        chunks = converter.chunk_by_headings(content, level=1)
        assert len(chunks) == 1  # No split at ## headings

    def test_content_before_first_heading(self, converter: EpubConverter) -> None:
        """Content before first heading is preserved."""
        content = "Intro text\n# Chapter 1\nFirst\n# Chapter 2\nSecond"
        chunks = converter.chunk_by_headings(content, level=1)
        assert len(chunks) == 3
        assert chunks[0] == "Intro text"

    def test_empty_content(self, converter: EpubConverter) -> None:
        """Empty content returns empty string chunk."""
        chunks = converter.chunk_by_headings("", level=1)
        assert len(chunks) == 1
        assert chunks[0] == ""


class TestStripFrontMatter:
    """Tests for EpubConverter.strip_front_matter method."""

    @pytest.fixture
    def converter(self) -> EpubConverter:
        """Create converter instance."""
        return EpubConverter()

    def test_no_front_matter(self, converter: EpubConverter) -> None:
        """Content without front matter is unchanged."""
        content = "# Recipe\n1 cup flour\n2 tablespoon sugar\n1 teaspoon salt\nMix all."
        result = converter.strip_front_matter(content)
        assert result == content

    def test_strips_foreword(self, converter: EpubConverter) -> None:
        """Strips foreword before recipes when there's enough preamble."""
        # Create many lines of foreword without ingredient patterns (20+ lines)
        foreword_lines = [f"Line {i} of introduction without ingredients." for i in range(25)]
        foreword_lines.insert(0, "# Foreword")
        foreword_lines.insert(10, "# About the Author")

        # Then recipe content with many ingredient patterns
        recipe_lines = [
            "# Main Dishes",
            "## Roasted Chicken",
            "Serves 4",
            "Yield: 4 portions",
            "1 cup breadcrumbs",
            "2 tablespoon olive oil",
            "1 teaspoon salt",
            "500g chicken",
            "250 ml stock",
            "100g butter",
        ]
        content = "\n".join(foreword_lines + recipe_lines)
        result = converter.strip_front_matter(content)

        # The function found recipe start and backed up 10 lines
        # Result should be smaller than original (stripped some foreword)
        assert len(result) < len(content)
        # Recipe content should be preserved
        assert "cup breadcrumbs" in result

    def test_detects_ingredient_patterns(self, converter: EpubConverter) -> None:
        """Detects various ingredient patterns."""
        patterns_content = "cup sugar\ntablespoon butter\nteaspoon vanilla\n500g flour\n250 ml milk"
        result = converter.strip_front_matter(patterns_content)
        # All content has ingredients, should be mostly preserved
        assert "cup sugar" in result

    def test_custom_min_ingredient_count(self, converter: EpubConverter) -> None:
        """Respects min_ingredient_count parameter."""
        # Only 2 ingredient-like lines
        content = "Intro\n1 cup flour\n1 tablespoon oil\nEnd"
        # With min=3, won't find recipe start
        result = converter.strip_front_matter(content, min_ingredient_count=3)
        assert result == content


class TestSmartChunk:
    """Tests for EpubConverter.smart_chunk method."""

    @pytest.fixture
    def converter(self) -> EpubConverter:
        """Create converter instance."""
        return EpubConverter()

    def test_small_content_no_chunk(self, converter: EpubConverter) -> None:
        """Small content returns single chunk."""
        content = "Small recipe content"
        chunks = converter.smart_chunk(content)
        assert len(chunks) == 1

    def test_large_content_chunks_by_heading(self, converter: EpubConverter) -> None:
        """Large content is chunked by headings."""
        # Create content that exceeds the limit
        large_section = "x" * 100_000  # 25k tokens each
        content = f"# Chapter 1\n{large_section}\n# Chapter 2\n{large_section}"

        # With 50k token limit, each chapter should fit
        chunks = converter.smart_chunk(content, max_tokens=50_000)
        assert len(chunks) >= 2


class TestConvertEpubByChapters:
    """Tests for convert_epub_by_chapters function."""

    def test_converts_chapters(self, mock_epub_read, mock_markitdown) -> None:
        """Converts EPUB chapters to markdown."""
        book, chapters = convert_epub_by_chapters("test.epub")

        assert book is not None
        assert len(chapters) == 1
        assert chapters[0].name == "chapter1.xhtml"
        assert chapters[0].index == 0

    def test_multiple_chapters(self, monkeypatch) -> None:
        """Handles multiple chapters."""
        from unittest.mock import MagicMock

        # Create multiple mock items
        items = []
        for i in range(3):
            item = MagicMock()
            item.get_content.return_value = f"<html><body>Chapter {i}</body></html>".encode()
            item.get_name.return_value = f"chapter{i}.xhtml"
            items.append(item)

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = items

        # Mock MarkItDown
        mock_result = MagicMock()
        mock_result.text_content = "Converted markdown"
        mock_md = MagicMock()
        mock_md.convert_stream.return_value = mock_result

        monkeypatch.setattr("ebooklib.epub.read_epub", lambda *a, **kw: mock_book)
        monkeypatch.setattr("markitdown.MarkItDown", lambda: mock_md)

        _, chapters = convert_epub_by_chapters("test.epub")

        assert len(chapters) == 3
        assert [c.index for c in chapters] == [0, 1, 2]


class TestEpubConverterConvert:
    """Tests for EpubConverter.convert_epub_to_markdown method."""

    def test_convert_success(self, monkeypatch) -> None:
        """Successful conversion returns markdown."""
        mock_result = MagicMock()
        mock_result.text_content = "# Recipe\n\nIngredients and instructions here."

        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        converter = EpubConverter()
        converter.md = mock_md

        result = converter.convert_epub_to_markdown("test.epub")

        assert result == "# Recipe\n\nIngredients and instructions here."
        mock_md.convert.assert_called_once_with("test.epub")

    def test_convert_minimal_content_warns(self, monkeypatch, caplog) -> None:
        """Warns when conversion produces minimal content."""
        import logging

        mock_result = MagicMock()
        mock_result.text_content = "tiny"  # Less than 100 chars

        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        converter = EpubConverter()
        converter.md = mock_md

        with caplog.at_level(logging.WARNING):
            result = converter.convert_epub_to_markdown("test.epub")

        assert result == "tiny"
        assert "minimal content" in caplog.text

    def test_convert_failure_raises(self, monkeypatch) -> None:
        """Conversion failure is re-raised."""
        mock_md = MagicMock()
        mock_md.convert.side_effect = ValueError("Conversion failed")

        converter = EpubConverter()
        converter.md = mock_md

        with pytest.raises(ValueError, match="Conversion failed"):
            converter.convert_epub_to_markdown("test.epub")
