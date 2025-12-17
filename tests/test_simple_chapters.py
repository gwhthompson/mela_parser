#!/usr/bin/env python3
"""
Tests for recipe extraction from EPUB files.
Tests run against any EPUB files found in examples/input/.
"""

import re
import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples/input")


def get_available_epubs() -> list[Path]:
    """Find all EPUB files in examples/input/."""
    if not EXAMPLES_DIR.exists():
        return []
    return list(EXAMPLES_DIR.glob("*.epub"))


def run_extraction(epub_path: Path) -> dict:
    """Run extraction and return results."""
    result = subprocess.run(
        ["uv", "run", "mela-parse", str(epub_path), "--model", "gpt-5-nano"],
        capture_output=True,
        text=True,
    )

    output = result.stdout

    extracted_match = re.search(r"Extracted: (\d+)", output)
    written_match = re.search(r"Written: (\d+)", output)

    return {
        "exit_code": result.returncode,
        "extracted": int(extracted_match.group(1)) if extracted_match else 0,
        "written": int(written_match.group(1)) if written_match else 0,
        "output": output,
        "stderr": result.stderr,
    }


# Skip all tests if no EPUBs available
pytestmark = pytest.mark.skipif(
    len(get_available_epubs()) == 0,
    reason="No EPUB files in examples/input/. Add your own EPUBs to run tests.",
)


@pytest.fixture
def sample_epub() -> Path:
    """Return the first available EPUB for testing."""
    epubs = get_available_epubs()
    if not epubs:
        pytest.skip("No EPUB files available")
    return epubs[0]


class TestExtraction:
    """Tests for basic extraction functionality."""

    def test_extraction_completes(self, sample_epub: Path):
        """Extraction should complete without errors."""
        results = run_extraction(sample_epub)
        assert results["exit_code"] == 0, f"Extraction failed: {results['stderr']}"

    def test_extracts_recipes(self, sample_epub: Path):
        """Extraction should find at least one recipe."""
        results = run_extraction(sample_epub)
        assert results["extracted"] > 0, "Should extract at least one recipe"

    def test_writes_recipes(self, sample_epub: Path):
        """Extraction should write recipe files."""
        results = run_extraction(sample_epub)
        assert results["written"] > 0, "Should write at least one recipe"

    def test_output_files_created(self, sample_epub: Path):
        """Recipe files should be created in output directory."""
        run_extraction(sample_epub)

        output_dir = Path("output") / sample_epub.stem
        if not output_dir.exists():
            pytest.skip("Output directory not created")

        recipe_files = list(output_dir.glob("*.melarecipe"))
        assert len(recipe_files) > 0, "Should create .melarecipe files"


class TestRecipeQuality:
    """Tests for recipe quality and filtering."""

    def test_no_component_recipes(self, sample_epub: Path):
        """Should not extract standalone component recipes."""
        run_extraction(sample_epub)

        output_dir = Path("output") / sample_epub.stem
        if not output_dir.exists():
            pytest.skip("Output directory not created")

        recipe_files = list(output_dir.glob("*.melarecipe"))

        component_patterns = [
            "for-the-",
            "-sauce.melarecipe",
            "-marinade.melarecipe",
        ]

        found_components = [
            f.name for f in recipe_files if any(p in f.name.lower() for p in component_patterns)
        ]

        assert len(found_components) == 0, (
            f"Found component recipes that should be excluded: {found_components[:5]}"
        )


@pytest.mark.parametrize("epub_path", get_available_epubs(), ids=lambda p: p.stem)
def test_each_epub_extracts(epub_path: Path):
    """Each available EPUB should extract successfully."""
    results = run_extraction(epub_path)
    assert results["exit_code"] == 0, f"Failed to extract {epub_path.name}"
    assert results["written"] > 0, f"No recipes written from {epub_path.name}"
