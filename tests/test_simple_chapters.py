#!/usr/bin/env python3
"""
TDD tests for main_simple_chapters.py
Tests extraction accuracy against expected recipe lists.
"""
import subprocess
import json
from pathlib import Path


def load_expected_recipes(filename: str) -> list[str]:
    """Load expected recipe list."""
    path = Path("examples/output/recipe-lists") / filename
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_extraction(epub_path: str) -> dict:
    """Run extraction and return results."""
    result = subprocess.run(
        ["uv", "run", "python", "main_simple_chapters.py", epub_path, "--model", "gpt-5-nano"],
        capture_output=True,
        text=True
    )

    # Parse output for metrics
    output = result.stdout

    # Extract counts from log
    import re
    extracted_match = re.search(r'Extracted: (\d+)', output)
    written_match = re.search(r'Written: (\d+)', output)
    match_match = re.search(r'Match: (\d+)/(\d+)', output)

    return {
        "exit_code": result.returncode,
        "extracted": int(extracted_match.group(1)) if extracted_match else 0,
        "written": int(written_match.group(1)) if written_match else 0,
        "matched": int(match_match.group(1)) if match_match else 0,
        "expected": int(match_match.group(2)) if match_match else 0,
        "output": output
    }


def test_jerusalem_exact_count():
    """Jerusalem must extract and write exactly 125 recipes."""
    expected = load_expected_recipes("jerusalem-recipe-list.txt")
    assert len(expected) == 125, "Ground truth should have 125 recipes"

    results = run_extraction("examples/input/jerusalem.epub")

    assert results["exit_code"] == 0, "Extraction should complete successfully"
    assert results["written"] == 125, (
        f"Must write exactly 125 recipes, got {results['written']}. "
        f"Difference: {results['written'] - 125} (extras if positive, missing if negative)"
    )


def test_modern_way_exact_count():
    """Modern Way must extract and write exactly 142 recipes."""
    expected = load_expected_recipes("a-modern-way-to-eat-recipe-list.txt")
    assert len(expected) == 142, "Ground truth should have 142 recipes"

    results = run_extraction("examples/input/a-modern-way-to-eat.epub")

    assert results["exit_code"] == 0, "Extraction should complete successfully"
    assert results["written"] == 142, (
        f"Must write exactly 142 recipes, got {results['written']}. "
        f"Difference: {results['written'] - 142}"
    )


def test_completely_perfect_exact_count():
    """Completely Perfect must extract and write exactly 122 recipes."""
    expected = load_expected_recipes("completely-perfect-recipe-list.txt")
    assert len(expected) == 122, "Ground truth should have 122 recipes"

    results = run_extraction("examples/input/completely-perfect.epub")

    assert results["exit_code"] == 0, "Extraction should complete successfully"
    assert results["written"] == 122, (
        f"Must write exactly 122 recipes, got {results['written']}. "
        f"Difference: {results['written'] - 122}"
    )


def test_no_component_recipes_extracted():
    """System must NOT extract component recipes (For the X, SAUCE, etc.)."""
    results = run_extraction("examples/input/jerusalem.epub")

    # Read actual written files
    output_dir = Path("output/jerusalem-simple-chapters")
    if not output_dir.exists():
        pytest.fail("Output directory not found")

    recipe_files = list(output_dir.glob("*.melarecipe"))

    # Check for known component recipes
    component_patterns = [
        "for-the-",
        "-sauce.melarecipe",
        "-marinade.melarecipe",
        "crumble.melarecipe",  # Standalone component
        "cream.melarecipe",     # Standalone component
    ]

    found_components = []
    for recipe_file in recipe_files:
        filename = recipe_file.name
        if any(pattern in filename for pattern in component_patterns):
            found_components.append(filename)

    assert len(found_components) == 0, (
        f"Found {len(found_components)} component recipes that should be excluded: "
        f"{found_components[:10]}"
    )


def test_toc_discovery_completeness():
    """TOC discovery must find all recipes listed in TOC/Index."""
    # Run and parse discovery phase output
    results = run_extraction("examples/input/jerusalem.epub")

    # Check discovered count
    import re
    discovered_match = re.search(r'Discovered (\d+) recipes', results["output"])
    discovered = int(discovered_match.group(1)) if discovered_match else 0

    expected = load_expected_recipes("jerusalem-recipe-list.txt")

    assert discovered == len(expected), (
        f"TOC discovery must find all {len(expected)} recipes, found {discovered}. "
        f"Missing: {len(expected) - discovered}"
    )


if __name__ == "__main__":
    # Run tests manually for TDD
    import sys

    print("=== TDD Test Suite for Simple Chapter Extraction ===\n")

    tests = [
        ("Jerusalem exact count (125)", test_jerusalem_exact_count),
        ("Modern Way exact count (142)", test_modern_way_exact_count),
        ("Completely Perfect exact count (122)", test_completely_perfect_exact_count),
        ("No component recipes", test_no_component_recipes_extracted),
        ("TOC discovery completeness", test_toc_discovery_completeness),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"Running: {name}...", end=" ")
            test_func()
            print("✅ PASS")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL")
            print(f"  {e}\n")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {e}\n")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)
