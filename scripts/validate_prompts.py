#!/usr/bin/env python3
"""
Prompt Validation Script
========================

Validates prompt performance against ground truth recipe lists.
Calculates accuracy metrics and suggests improvements.

Usage:
    python validate_prompts.py --epub examples/jerusalem.epub --ground-truth examples/output/recipe-lists/jerusalem-recipe-list.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from mela_parser.prompt_library import (
    PromptLibrary,
    PromptType,
    PerformanceMetrics,
)


def load_ground_truth(file_path: str) -> Set[str]:
    """
    Load ground truth recipe list from file.

    Args:
        file_path: Path to recipe list file

    Returns:
        Set of recipe titles (normalized)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse numbered list format: "  123→Recipe Title"
    titles = set()
    for line in lines:
        line = line.strip()
        if '→' in line:
            # Split on arrow, take right side
            title = line.split('→', 1)[1].strip()
            if title:  # Skip empty lines
                titles.add(title)

    return titles


def calculate_validation_metrics(
    extracted_titles: Set[str],
    expected_titles: Set[str],
) -> Tuple[PerformanceMetrics, Dict[str, List[str]]]:
    """
    Calculate validation metrics comparing extracted vs. expected.

    Args:
        extracted_titles: Set of titles extracted by the system
        expected_titles: Set of titles from ground truth

    Returns:
        Tuple of (PerformanceMetrics, detailed_breakdown)
    """
    # Exact matches
    correct = extracted_titles & expected_titles

    # Missing (expected but not extracted)
    missing = expected_titles - extracted_titles

    # Extra (extracted but not expected)
    extra = extracted_titles - expected_titles

    # Title modifications (case-insensitive near matches)
    # This catches "Recipe Name" vs "recipe name" or punctuation changes
    modifications = []

    # For each missing title, check if there's a similar extracted title
    for exp_title in missing:
        exp_normalized = exp_title.lower().replace(" ", "").replace("-", "")
        for ext_title in extra:
            ext_normalized = ext_title.lower().replace(" ", "").replace("-", "")
            if exp_normalized == ext_normalized:
                modifications.append(f"Expected: '{exp_title}' | Got: '{ext_title}'")

    # Create metrics
    metrics = PerformanceMetrics(
        total_recipes=len(extracted_titles),
        expected_recipes=len(expected_titles),
        correctly_extracted=len(correct),
        missing_recipes=len(missing),
        extra_recipes=len(extra),
        title_modifications=len(modifications),
    )
    metrics.calculate()

    # Detailed breakdown
    breakdown = {
        "correct": sorted(list(correct)),
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra)),
        "modifications": modifications,
    }

    return metrics, breakdown


def print_validation_report(
    metrics: PerformanceMetrics,
    breakdown: Dict[str, List[str]],
    verbose: bool = False,
) -> None:
    """Print formatted validation report."""
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Expected recipes:      {metrics.expected_recipes}")
    print(f"  Extracted recipes:     {metrics.total_recipes}")
    print(f"  Correctly extracted:   {metrics.correctly_extracted}")
    print(f"  Missing:               {metrics.missing_recipes}")
    print(f"  Extra:                 {metrics.extra_recipes}")
    print(f"  Title modifications:   {metrics.title_modifications}")

    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy:   {metrics.accuracy_percent:.2f}%")
    print(f"  Precision:  {metrics.precision_percent:.2f}%")
    print(f"  Recall:     {metrics.recall_percent:.2f}%")

    # Success or failure
    if metrics.accuracy_percent == 100.0 and metrics.extra_recipes == 0:
        print(f"\n{'🎉 SUCCESS! 100% ACCURACY ACHIEVED!' :^80}")
    else:
        print(f"\n{'⚠️  VALIDATION FAILED - Improvements needed' :^80}")

    # Detailed breakdowns
    if verbose or metrics.missing_recipes > 0:
        print(f"\nMissing Recipes ({len(breakdown['missing'])}):")
        for i, title in enumerate(breakdown['missing'][:20], 1):  # Show max 20
            print(f"  {i}. {title}")
        if len(breakdown['missing']) > 20:
            print(f"  ... and {len(breakdown['missing']) - 20} more")

    if verbose or metrics.extra_recipes > 0:
        print(f"\nExtra Recipes ({len(breakdown['extra'])}):")
        for i, title in enumerate(breakdown['extra'][:20], 1):
            print(f"  {i}. {title}")
        if len(breakdown['extra']) > 20:
            print(f"  ... and {len(breakdown['extra']) - 20} more")

    if metrics.title_modifications > 0:
        print(f"\nTitle Modifications ({len(breakdown['modifications'])}):")
        for i, mod in enumerate(breakdown['modifications'][:10], 1):
            print(f"  {i}. {mod}")

    print("\n" + "=" * 80)


def suggest_improvements(
    metrics: PerformanceMetrics,
    breakdown: Dict[str, List[str]],
) -> None:
    """Suggest prompt improvements based on failure patterns."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 80)

    if metrics.missing_recipes > 0:
        print(f"\n1. MISSING RECIPES ({metrics.missing_recipes} recipes)")
        print("   Possible causes:")
        print("   - Recipes are incomplete (missing ingredients or instructions)")
        print("   - Recipes span multiple pages (continuations)")
        print("   - Prompt is too conservative in completeness checks")
        print("\n   Recommended actions:")
        print("   - Review sample missing recipes manually")
        print("   - Check if they have all three components (title, ingredients, instructions)")
        print("   - If complete, relax 'minimum requirements' in prompt")
        print("   - Add examples showing edge cases that should be extracted")

    if metrics.extra_recipes > 0:
        print(f"\n2. EXTRA RECIPES ({metrics.extra_recipes} recipes)")
        print("   Possible causes:")
        print("   - Extracting recipe teasers or previews")
        print("   - Extracting recipe continuations as separate recipes")
        print("   - Extracting section headers as recipes")
        print("\n   Recommended actions:")
        print("   - Review sample extra recipes manually")
        print("   - Strengthen exclusion rules for incomplete recipes")
        print("   - Add examples of recipe teasers that should NOT be extracted")
        print("   - Add continuation detection (titles with 'continued', '(cont)')")

    if metrics.title_modifications > 0:
        print(f"\n3. TITLE MODIFICATIONS ({metrics.title_modifications} modifications)")
        print("   Possible causes:")
        print("   - Model standardizing punctuation (&→and, apostrophes)")
        print("   - Model changing capitalization")
        print("   - Model adding labels or categories")
        print("\n   Recommended actions:")
        print("   - Review examples of modified titles")
        print("   - Strengthen 'NEVER modify' constraints in prompt")
        print("   - Add more examples showing exact punctuation preservation")
        print("   - Add quality check: 'Title matches source character-by-character'")

    if metrics.accuracy_percent == 100.0 and metrics.extra_recipes == 0:
        print("\n🎉 No improvements needed - prompt is performing perfectly!")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate prompt performance")
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth recipe list file"
    )
    parser.add_argument(
        "--extracted",
        help="Path to file with extracted recipe titles (one per line)"
    )
    parser.add_argument(
        "--prompt-version",
        default="1.1.0",
        help="Prompt version to validate (default: 1.1.0)"
    )
    parser.add_argument(
        "--prompt-type",
        default="chapter_extraction",
        choices=["recipe_list_discovery", "chapter_extraction"],
        help="Type of prompt to validate"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed breakdown even if no errors"
    )
    parser.add_argument(
        "--update-library",
        action="store_true",
        help="Update prompt library with these metrics"
    )

    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from: {args.ground_truth}")
    expected_titles = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(expected_titles)} expected recipes")

    # Load extracted titles
    if args.extracted:
        print(f"Loading extracted titles from: {args.extracted}")
        with open(args.extracted, 'r', encoding='utf-8') as f:
            extracted_titles = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(extracted_titles)} extracted recipes")
    else:
        print("\nNo --extracted file provided.")
        print("This script will show expected counts and exit.")
        print(f"\nExpected recipes: {len(expected_titles)}")
        print("\nTo validate extraction results, run:")
        print(f"  python validate_prompts.py --ground-truth {args.ground_truth} --extracted your_results.txt")
        return

    # Calculate metrics
    metrics, breakdown = calculate_validation_metrics(extracted_titles, expected_titles)

    # Print report
    print_validation_report(metrics, breakdown, args.verbose)

    # Suggest improvements
    suggest_improvements(metrics, breakdown)

    # Update library if requested
    if args.update_library:
        library = PromptLibrary()
        prompt_type = PromptType(args.prompt_type)

        library.update_performance(
            prompt_type,
            args.prompt_version,
            metrics,
        )

        print(f"\n✓ Updated prompt library with metrics for {prompt_type.value} v{args.prompt_version}")
        print(f"  Saved to: {library.library_path}")


if __name__ == "__main__":
    main()
