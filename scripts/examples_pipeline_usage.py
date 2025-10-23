#!/usr/bin/env python3
"""
Example usage of the LLM-powered recipe extraction pipeline v2.

This file demonstrates:
1. Basic usage (iterative mode)
2. Single-pass extraction
3. Custom prompt library
4. Batch processing
5. Integration with image processing
"""
import asyncio
import json
from pathlib import Path

from main_chapters_v2 import (
    ExtractionPipeline,
    PromptLibrary,
    write_recipes_to_disk,
)
from ebooklib import epub


# ============================================================================
# EXAMPLE 1: Basic Iterative Extraction
# ============================================================================


async def example_iterative_extraction():
    """
    Run iterative extraction with automatic prompt improvement.

    This will:
    - Extract recipes with default prompts
    - Validate against discovered list
    - Improve prompts if not 100% match
    - Retry until perfect or max iterations
    """
    print("="*80)
    print("EXAMPLE 1: Iterative Extraction with Automatic Prompt Improvement")
    print("="*80)

    epub_path = "examples/input/jerusalem.epub"
    output_dir = "output/example1"

    pipeline = ExtractionPipeline(max_concurrent_chapters=5)

    # Run iterative refinement
    final_recipes, final_prompts, history = await pipeline.iterative_refinement(
        epub_path=epub_path,
        max_iterations=10,
        model="gpt-5-nano",
        output_dir=output_dir,
    )

    print(f"\nExtracted {len(final_recipes)} recipes")
    print(f"Prompt version: {final_prompts.version}")
    print(f"Iterations: {len(history)}")

    # Show iteration progress
    for record in history:
        print(f"  Iteration {record['iteration']}: {record['match_percentage']:.1f}% match")

    # Save final prompts for reuse
    with open(Path(output_dir) / "final_prompts.json", "w") as f:
        json.dump(final_prompts.to_dict(), f, indent=2)

    print(f"\nFinal prompts saved to: {output_dir}/final_prompts.json")


# ============================================================================
# EXAMPLE 2: Single-Pass Extraction
# ============================================================================


async def example_single_pass():
    """
    Run single-pass extraction without iteration.

    Use this when:
    - You have pre-tuned prompts
    - You want quick results
    - Cost optimization is critical
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Single-Pass Extraction (No Iteration)")
    print("="*80)

    epub_path = "examples/input/jerusalem.epub"
    output_dir = "output/example2"

    pipeline = ExtractionPipeline()

    # Use default prompts
    prompts = PromptLibrary.default()

    # Extract once
    result, chapters, discovered = await pipeline.extract_recipes(
        epub_path=epub_path,
        prompts=prompts,
        model="gpt-5-nano",
    )

    print(f"\nExtracted {len(result.recipes)} recipes")
    print(f"Extraction time: {result.extraction_time:.1f}s")
    print(f"Chapters processed: {result.chapters_processed}")

    # Validate if we have discovered list
    if discovered:
        validation = pipeline.validate_extraction(result, discovered)
        print(f"Match rate: {validation.match_percentage:.1f}%")
        print(f"Missing: {len(validation.missing_titles)}")
        print(f"Extra: {len(validation.extra_titles)}")


# ============================================================================
# EXAMPLE 3: Using Custom Prompt Library
# ============================================================================


async def example_custom_prompts():
    """
    Use custom or pre-tuned prompts from a previous run.

    This allows you to:
    - Reuse successful prompts on similar cookbooks
    - Skip iteration for known cookbook styles
    - Maintain consistency across extractions
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Using Custom Prompt Library")
    print("="*80)

    epub_path = "examples/input/jerusalem.epub"
    prompt_file = "output/example1/final_prompts.json"
    output_dir = "output/example3"

    # Load custom prompts
    if Path(prompt_file).exists():
        with open(prompt_file, "r") as f:
            prompts = PromptLibrary.from_dict(json.load(f))
        print(f"Loaded prompts from: {prompt_file}")
        print(f"Prompt version: {prompts.version}")
    else:
        print(f"Prompt file not found: {prompt_file}")
        print("Using default prompts instead")
        prompts = PromptLibrary.default()

    pipeline = ExtractionPipeline()

    # Extract with custom prompts
    result, chapters, discovered = await pipeline.extract_recipes(
        epub_path=epub_path,
        prompts=prompts,
        model="gpt-5-nano",
    )

    print(f"\nExtracted {len(result.recipes)} recipes using custom prompts")


# ============================================================================
# EXAMPLE 4: Batch Processing Multiple EPUBs
# ============================================================================


async def example_batch_processing():
    """
    Process multiple EPUBs in batch.

    Useful for:
    - Processing entire cookbook collections
    - Comparing extraction across different books
    - Building prompt libraries for different styles
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing Multiple EPUBs")
    print("="*80)

    input_dir = Path("examples/input")
    output_base = Path("output/batch")

    epub_files = list(input_dir.glob("*.epub"))
    print(f"Found {len(epub_files)} EPUB files")

    pipeline = ExtractionPipeline()
    prompts = PromptLibrary.default()

    results_summary = []

    for epub_path in epub_files[:2]:  # Limit to 2 for demo
        book_name = epub_path.stem
        print(f"\nProcessing: {book_name}")

        output_dir = output_base / book_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract
        result, chapters, discovered = await pipeline.extract_recipes(
            epub_path=str(epub_path),
            prompts=prompts,
            model="gpt-5-nano",
        )

        # Validate
        if discovered:
            validation = pipeline.validate_extraction(result, discovered)
            match_rate = validation.match_percentage
        else:
            match_rate = None

        # Record
        results_summary.append({
            "book": book_name,
            "recipes": len(result.recipes),
            "match_rate": match_rate,
            "time": result.extraction_time,
        })

        print(f"  Extracted: {len(result.recipes)} recipes")
        if match_rate:
            print(f"  Match rate: {match_rate:.1f}%")

    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    for record in results_summary:
        print(f"{record['book']}: {record['recipes']} recipes, "
              f"{record['match_rate']:.1f}% match" if record['match_rate'] else "no validation")


# ============================================================================
# EXAMPLE 5: Custom Gap Analysis
# ============================================================================


class CustomPipeline(ExtractionPipeline):
    """
    Custom pipeline with domain-specific gap analysis rules.

    Extend this to add:
    - Cookbook-specific patterns
    - Custom validation logic
    - Specialized prompt improvements
    """

    async def analyze_gaps(self, validation_report, chapters, prompts):
        """Add custom analysis rules."""
        # Call parent analysis first
        improvements = await super().analyze_gaps(validation_report, chapters, prompts)

        # Custom rule: Check for multi-page recipes
        missing_titles = list(validation_report.missing_titles)
        if any("continued" in title.lower() for title in missing_titles):
            improvements.suggested_extraction_changes += (
                "\n\nCUSTOM RULE: Skip recipes with '(continued)' suffix. "
                "They are multi-page splits of previous recipes."
            )
            improvements.missing_recipe_patterns.append("Multi-page recipe continuations")

        # Custom rule: Check for recipe variations
        if any("variation" in title.lower() for title in missing_titles):
            improvements.suggested_extraction_changes += (
                "\n\nCUSTOM RULE: Extract recipe variations as separate recipes. "
                "Look for 'Variation:', 'Alternative:', or 'Also try:' sections."
            )

        return improvements


async def example_custom_gap_analysis():
    """
    Use custom pipeline with specialized gap analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Gap Analysis")
    print("="*80)

    epub_path = "examples/input/jerusalem.epub"
    output_dir = "output/example5"

    # Use custom pipeline
    pipeline = CustomPipeline(max_concurrent_chapters=5)

    # Run iterative refinement with custom analyzer
    final_recipes, final_prompts, history = await pipeline.iterative_refinement(
        epub_path=epub_path,
        max_iterations=5,
        model="gpt-5-nano",
        output_dir=output_dir,
    )

    print(f"\nExtracted {len(final_recipes)} recipes with custom analysis")


# ============================================================================
# EXAMPLE 6: Monitoring and Metrics
# ============================================================================


async def example_monitoring():
    """
    Extract with detailed monitoring and metrics.

    Track:
    - Extraction speed per chapter
    - API usage
    - Iteration convergence
    - Prompt effectiveness
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Monitoring and Metrics")
    print("="*80)

    epub_path = "examples/input/jerusalem.epub"
    output_dir = "output/example6"

    pipeline = ExtractionPipeline()

    # Run with metrics collection
    final_recipes, final_prompts, history = await pipeline.iterative_refinement(
        epub_path=epub_path,
        max_iterations=10,
        model="gpt-5-nano",
        output_dir=output_dir,
    )

    # Analyze convergence
    print("\nIteration Convergence:")
    print(f"{'Iter':<6} {'Match %':<10} {'Missing':<10} {'Extra':<10} {'Time (s)':<10}")
    print("-" * 50)

    for record in history:
        print(f"{record['iteration']:<6} "
              f"{record['match_percentage']:<10.1f} "
              f"{record['missing']:<10} "
              f"{record['extra']:<10} "
              f"{record['extraction_time']:<10.1f}")

    # Calculate metrics
    if len(history) > 1:
        improvement_per_iteration = (
            (history[-1]['match_percentage'] - history[0]['match_percentage']) /
            len(history)
        )
        print(f"\nAverage improvement per iteration: {improvement_per_iteration:.2f}%")

    iterations_to_90 = next(
        (i for i, h in enumerate(history, 1) if h['match_percentage'] >= 90),
        None
    )
    if iterations_to_90:
        print(f"Iterations to reach 90%: {iterations_to_90}")


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Run all examples."""
    # Example 1: Iterative extraction
    await example_iterative_extraction()

    # Example 2: Single-pass
    await example_single_pass()

    # Example 3: Custom prompts
    await example_custom_prompts()

    # Example 4: Batch processing
    # await example_batch_processing()  # Commented out - can be slow

    # Example 5: Custom gap analysis
    # await example_custom_gap_analysis()  # Commented out - requires API

    # Example 6: Monitoring
    # await example_monitoring()  # Commented out - requires API

    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)


if __name__ == "__main__":
    # Run examples
    # asyncio.run(main())

    # For quick testing, just show what each example does
    print("Available examples:")
    print("1. example_iterative_extraction() - Full iterative pipeline")
    print("2. example_single_pass() - Quick single extraction")
    print("3. example_custom_prompts() - Use pre-tuned prompts")
    print("4. example_batch_processing() - Process multiple EPUBs")
    print("5. example_custom_gap_analysis() - Custom analysis rules")
    print("6. example_monitoring() - Detailed metrics and monitoring")
    print("\nUncomment asyncio.run(main()) to run examples")
