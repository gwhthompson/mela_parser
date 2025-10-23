#!/usr/bin/env python3
"""
CLI script to extract recipes from EPUB cookbooks.

Uses list-based extraction pipeline:
1. Extract recipe list from TOC (StructuredListExtractor)
2. Extract each recipe content via MarkItDown + OpenAI
3. Save as .melarecipe files

Usage:
    python scripts/extract_recipes.py <epub_path> [output_dir] [--batch-size N]

Examples:
    python scripts/extract_recipes.py examples/input/jerusalem.epub output/jerusalem/
    python scripts/extract_recipes.py examples/input/simple.epub output/simple/ --batch-size 20
"""
import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mela_parser.extractors.list_based_extractor import ListBasedRecipeExtractor


async def main():
    parser = argparse.ArgumentParser(
        description="Extract recipes from EPUB cookbooks using list-based approach"
    )
    parser.add_argument("epub_path", help="Path to EPUB cookbook file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output",
        help="Output directory for .melarecipe files (default: output/)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of recipes to process concurrently (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate input
    epub_path = Path(args.epub_path)
    if not epub_path.exists():
        print(f"Error: EPUB file not found: {epub_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"List-Based Recipe Extraction")
    print(f"{'='*70}")
    print(f"EPUB: {epub_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

    # Extract recipes
    extractor = ListBasedRecipeExtractor(str(epub_path))
    result = await extractor.extract_all_recipes(args.output_dir, args.batch_size)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Extraction Complete!")
    print(f"{'='*70}")
    print(f"Total recipes in TOC: {result.total}")
    print(f"Successfully extracted: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Success rate: {result.successful/result.total*100:.1f}%")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

    # Show failures if any
    if result.failed > 0:
        print("Failed recipes:")
        for r in result.results:
            if not r.success:
                print(f"  - {r.recipe_link.title}: {r.error}")

    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
