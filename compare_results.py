#!/usr/bin/env python3
"""
Compare before/after results to validate TDD improvements.
"""
import re
from pathlib import Path


def parse_log_results(log_file: str) -> dict:
    """Extract metrics from log file."""
    with open(log_file, 'r') as f:
        content = f.read()

    results = {}

    # Extract chunks processed
    chunks_match = re.search(r'Chunks processed: (\d+)', content)
    results['chunks'] = int(chunks_match.group(1)) if chunks_match else 0

    # Extract total extracted (with duplicates)
    total_match = re.search(r'Total recipes extracted: (\d+) \(with duplicates\)', content)
    results['total_extracted'] = int(total_match.group(1)) if total_match else 0

    # Extract unique recipes
    unique_match = re.search(r'Unique recipes: (\d+)', content)
    results['unique_recipes'] = int(unique_match.group(1)) if unique_match else 0

    # Extract written
    written_match = re.search(r'Recipes written: (\d+)', content)
    results['written'] = int(written_match.group(1)) if written_match else 0

    # Extract success rate
    rate_match = re.search(r'Success rate: ([\d.]+)%', content)
    results['success_rate'] = float(rate_match.group(1)) if rate_match else 0.0

    # Extract time
    time_match = re.search(r'Total time: ([\d.]+)s', content)
    results['time_seconds'] = float(time_match.group(1)) if time_match else 0.0

    # Check for filtered incomplete recipes
    filtered_match = re.search(r'Filtered out (\d+) incomplete recipes', content)
    results['filtered_incomplete'] = int(filtered_match.group(1)) if filtered_match else 0

    return results


def compare_results(before_file: str, after_file: str, book_name: str):
    """Compare before and after results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {book_name}")
    print(f"{'='*80}\n")

    before = parse_log_results(before_file)
    after = parse_log_results(after_file)

    print(f"{'Metric':<30} | {'Before':<12} | {'After':<12} | {'Change':<15}")
    print(f"{'-'*30} | {'-'*12} | {'-'*12} | {'-'*15}")

    # Chunks
    print(f"{'Chunks processed':<30} | {before['chunks']:<12} | {after['chunks']:<12} | {after['chunks'] - before['chunks']:+d}")

    # Total extracted
    print(f"{'Total extracted (w/ dups)':<30} | {before['total_extracted']:<12} | {after['total_extracted']:<12} | {after['total_extracted'] - before['total_extracted']:+d}")

    # Unique recipes
    print(f"{'Unique recipes':<30} | {before['unique_recipes']:<12} | {after['unique_recipes']:<12} | {after['unique_recipes'] - before['unique_recipes']:+d}")

    # Filtered incomplete
    if after['filtered_incomplete'] > 0:
        print(f"{'Filtered incomplete':<30} | {before.get('filtered_incomplete', 0):<12} | {after['filtered_incomplete']:<12} | New feature")

    # Written
    print(f"{'Recipes written':<30} | {before['written']:<12} | {after['written']:<12} | {after['written'] - before['written']:+d}")

    # Success rate
    print(f"{'Success rate':<30} | {before['success_rate']:.1f}%{' ':<8} | {after['success_rate']:.1f}%{' ':<8} | {after['success_rate'] - before['success_rate']:+.1f}%")

    # Time
    before_min = before['time_seconds'] / 60
    after_min = after['time_seconds'] / 60
    print(f"{'Processing time':<30} | {before_min:.1f} min{' ':<6} | {after_min:.1f} min{' ':<6} | {after_min - before_min:+.1f} min")

    # Summary
    print(f"\n{'SUMMARY':<30}")
    print(f"{'-'*30}")

    if after['written'] > before['written']:
        print(f"✅ More recipes extracted: +{after['written'] - before['written']}")
    elif after['written'] < before['written']:
        print(f"⚠️  Fewer recipes extracted: {after['written'] - before['written']}")
    else:
        print(f"→ Same number of recipes extracted")

    if after['success_rate'] > before['success_rate']:
        print(f"✅ Higher success rate: +{after['success_rate'] - before['success_rate']:.1f}%")
    elif after['success_rate'] < before['success_rate']:
        print(f"⚠️  Lower success rate: {after['success_rate'] - before['success_rate']:.1f}%")
    else:
        print(f"→ Same success rate")

    if after['filtered_incomplete'] > 0:
        print(f"✅ Now filtering {after['filtered_incomplete']} incomplete recipes (quality improvement)")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python compare_results.py <before_log> <after_log> <book_name>")
        print("\nExample:")
        print("  python compare_results.py jerusalem_overlap_test.log jerusalem_improved_test.log 'Jerusalem'")
        sys.exit(1)

    before_file = sys.argv[1]
    after_file = sys.argv[2]
    book_name = sys.argv[3]

    compare_results(before_file, after_file, book_name)
