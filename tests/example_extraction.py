#!/usr/bin/env python3
"""Test recipe extraction with a small sample."""

import logging

from mela_parser.parse import CookbookParser

logging.basicConfig(level=logging.INFO)

# Generic sample containing 2 clear recipes
sample_markdown = """
# Sample Cookbook

## Breakfast

### Simple Porridge

A warming start to the day.

SERVES 2

100g oats
400ml water or milk
pinch of salt
honey to serve

Put the oats in a pan with the liquid and salt. Bring to a simmer, stirring often.
Cook for 4-5 minutes until thick and creamy. Serve with honey.

### Scrambled Eggs

Quick and satisfying.

SERVES 2

4 eggs
knob of butter
salt and pepper
toast to serve

Beat the eggs with seasoning. Melt butter in a pan over low heat.
Add eggs and stir gently until just set. Serve on toast.
"""

print("Testing with sample markdown...")
print(f"Sample length: {len(sample_markdown)} chars")
print()

# Test with gpt-5-nano
parser_nano = CookbookParser(model="gpt-5-nano")
try:
    result_nano = parser_nano.parse_cookbook(sample_markdown, "Test Book")
    print(f"\nGPT-5-NANO extracted {len(result_nano.recipes)} recipes:")
    for r in result_nano.recipes:
        print(f"  - {r.title}")
except Exception as e:
    print(f"\nGPT-5-NANO failed: {e}")

print("\n" + "=" * 80 + "\n")

# Test with gpt-5-mini
parser_mini = CookbookParser(model="gpt-5-mini")
try:
    result_mini = parser_mini.parse_cookbook(sample_markdown, "Test Book")
    print(f"\nGPT-5-MINI extracted {len(result_mini.recipes)} recipes:")
    for r in result_mini.recipes:
        print(f"  - {r.title}")
except Exception as e:
    print(f"\nGPT-5-MINI failed: {e}")
