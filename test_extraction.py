#!/usr/bin/env python3
"""Test recipe extraction with a small sample."""

import logging
from parse import CookbookParser

logging.basicConfig(level=logging.INFO)

# Sample containing 2 clear recipes
sample_markdown = """
# A Modern Way to Eat

## Breakfasts

### Overnight Bircher with peaches

Weekday breakfasts for me are usually two bleary minutes before I run out of the door.

SERVES 2

100g oats
2 tablespoons white chia seeds
1 tablespoon pumpkin seeds
350ml milk of your choice (I use almond or coconut)
1 tablespoon maple syrup
a dash of all-natural vanilla extract
a little squeeze of lemon juice
2 ripe peaches

The night before, put the oats, chia seeds and pumpkin seeds into a bowl or container, pour over the milk, maple syrup, vanilla and lemon juice and stir well.

In the morning, chop the peaches into little chunks, squeeze over a little more lemon and either layer into two glasses with the Bircher or just stir through.

###Turkish fried eggs

This is a really good weekend breakfast, easily quick enough to squeeze in on weekdays too.

SERVES 2

4 tablespoons Greek yoghurt
a good pinch of sea salt
a good knob of butter
4 organic or free-range eggs
2 wholemeal pittas or flatbreads
1 teaspoon Turkish chilli flakes
a good pinch of sumac
a few sprigs of fresh mint, parsley and dill, leaves picked and chopped

Mix the yoghurt and salt in a bowl and leave to one side.

Heat the butter in a large non-stick frying pan on a medium heat. Allow it to begin to brown, then crack in the eggs and cook to your liking.

Once your eggs are ready, quickly toast your pittas or flatbreads then top with a good spoonful of yoghurt, then your eggs.

Sprinkle over the chilli flakes, sumac and herbs and eat straight away.
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

print("\n" + "="*80 + "\n")

# Test with gpt-5-mini
parser_mini = CookbookParser(model="gpt-5-mini")
try:
    result_mini = parser_mini.parse_cookbook(sample_markdown, "Test Book")
    print(f"\nGPT-5-MINI extracted {len(result_mini.recipes)} recipes:")
    for r in result_mini.recipes:
        print(f"  - {r.title}")
except Exception as e:
    print(f"\nGPT-5-MINI failed: {e}")
