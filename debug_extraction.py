#!/usr/bin/env python3
"""Debug script to test recipe extraction from EPUB files."""

import os
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from src.mela_parser.extractors.structured_list import CandidateLink, StructuredListExtractor

# Disable OpenAI calls for debugging
os.environ["OPENAI_API_KEY"] = "dummy"


def debug_epub(epub_file: str):
    """Debug extraction from a single EPUB."""
    print(f"\n{'=' * 60}")
    print(f"Debugging: {epub_file}")
    print("=" * 60)

    epub_path = Path("examples/input") / epub_file
    if not epub_path.exists():
        print(f"File not found: {epub_path}")
        return

    book = epub.read_epub(str(epub_path))
    extractor = StructuredListExtractor()

    # Try to find recipe list pages
    print("\n1. Looking for recipe list pages...")
    list_pages = extractor.find_recipe_list_pages(book)
    print(f"   Found {len(list_pages)} list pages")

    # If no list pages found, check navigation/TOC
    all_links = []
    if not list_pages:
        print("\n2. No dedicated list pages found, checking navigation...")
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if "nav" in item.file_name.lower() or "toc" in item.file_name.lower():
                print(f"   Checking: {item.file_name}")
                content = item.get_content().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(content, "html.parser")

                links = soup.find_all("a", href=True)
                print(f"   Found {len(links)} links")

                # Show first 10 links as sample
                for i, link in enumerate(links[:10]):
                    href = link.get("href", "")
                    text = link.get_text(strip=True)
                    print(f"      {i + 1}. '{text}' -> {href}")

                # Convert to CandidateLinks
                for link in links:
                    href = link.get("href", "")
                    text = link.get_text(strip=True)
                    fragment = None

                    if "#" in href:
                        href_parts = href.split("#")
                        href = href_parts[0]
                        fragment = href_parts[1] if len(href_parts) > 1 else None

                    if text and href:
                        all_links.append(
                            CandidateLink(
                                title=text,
                                href=href,
                                fragment=fragment,
                                source_page=item.file_name,
                                css_class=link.get("class", [""])[0] if link.get("class") else "",
                            )
                        )
                break
    else:
        # Extract from list pages
        print("\n2. Extracting links from list pages...")
        for page in list_pages:
            links = extractor.extract_links_from_page(page)
            all_links.extend(links)

    print(f"\n3. Total links extracted: {len(all_links)}")

    # Apply structural filters
    print("\n4. Applying structural filters...")
    filtered = extractor.apply_structural_filters(all_links)
    print(f"   Candidates: {len(filtered.candidates)}")
    print(f"   Excluded: {len(filtered.excluded)}")

    # Show some examples
    print("\n5. Sample candidates:")
    for i, candidate in enumerate(filtered.candidates[:10]):
        print(f"   {i + 1}. '{candidate.title}' [{candidate.href}]")

    print("\n6. Sample excluded (if any):")
    for i, excluded in enumerate(filtered.excluded[:10]):
        print(f"   {i + 1}. '{excluded.title}' (reason: likely page number or too short)")

    return filtered.candidates


if __name__ == "__main__":
    # Test each EPUB
    epubs = ["a-modern-way-to-eat.epub", "completely-perfect.epub", "jerusalem.epub", "simple.epub"]

    for epub_file in epubs:
        candidates = debug_epub(epub_file)
        if candidates:
            print(f"\nFinal candidate count for {epub_file}: {len(candidates)}")
