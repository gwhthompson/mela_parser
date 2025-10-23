import sys
from html.parser import HTMLParser
from pathlib import Path

import ebooklib
from ebooklib import epub
from ebooklib.utils import debug


class TitleExtractor(HTMLParser):
    """Extract title from HTML content"""

    def __init__(self):
        super().__init__()
        self.in_title = False
        self.in_h1 = False
        self.title = None
        self.h1 = None

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag == "h1":
            self.in_h1 = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "h1":
            self.in_h1 = False

    def handle_data(self, data):
        if self.in_title and not self.title:
            self.title = data.strip()
        elif self.in_h1 and not self.h1:
            self.h1 = data.strip()


class LinkExtractor(HTMLParser):
    """Extract all links from HTML content"""

    def __init__(self):
        super().__init__()
        self.links = []
        self.current_link = None
        self.current_link_text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            if "href" in attrs_dict:
                self.current_link = {
                    "href": attrs_dict["href"],
                    "class": attrs_dict.get("class", ""),
                    "id": attrs_dict.get("id", ""),
                    "text": "",
                }
                self.current_link_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self.current_link:
            self.current_link["text"] = "".join(self.current_link_text).strip()
            self.links.append(self.current_link)
            self.current_link = None
            self.current_link_text = []

    def handle_data(self, data):
        if self.current_link is not None:
            self.current_link_text.append(data)


def extract_title_from_html(content):
    """Extract title from HTML content"""
    try:
        parser = TitleExtractor()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        parser.feed(content)
        return parser.h1 or parser.title or "No title found"
    except Exception as e:
        return f"Error extracting title: {e}"


def extract_links_from_html(content):
    """Extract all links from HTML content"""
    try:
        parser = LinkExtractor()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        parser.feed(content)
        return parser.links
    except Exception:
        return []


def print_object_details(obj, name="Object", indent=0):
    """Print detailed information about an object"""
    prefix = "  " * indent
    print(f"\n{prefix}{'=' * 60}")
    print(f"{prefix}{name}: {type(obj).__name__}")
    print(f"{prefix}{'=' * 60}")

    # Get all attributes that don't start with underscore
    attrs = [attr for attr in dir(obj) if not attr.startswith("_")]

    for attr in attrs:
        try:
            value = getattr(obj, attr)
            # Skip methods and content/data attributes to avoid printing binary data
            if not callable(value) and attr not in ["content", "data"]:
                print(f"{prefix}  {attr}: {value}")
        except Exception as e:
            print(f"{prefix}  {attr}: <Error accessing: {e}>")
    print()


def print_toc_item(item, name="TOC Item", indent=0):
    """Print TOC item, handling both Link objects and tuples"""
    prefix = "  " * indent

    if isinstance(item, tuple):
        # Tuple format is usually (Link, [children])
        print(f"\n{prefix}{'=' * 60}")
        print(f"{prefix}{name}: Tuple (Section with subsections)")
        print(f"{prefix}{'=' * 60}")

        if len(item) > 0:
            print(f"{prefix}Link:")
            link = item[0]
            print(f"{prefix}  title: {getattr(link, 'title', 'N/A')}")
            print(f"{prefix}  href: {getattr(link, 'href', 'N/A')}")
            print(f"{prefix}  uid: {getattr(link, 'uid', 'N/A')}")

        if len(item) > 1 and item[1]:
            print(f"{prefix}Subsections ({len(item[1])} items):")
            for i, subitem in enumerate(item[1]):
                print_toc_item(subitem, f"Subsection {i + 1}", indent + 1)
    else:
        # It's a Link object
        print(f"\n{prefix}{'=' * 60}")
        print(f"{prefix}{name}: Link")
        print(f"{prefix}{'=' * 60}")
        print(f"{prefix}  title: {getattr(item, 'title', 'N/A')}")
        print(f"{prefix}  href: {getattr(item, 'href', 'N/A')}")
        print(f"{prefix}  uid: {getattr(item, 'uid', 'N/A')}")
        print()


def print_toc_tree(items, prefix=""):
    """Print TOC as a tree structure with all nesting levels"""
    if not items:
        return

    for idx, item in enumerate(items):
        is_last_item = idx == len(items) - 1

        # Tree characters
        if is_last_item:
            current_prefix = prefix + "└── "
            child_prefix = prefix + "    "
        else:
            current_prefix = prefix + "├── "
            child_prefix = prefix + "│   "

        if isinstance(item, tuple):
            # Tuple format: (Link, [children])
            link = item[0]
            children = item[1] if len(item) > 1 else []

            title = getattr(link, "title", "N/A")
            href = getattr(link, "href", "N/A")
            uid = getattr(link, "uid", "N/A")

            print(f"{current_prefix}{title}")
            print(f"{child_prefix.replace('├', '│').replace('└', ' ')}    href: {href}")
            print(f"{child_prefix.replace('├', '│').replace('└', ' ')}    uid: {uid}")

            if children:
                print_toc_tree(children, child_prefix)
        else:
            # It's a Link object
            title = getattr(item, "title", "N/A")
            href = getattr(item, "href", "N/A")
            uid = getattr(item, "uid", "N/A")

            print(f"{current_prefix}{title}")
            print(f"{child_prefix.replace('├', '│').replace('└', ' ')}    href: {href}")
            print(f"{child_prefix.replace('├', '│').replace('└', ' ')}    uid: {uid}")


# Setup directories (relative to script location)
script_dir = Path(__file__).parent
input_dir = script_dir / "input"
output_dir = script_dir / "output"
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Find all epub files in input directory
epub_files = list(input_dir.glob("*.epub"))

if not epub_files:
    print(f"No .epub files found in {input_dir}")
    print(f"Please place your epub files in the {input_dir} directory")
else:
    print(f"Found {len(epub_files)} epub file(s) to process\n")

    for input_file in epub_files:
        print(f"Processing: {input_file.name}...")

        input_name = input_file.stem  # Gets filename without extension
        output_file = output_dir / f"{input_name}_debug.txt"

        # Redirect debug output to file
        with open(output_file, "w") as f:
            sys.stdout = f

            try:
                book = epub.read_epub(str(input_file))

                print(f"DEBUG OUTPUT FOR: {input_file.name}")
                print("=" * 80)

                print("\n\nMETADATA:")
                debug(book.metadata)

                print("\n\nSPINE:")
                debug(book.spine)

                print("\n\nTABLE OF CONTENTS:")
                for i, item in enumerate(book.toc):
                    print_toc_item(item, f"TOC Item {i + 1}")

                print("\n\nTABLE OF CONTENTS - TREE VIEW:")
                print("=" * 80)
                print("Complete hierarchical structure of all TOC links")
                print("=" * 80)
                print()
                print_toc_tree(book.toc)

                print("\n\nGUIDE:")
                if hasattr(book, "guide") and book.guide:
                    for i, item in enumerate(book.guide):
                        print(f"\nGuide Item {i + 1}:")
                        print(f"  type: {getattr(item, 'type', 'N/A')}")
                        print(f"  title: {getattr(item, 'title', 'N/A')}")
                        print(f"  href: {getattr(item, 'href', 'N/A')}")
                else:
                    print("No guide information available")

                print("\n\nALL ITEMS SUMMARY:")
                print("=" * 80)
                all_items = list(book.get_items())
                print(f"Total items in book: {len(all_items)}")
                print("=" * 80)

                for i, item in enumerate(all_items, 1):
                    item_type = type(item).__name__
                    file_name = getattr(item, "file_name", getattr(item, "id", "Unknown"))

                    # Try to extract title for documents
                    title = "N/A"
                    if hasattr(item, "content") and item.get_type() == ebooklib.ITEM_DOCUMENT:
                        title = extract_title_from_html(item.content)

                    print(f"\n[{i}] {item_type}")
                    print(f"    File: {file_name}")
                    print(f"    Title: {title}")
                    print(f"    ID: {getattr(item, 'id', 'N/A')}")

                print("\n\nNAVIGATION ITEMS:")
                nav_items = list(book.get_items_of_type(ebooklib.ITEM_NAVIGATION))
                if nav_items:
                    print(f"Found {len(nav_items)} navigation item(s)")
                    for i, nav in enumerate(nav_items, 1):
                        print(f"\n{'=' * 60}")
                        print(f"Navigation Item {i}")
                        print(f"{'=' * 60}")
                        print(f"  file_name: {getattr(nav, 'file_name', 'N/A')}")
                        print(f"  id: {getattr(nav, 'id', 'N/A')}")
                        print(f"  media_type: {getattr(nav, 'media_type', 'N/A')}")

                        # Extract links from navigation content
                        nav_links = extract_links_from_html(nav.content)
                        if nav_links:
                            print(f"\n  Links in navigation ({len(nav_links)}):")
                            for link_idx, link in enumerate(nav_links, 1):
                                print(f"\n    [{link_idx}] {link['text']}")
                                print(f"        href: {link['href']}")
                                if link["class"]:
                                    print(f"        class: {link['class']}")
                else:
                    print("No ITEM_NAVIGATION items found in this book")

                print("\n\nIMAGES (Detailed):")
                for i, x in enumerate(book.get_items_of_type(ebooklib.ITEM_IMAGE)):
                    print_object_details(x, f"Image {i + 1}")

                print("\n\nDOCUMENTS (Detailed):")
                for i, x in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
                    title = extract_title_from_html(x.content)
                    print(f"\n{'=' * 60}")
                    print(f"Document {i + 1}: {title}")
                    print(f"{'=' * 60}")
                    print(f"  file_name: {getattr(x, 'file_name', 'N/A')}")
                    print(f"  id: {getattr(x, 'id', 'N/A')}")
                    print(f"  media_type: {getattr(x, 'media_type', 'N/A')}")

                print("\n\nALL HYPERLINKS IN BOOK:")
                print("=" * 80)
                print("Every <a> tag extracted from all HTML documents")
                print("=" * 80)

                all_links = []
                documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

                for doc_idx, doc in enumerate(documents, 1):
                    doc_title = extract_title_from_html(doc.content)
                    doc_filename = getattr(doc, "file_name", "Unknown")
                    links = extract_links_from_html(doc.content)

                    if links:
                        print(f"\n--- Document {doc_idx}: {doc_title} ---")
                        print(f"    Source: {doc_filename}")
                        print(f"    Links found: {len(links)}")

                        for link_idx, link in enumerate(links, 1):
                            all_links.append(
                                {
                                    "doc_num": doc_idx,
                                    "doc_title": doc_title,
                                    "doc_file": doc_filename,
                                    **link,
                                }
                            )
                            print(f"\n    [{link_idx}] {link['text']}")
                            print(f"        href: {link['href']}")
                            if link["class"]:
                                print(f"        class: {link['class']}")
                            if link["id"]:
                                print(f"        id: {link['id']}")

                print(f"\n\n{'=' * 80}")
                print(f"TOTAL LINKS FOUND: {len(all_links)}")
                print(f"{'=' * 80}")

            except Exception as e:
                print(f"ERROR processing {input_file.name}: {e}")

            sys.stdout = sys.__stdout__

        print(f"  ✓ Saved to: {output_file}")

    print(f"\nAll files processed. Output saved to {output_dir}")
