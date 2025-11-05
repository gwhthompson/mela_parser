#!/usr/bin/env python3
"""CLI for mela-parser: Extract recipes from EPUB cookbooks to Mela format.

This module provides the command-line interface for extracting recipes from EPUB
cookbooks using chapter-based extraction with parallel async processing for
maximum speed and efficiency.

The CLI supports various OpenAI models and allows customization of output
directories and image extraction options.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import time
import uuid
import warnings
from io import BytesIO, StringIO
from pathlib import Path

import ebooklib
from ebooklib import epub
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .chapter_extractor import AsyncChapterExtractor, Chapter
from .recipe import RecipeProcessor

# Suppress python-dotenv warnings during markitdown import
with contextlib.redirect_stderr(StringIO()):
    from markitdown import MarkItDown

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
warnings.filterwarnings("ignore", category=UserWarning, module="pydub")

# Create global Rich console for styled output
console = Console()


def setup_logging(log_file: str = "mela_parser.log") -> None:
    """Set up logging configuration for the application.

    Configures logging to output detailed logs to a file only.
    Console output is handled separately via Rich for better visual presentation.

    Args:
        log_file: Path to the log file. Defaults to "mela_parser.log".
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w")],
    )


def convert_epub_by_chapters(epub_path: str) -> tuple[epub.EpubBook, list[tuple[str, str]]]:
    """Convert each EPUB chapter to markdown.

    Reads an EPUB file and converts each document item (chapter) to markdown
    format using MarkItDown for LLM-friendly text processing.

    Args:
        epub_path: Path to the EPUB file to convert.

    Returns:
        A tuple containing:
            - EpubBook object with metadata and images
            - List of tuples, each containing (chapter_name, markdown_content)

    Raises:
        FileNotFoundError: If the EPUB file doesn't exist.
        Exception: If the EPUB file is corrupted or cannot be read.
    """
    book = epub.read_epub(epub_path, {"ignore_ncx": True})
    md = MarkItDown()

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    chapters: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting chapters to markdown", total=len(items))

        for item in items:
            html_content = item.get_content()
            result = md.convert_stream(BytesIO(html_content), file_extension=".html")
            markdown_content = result.text_content

            chapter_name = item.get_name()
            chapters.append((chapter_name, markdown_content))
            progress.update(task, advance=1)

    logging.info(f"Converted {len(chapters)} chapters to markdown")
    console.print(f"[green]✓[/green] Converted {len(chapters)} chapters to markdown")
    return book, chapters


async def main_async() -> None:
    """Main async function for parallel recipe extraction processing.

    This is the core async function that orchestrates the entire extraction
    workflow:
    1. Parse command-line arguments
    2. Convert EPUB chapters to markdown
    3. Extract recipes from chapters in parallel
    4. Deduplicate recipes
    5. Write recipes to output directory
    6. Create .melarecipes archive

    The function uses high concurrency (200 parallel requests) for optimal
    performance with modern LLM APIs.

    Raises:
        FileNotFoundError: If the EPUB file doesn't exist.
        Exception: If any step in the extraction pipeline fails.
    """
    parser = argparse.ArgumentParser(
        description="Extract recipes from EPUB cookbooks to Mela format", prog="mela-parse"
    )
    parser.add_argument("epub_path", type=str, help="Path to EPUB cookbook file")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5-nano", "gpt-5-mini"],
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory (default: output)"
    )
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")

    args = parser.parse_args()

    if not os.path.exists(args.epub_path):
        console.print()
        console.print(
            Panel(
                f"[bold red]File not found:[/bold red]\n{args.epub_path}\n\n"
                f"[dim]Please check the file path and try again.[/dim]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        console.print()
        parser.exit(1)

    setup_logging()

    start_time = time.time()

    # Get metadata
    book_temp = epub.read_epub(args.epub_path, {"ignore_ncx": True})
    book_title = book_temp.get_metadata("DC", "title")[0][0]
    book_slug = RecipeProcessor.slugify(book_title)

    # Log to file
    logging.info("=" * 80)
    logging.info(f"Processing: {book_title}")
    logging.info("Method: Chapter-based extraction")
    logging.info("=" * 80)

    # Display to console with Rich
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{book_title}[/bold cyan]\n[dim]Method: Chapter-based extraction[/dim]",
            title="[bold]Mela Recipe Parser[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    # PHASE 1: Convert chapters
    logging.info("\nPHASE 1: Converting chapters")
    console.print(
        Panel("[bold]Phase 1:[/bold] Converting chapters to markdown", border_style="blue")
    )
    book, chapters = convert_epub_by_chapters(args.epub_path)

    # PHASE 2: Extract from all chapters
    logging.info("\nPHASE 2: Extracting from chapters (PARALLEL)")
    console.print()
    console.print(
        Panel(
            "[bold]Phase 2:[/bold] Extracting recipes from chapters (parallel)", border_style="blue"
        )
    )

    # Convert to Chapter objects
    chapter_objs = [
        Chapter(name=name, content=content, index=i) for i, (name, content) in enumerate(chapters)
    ]

    # Extract recipes with high concurrency and progress tracking
    extractor = AsyncChapterExtractor(model=args.model)

    # Create a mapping of chapter names to original Chapter objects for retries
    chapter_map = {chapter.name: chapter for chapter in chapter_objs}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        extract_task = progress.add_task(
            "Extracting recipes from chapters", total=len(chapter_objs)
        )

        # Use a wrapper to track progress
        extraction_results = []
        failed_chapter_names = []
        retry_count = 0

        try:
            # First attempt
            results = await extractor.extract_from_chapters(
                chapters=chapter_objs, expected_titles=None, max_concurrent=200
            )

            # Check for failures
            for result in results:
                if (
                    result.recipes
                    or result.chapter_name.startswith("nav")
                    or "toc" in result.chapter_name.lower()
                ):
                    extraction_results.append(result)
                else:
                    failed_chapter_names.append(result.chapter_name)
                progress.update(extract_task, advance=1)

            # Auto-retry failed chapters (max 2 retries)
            if failed_chapter_names and retry_count < 2:
                retry_count += 1
                fail_count = len(failed_chapter_names)
                attempt_num = retry_count + 1
                progress.update(
                    extract_task,
                    description=(
                        f"[yellow]Retrying {fail_count} failed chapters "
                        f"(attempt {attempt_num})[/yellow]"
                    ),
                )
                console.print(
                    f"[yellow]⚠[/yellow] Retrying {fail_count} chapters that had no recipes..."
                )
                logging.info(f"Retrying {fail_count} failed chapters (attempt {attempt_num})")

                # Retry failed chapters using original Chapter objects
                retry_chapters = [
                    chapter_map[name] for name in failed_chapter_names if name in chapter_map
                ]
                retry_results = await extractor.extract_from_chapters(
                    chapters=retry_chapters, expected_titles=None, max_concurrent=200
                )

                # Add successful retries to results
                for retry_result in retry_results:
                    if retry_result.recipes:
                        extraction_results.append(retry_result)
                        console.print(
                            f"[green]✓[/green] Retry successful: {retry_result.chapter_name}"
                        )
                        logging.info(f"Retry successful: {retry_result.chapter_name}")

        except Exception as e:
            console.print(f"[red]✗[/red] Error during extraction: {e}")
            logging.error(f"Error during extraction: {e}")
            raise

    # Flatten all recipes from all chapters
    all_recipes = []
    for result in extraction_results:
        all_recipes.extend(result.recipes)
        if result.recipes:
            console.print(
                f"  [green]✓[/green] {result.chapter_name}: {len(result.recipes)} recipes"
            )
            logging.info(f"{result.chapter_name}: ✓ {len(result.recipes)} recipes")
        else:
            console.print(f"  [dim]{result.chapter_name}[/dim]")
            logging.info(f"{result.chapter_name}")

    console.print()
    console.print(
        f"[bold green]✓[/bold green] Total extracted: [bold]{len(all_recipes)}[/bold] recipes"
    )
    logging.info(f"\nTotal extracted: {len(all_recipes)} recipes")

    # PHASE 3: Deduplication
    logging.info("\nPHASE 3: Deduplication")
    console.print()
    console.print(Panel("[bold]Phase 3:[/bold] Removing duplicate recipes", border_style="blue"))

    seen: set[str] = set()
    unique_recipes = []
    duplicates = 0

    for recipe in all_recipes:
        if recipe.title not in seen:
            seen.add(recipe.title)
            unique_recipes.append(recipe)
        else:
            duplicates += 1
            logging.debug(f"  Duplicate: {recipe.title}")

    if duplicates > 0:
        console.print(f"[yellow]⚠[/yellow] Removed {duplicates} duplicate(s)")
    console.print(f"[green]✓[/green] {len(all_recipes)} → {len(unique_recipes)} unique recipes")
    logging.info(f"Total: {len(all_recipes)} → Unique: {len(unique_recipes)}")

    # PHASE 4: Write recipes
    logging.info("\nPHASE 4: Writing recipes")
    console.print()
    console.print(Panel("[bold]Phase 4:[/bold] Writing recipes to disk", border_style="blue"))

    out_dir = Path(args.output_dir) / book_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = RecipeProcessor(args.epub_path, book)
    written = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        write_task = progress.add_task("Writing recipes", total=len(unique_recipes))

        for recipe in unique_recipes:
            recipe_dict = processor._mela_recipe_to_object(recipe)
            recipe_dict["link"] = book_title
            recipe_dict["id"] = str(uuid.uuid4())

            # Pass through image paths from extracted recipe for processing
            if recipe.images:
                recipe_dict["images"] = recipe.images
                processor._process_images(recipe_dict)
            else:
                recipe_dict["images"] = []

            if processor.write_recipe(recipe_dict, output_dir=str(out_dir)):
                written += 1
            progress.update(write_task, advance=1)

    console.print(f"[green]✓[/green] Wrote {written} recipes to {out_dir}")

    # Create archive
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Creating .melarecipes archive", total=None)
        archive_zip = shutil.make_archive(
            base_name=str(out_dir), format="zip", root_dir=str(out_dir)
        )
        archive_mela = archive_zip.replace(".zip", ".melarecipes")
        os.rename(archive_zip, archive_mela)

    console.print(f"[green]✓[/green] Created archive: {archive_mela}")

    # Summary
    elapsed_time = time.time() - start_time

    # Format time as minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_formatted = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    logging.info("\n" + "=" * 80)
    logging.info("COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Chapters: {len(chapters)}")
    logging.info(f"Extracted: {len(all_recipes)}")
    logging.info(f"Unique: {len(unique_recipes)}")
    logging.info(f"Written: {written}")
    logging.info(f"Time: {elapsed_time:.1f}s")
    logging.info(f"Output: {archive_mela}")

    # Create summary table
    console.print()
    summary_table = Table(
        title="[bold green]✓ Processing Complete[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Chapters Processed", str(len(chapters)))
    summary_table.add_row("Recipes Extracted", str(len(all_recipes)))
    summary_table.add_row("Unique Recipes", str(len(unique_recipes)))
    summary_table.add_row("Recipes Written", str(written))
    summary_table.add_row("Processing Time", time_formatted)

    console.print(summary_table)
    console.print()
    console.print(f"[bold]Output:[/bold] [cyan]{archive_mela}[/cyan]")
    console.print()


def main() -> None:
    """Entry point for mela-parse CLI command.

    This function serves as the synchronous entry point that launches the async
    main function. It's called when running 'mela-parse' from the command line.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                "[yellow]Processing interrupted by user[/yellow]\n\n"
                "[dim]Partial results may be available in the output directory.[/dim]",
                title="[bold yellow]Interrupted[/bold yellow]",
                border_style="yellow",
            )
        )
        console.print()
    except Exception as e:
        console.print()
        console.print(
            Panel(
                f"[bold red]An unexpected error occurred:[/bold red]\n\n"
                f"{e!s}\n\n"
                f"[dim]Check mela_parser.log for detailed error information.[/dim]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        console.print()
        logging.exception("Unexpected error during processing")
        raise


if __name__ == "__main__":
    main()
