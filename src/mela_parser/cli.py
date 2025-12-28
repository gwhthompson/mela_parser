#!/usr/bin/env python3
"""CLI for mela-parser: Extract recipes from EPUB cookbooks to Mela format.

This module provides the command-line interface for extracting recipes from EPUB
cookbooks using chapter-based extraction with parallel async processing.

The CLI is responsible for:
- Argument parsing
- Progress display (Rich UI)
- Error presentation
- Calling the pipeline for business logic

The actual extraction logic is delegated to the pipeline architecture.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, cast

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

from .config import ExtractionConfig
from .pipeline import PipelineContext, create_default_pipeline
from .repository import slugify
from .services import ServiceFactory

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


def create_progress() -> Progress:
    """Create a Rich progress bar with standard configuration."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
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
    return parser.parse_args()


def display_header(book_title: str) -> None:
    """Display the header panel with book information."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{book_title}[/bold cyan]\n[dim]Method: Pipeline extraction[/dim]",
            title="[bold]Mela Recipe Parser[/bold]",
            border_style="cyan",
        )
    )
    console.print()


def display_phase(phase_num: int, description: str) -> None:
    """Display a phase header."""
    console.print()
    console.print(Panel(f"[bold]Phase {phase_num}:[/bold] {description}", border_style="blue"))


def display_summary(
    chapters_count: int,
    extracted_count: int,
    unique_count: int,
    written_count: int,
    elapsed_time: float,
    archive_path: str,
) -> None:
    """Display the completion summary table."""
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_formatted = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    console.print()
    summary_table = Table(
        title="[bold green]✓ Processing Complete[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Chapters Processed", str(chapters_count))
    summary_table.add_row("Recipes Extracted", str(extracted_count))
    summary_table.add_row("Unique Recipes", str(unique_count))
    summary_table.add_row("Recipes Written", str(written_count))
    summary_table.add_row("Processing Time", time_formatted)

    console.print(summary_table)
    console.print()
    console.print(f"[bold]Output:[/bold] [cyan]{archive_path}[/cyan]")
    console.print()


def display_error(title: str, message: str) -> None:
    """Display an error panel."""
    console.print()
    console.print(
        Panel(
            message,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        )
    )
    console.print()


async def main_async() -> None:
    """Main async function for parallel recipe extraction processing.

    Orchestrates the extraction workflow using the pipeline architecture:
    1. Parse arguments and validate input
    2. Run pipeline stages (conversion, extraction, deduplication, images, persistence)
    3. Create archive and display summary

    The actual business logic is delegated to the pipeline stages.
    """
    args = parse_args()
    epub_path = Path(args.epub_path)

    # Validate input file exists
    if not epub_path.exists():
        display_error(
            "Error",
            f"[bold red]File not found:[/bold red]\n{args.epub_path}\n\n"
            f"[dim]Please check the file path and try again.[/dim]",
        )
        raise SystemExit(1)

    setup_logging()
    start_time = time.time()

    # Create configuration from args
    config = ExtractionConfig(
        model=args.model,
        extract_images=not args.no_images,
    )

    # Create service factory and pipeline
    factory = ServiceFactory(config=config)
    pipeline = create_default_pipeline()

    # Create pipeline context
    output_dir = Path(args.output_dir)
    ctx = PipelineContext(
        epub_path=epub_path,
        output_dir=output_dir,
        config=config,
    )

    # Get book title for display (before pipeline runs)
    # ebooklib has no type stubs, suppress partial type warnings
    from ebooklib import epub as epub_lib

    book_temp = epub_lib.read_epub(str(epub_path), {"ignore_ncx": True})  # pyright: ignore[reportUnknownMemberType]
    metadata: list[tuple[Any, ...]] = cast(
        list[tuple[Any, ...]],
        book_temp.get_metadata("DC", "title"),  # pyright: ignore[reportUnknownMemberType]
    )
    book_title: str = str(metadata[0][0])
    book_slug = slugify(book_title)

    # Update output_dir to include book slug (so persistence writes to output/{book}/)
    ctx.output_dir = output_dir / book_slug

    # Log start
    logging.info("=" * 80)
    logging.info(f"Processing: {book_title}")
    logging.info("Method: Pipeline extraction")
    logging.info("=" * 80)

    display_header(book_title)

    # Run pipeline stages with progress tracking
    for i, stage in enumerate(pipeline.stages, 1):
        display_phase(i, stage.name)
        logging.info(f"\nPHASE {i}: {stage.name}")

        with create_progress() as progress:
            task = progress.add_task(f"{stage.name}...", total=None)
            await stage.execute(ctx, factory)
            progress.update(task, completed=1, total=1)

        # Display stage-specific results
        if stage.name == "Conversion":
            console.print(f"[green]✓[/green] Converted {len(ctx.chapters)} chapters")
        elif stage.name == "Extraction":
            total_recipes = len(ctx.recipes)
            console.print(f"[green]✓[/green] Extracted {total_recipes} recipes")
        elif stage.name == "Deduplication":
            console.print(f"[green]✓[/green] {len(ctx.unique_recipes)} unique recipes")
        elif stage.name == "Images":
            with_images = sum(1 for r in ctx.unique_recipes if r.images)
            console.print(f"[green]✓[/green] Processed images for {with_images} recipes")
        elif stage.name == "Persistence":
            console.print(f"[green]✓[/green] Wrote recipes to {ctx.output_dir}")

    # Create archive
    display_phase(len(pipeline.stages) + 1, "Creating archive")
    out_dir = ctx.output_dir

    with create_progress() as progress:
        progress.add_task("Creating .melarecipes archive", total=None)
        archive_zip = shutil.make_archive(
            base_name=str(out_dir), format="zip", root_dir=str(out_dir)
        )
        archive_mela = archive_zip.replace(".zip", ".melarecipes")
        os.rename(archive_zip, archive_mela)

    console.print(f"[green]✓[/green] Created archive: {archive_mela}")

    # Log completion
    elapsed_time = time.time() - start_time
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Chapters: {len(ctx.chapters)}")
    logging.info(f"Extracted: {len(ctx.recipes)}")
    logging.info(f"Unique: {len(ctx.unique_recipes)}")
    logging.info(f"Time: {elapsed_time:.1f}s")
    logging.info(f"Output: {archive_mela}")

    # Display summary
    display_summary(
        chapters_count=len(ctx.chapters),
        extracted_count=len(ctx.recipes),
        unique_count=len(ctx.unique_recipes),
        written_count=len(ctx.unique_recipes),
        elapsed_time=elapsed_time,
        archive_path=archive_mela,
    )


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
    except Exception as e:  # Intentional catch-all for CLI entry point
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
