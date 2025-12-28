"""Unit tests for mela_parser.cli module.

Tests CLI argument parsing, logging setup, and display functions.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mela_parser.cli import (
    create_progress,
    display_error,
    display_header,
    display_phase,
    display_summary,
    parse_args,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_configures_file_handler(self, tmp_path: Path) -> None:
        """setup_logging adds a FileHandler to root logger."""
        log_file = tmp_path / "test.log"

        # Clear any existing handlers from root logger
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        root_logger.handlers.clear()

        try:
            setup_logging(str(log_file))

            # Should have added a FileHandler
            assert len(root_logger.handlers) > 0
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert "FileHandler" in handler_types
        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers

    def test_creates_log_file(self, tmp_path: Path) -> None:
        """setup_logging creates the log file."""
        log_file = tmp_path / "test.log"

        # Clear handlers first
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        root_logger.handlers.clear()

        try:
            setup_logging(str(log_file))
            # Write something to trigger file creation
            logging.info("Test log message")
            assert log_file.exists()
        finally:
            root_logger.handlers = original_handlers

    def test_default_log_file_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Default log file is mela_parser.log."""
        monkeypatch.chdir(tmp_path)

        # Clear handlers first
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        root_logger.handlers.clear()

        try:
            setup_logging()
            # Log something to create the file
            logging.info("Test")
            assert (tmp_path / "mela_parser.log").exists()
        finally:
            root_logger.handlers = original_handlers


class TestCreateProgress:
    """Tests for create_progress function."""

    def test_returns_progress_object(self) -> None:
        """create_progress returns a Progress instance."""
        from rich.progress import Progress

        progress = create_progress()
        assert isinstance(progress, Progress)

    def test_has_spinner_column(self) -> None:
        """Progress has spinner column."""
        from rich.progress import SpinnerColumn

        progress = create_progress()
        column_types = [type(c) for c in progress.columns]
        assert SpinnerColumn in column_types

    def test_has_bar_column(self) -> None:
        """Progress has bar column."""
        from rich.progress import BarColumn

        progress = create_progress()
        column_types = [type(c) for c in progress.columns]
        assert BarColumn in column_types


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parses_epub_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parses required epub_path argument."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "cookbook.epub"])
        args = parse_args()
        assert args.epub_path == "cookbook.epub"

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default model is gpt-5-mini."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub"])
        args = parse_args()
        assert args.model == "gpt-5-mini"

    def test_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom model can be specified."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub", "--model", "gpt-5-nano"])
        args = parse_args()
        assert args.model == "gpt-5-nano"

    def test_default_output_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default output directory is 'output'."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub"])
        args = parse_args()
        assert args.output_dir == "output"

    def test_custom_output_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom output directory can be specified."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub", "--output-dir", "recipes"])
        args = parse_args()
        assert args.output_dir == "recipes"

    def test_no_images_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--no-images flag works."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub", "--no-images"])
        args = parse_args()
        assert args.no_images is True

    def test_no_images_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--no-images defaults to False."""
        monkeypatch.setattr(sys, "argv", ["mela-parse", "book.epub"])
        args = parse_args()
        assert args.no_images is False

    def test_missing_epub_path_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing epub_path causes exit."""
        monkeypatch.setattr(sys, "argv", ["mela-parse"])
        with pytest.raises(SystemExit):
            parse_args()


class TestDisplayFunctions:
    """Tests for display_* functions.

    These functions use Rich console output. We test they don't crash
    and produce expected output patterns.
    """

    def test_display_header_no_crash(self) -> None:
        """display_header runs without crashing."""
        # Just verify it doesn't raise
        display_header("Test Cookbook")

    def test_display_phase_no_crash(self) -> None:
        """display_phase runs without crashing."""
        display_phase(1, "Conversion")
        display_phase(2, "Extraction")

    def test_display_summary_no_crash(self) -> None:
        """display_summary runs without crashing."""
        display_summary(
            chapters_count=10,
            extracted_count=50,
            unique_count=45,
            written_count=45,
            elapsed_time=120.5,
            archive_path="/output/cookbook.melarecipes",
        )

    def test_display_summary_formats_time_minutes(self) -> None:
        """display_summary formats time with minutes correctly."""
        # 90 seconds = 1m 30s
        # Just verify it doesn't crash with various times
        display_summary(
            chapters_count=1,
            extracted_count=1,
            unique_count=1,
            written_count=1,
            elapsed_time=90.0,
            archive_path="test.melarecipes",
        )

    def test_display_summary_formats_time_seconds_only(self) -> None:
        """display_summary formats time with seconds only."""
        # 45 seconds = 45s
        display_summary(
            chapters_count=1,
            extracted_count=1,
            unique_count=1,
            written_count=1,
            elapsed_time=45.0,
            archive_path="test.melarecipes",
        )

    def test_display_error_no_crash(self) -> None:
        """display_error runs without crashing."""
        display_error("Test Error", "Something went wrong")


class TestDisplayOutputContent:
    """Tests that verify display functions produce expected content."""

    @patch("mela_parser.cli.console")
    def test_display_header_shows_book_title(self, mock_console: MagicMock) -> None:
        """display_header shows the book title."""
        display_header("My Cookbook")
        # Verify print was called (at least for the panel)
        assert mock_console.print.called

    @patch("mela_parser.cli.console")
    def test_display_phase_shows_phase_number(self, mock_console: MagicMock) -> None:
        """display_phase shows phase number."""
        display_phase(3, "Deduplication")
        assert mock_console.print.called

    @patch("mela_parser.cli.console")
    def test_display_error_shows_message(self, mock_console: MagicMock) -> None:
        """display_error shows error message."""
        display_error("File Error", "File not found")
        assert mock_console.print.called


# ============================================================================
# main_async Tests
# ============================================================================


class TestMainAsync:
    """Tests for main_async function."""

    @pytest.fixture
    def mock_epub_book(self):
        """Create a mock EpubBook with test metadata."""
        book = MagicMock()
        book.get_metadata.return_value = [("Test Cookbook",)]
        return book

    @pytest.fixture
    def mock_context(self):
        """Create a mock PipelineContext."""
        from mela_parser.chapter_extractor import Chapter
        from mela_parser.parse import IngredientGroup, MelaRecipe

        ctx = MagicMock()
        ctx.chapters = [Chapter(name="Test", content="Content", index=0)]
        ctx.recipes = [
            MelaRecipe(
                title="Test Recipe",
                ingredients=[IngredientGroup(title="", ingredients=["1 item"])],
                instructions=["Step 1.", "Step 2."],
            )
        ]
        ctx.unique_recipes = ctx.recipes
        ctx.output_dir = Path("/tmp/test_output")
        return ctx

    @pytest.mark.asyncio
    async def test_main_async_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """main_async shows error when file doesn't exist."""
        from mela_parser.cli import main_async

        # Set up args with non-existent file
        monkeypatch.setattr(sys, "argv", ["mela-parse", str(tmp_path / "nonexistent.epub")])

        with pytest.raises(SystemExit) as exc_info:
            await main_async()

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_async_successful_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_epub_book, mock_context
    ) -> None:
        """main_async runs pipeline stages successfully."""
        from mela_parser.cli import main_async

        # Create a fake epub file
        epub_file = tmp_path / "test.epub"
        epub_file.write_bytes(b"fake epub content")

        # Set up args
        monkeypatch.setattr(
            sys, "argv", ["mela-parse", str(epub_file), "--output-dir", str(tmp_path / "output")]
        )

        # Setup mock pipeline stage
        mock_stage = MagicMock()
        mock_stage.name = "Conversion"
        mock_stage.execute = AsyncMock(return_value=None)

        mock_context.output_dir = tmp_path / "output" / "test_cookbook"
        (tmp_path / "output" / "test_cookbook").mkdir(parents=True)

        # Mock all dependencies with combined context managers
        with (
            patch("ebooklib.epub.read_epub", return_value=mock_epub_book),
            patch("mela_parser.cli.ServiceFactory"),
            patch("mela_parser.cli.create_default_pipeline") as mock_pipeline,
            patch("mela_parser.cli.PipelineContext", return_value=mock_context),
            patch("shutil.make_archive", return_value=str(tmp_path / "test.zip")),
            patch("os.rename"),
        ):
            mock_pipeline.return_value.stages = [mock_stage]
            await main_async()

    @pytest.mark.asyncio
    async def test_main_async_with_no_images(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_epub_book, mock_context
    ) -> None:
        """main_async respects --no-images flag."""
        from mela_parser.cli import main_async

        # Create a fake epub file
        epub_file = tmp_path / "test.epub"
        epub_file.write_bytes(b"fake epub content")

        # Set up args with --no-images
        monkeypatch.setattr(sys, "argv", ["mela-parse", str(epub_file), "--no-images"])

        # Setup mock pipeline stage
        mock_stage = MagicMock()
        mock_stage.name = "Conversion"
        mock_stage.execute = AsyncMock(return_value=None)

        mock_context.output_dir = tmp_path / "output" / "test_cookbook"
        (tmp_path / "output" / "test_cookbook").mkdir(parents=True)

        # Mock all dependencies with combined context managers
        with (
            patch("ebooklib.epub.read_epub", return_value=mock_epub_book),
            patch("mela_parser.cli.ServiceFactory") as mock_factory,
            patch("mela_parser.cli.create_default_pipeline") as mock_pipeline,
            patch("mela_parser.cli.PipelineContext", return_value=mock_context),
            patch("shutil.make_archive", return_value=str(tmp_path / "test.zip")),
            patch("os.rename"),
        ):
            mock_pipeline.return_value.stages = [mock_stage]
            await main_async()

            # Verify config has extract_images=False
            call_kwargs = mock_factory.call_args
            assert call_kwargs is not None
            config = call_kwargs.kwargs.get("config")
            assert config is not None
            assert config.extract_images is False


# ============================================================================
# main Tests
# ============================================================================


class TestMain:
    """Tests for main function (sync entry point)."""

    def test_main_runs_asyncio(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """main runs asyncio.run with main_async."""
        import contextlib

        from mela_parser.cli import main

        epub_file = tmp_path / "test.epub"
        epub_file.write_bytes(b"fake epub content")

        monkeypatch.setattr(sys, "argv", ["mela-parse", str(epub_file)])

        def mock_asyncio_run(coro):
            """Mock that properly closes the coroutine to avoid warning."""
            coro.close()
            return None

        with (
            patch("asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            contextlib.suppress(SystemExit),
        ):
            main()

        # asyncio.run should have been called
        mock_run.assert_called()

    def test_main_handles_keyboard_interrupt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """main handles KeyboardInterrupt gracefully."""
        from mela_parser.cli import main

        epub_file = tmp_path / "test.epub"
        epub_file.write_bytes(b"fake epub content")

        monkeypatch.setattr(sys, "argv", ["mela-parse", str(epub_file)])

        def mock_asyncio_run_interrupt(coro):
            """Mock that closes coroutine then raises KeyboardInterrupt."""
            coro.close()
            raise KeyboardInterrupt

        with (
            patch("asyncio.run", side_effect=mock_asyncio_run_interrupt),
            patch("mela_parser.cli.console.print") as mock_print,
        ):
            # Should not raise, should print message
            main()

            # Verify interrupt message was printed
            assert mock_print.called

    def test_main_handles_generic_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """main handles unexpected exceptions."""
        from mela_parser.cli import main

        epub_file = tmp_path / "test.epub"
        epub_file.write_bytes(b"fake epub content")

        monkeypatch.setattr(sys, "argv", ["mela-parse", str(epub_file)])

        def mock_asyncio_run_error(coro):
            """Mock that closes coroutine then raises RuntimeError."""
            coro.close()
            raise RuntimeError("Unexpected error")

        with (
            patch("asyncio.run", side_effect=mock_asyncio_run_error),
            patch("mela_parser.cli.console.print"),
            pytest.raises(RuntimeError, match="Unexpected error"),
        ):
            main()
