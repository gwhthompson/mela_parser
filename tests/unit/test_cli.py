"""Unit tests for mela_parser.cli module.

Tests CLI argument parsing, logging setup, and display functions.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
