"""Unit tests for mela_parser.exceptions module.

Tests the custom exception hierarchy, context formatting,
and RetryableError behavior.
"""

import pytest

from mela_parser.exceptions import (
    ConfigurationError,
    ConversionError,
    DeduplicationError,
    EpubProcessingError,
    ExtractionError,
    ImageProcessingError,
    MelaParserError,
    RetryableError,
    ValidationError,
)


class TestMelaParserError:
    """Tests for the base MelaParserError class."""

    def test_basic_message(self) -> None:
        """Error can be created with just a message."""
        error = MelaParserError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.context == {}

    def test_message_with_context(self) -> None:
        """Error includes context in string representation."""
        error = MelaParserError("Failed", chapter="Chapter 1", retry=3)
        result = str(error)
        assert "Failed" in result
        assert "chapter='Chapter 1'" in result
        assert "retry=3" in result

    def test_context_stored(self) -> None:
        """Context kwargs are stored as dict."""
        error = MelaParserError("Error", key1="value1", key2=42)
        assert error.context == {"key1": "value1", "key2": 42}

    def test_inherits_from_exception(self) -> None:
        """Base error inherits from Exception."""
        error = MelaParserError("Test")
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Tests that all exceptions inherit from MelaParserError."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            EpubProcessingError,
            ConversionError,
            ExtractionError,
            ValidationError,
            ImageProcessingError,
            ConfigurationError,
            DeduplicationError,
            RetryableError,
        ],
    )
    def test_inherits_from_base(self, exception_class: type) -> None:
        """All custom exceptions inherit from MelaParserError."""
        error = exception_class("Test error")
        assert isinstance(error, MelaParserError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize(
        "exception_class",
        [
            EpubProcessingError,
            ConversionError,
            ExtractionError,
            ValidationError,
            ImageProcessingError,
            ConfigurationError,
            DeduplicationError,
        ],
    )
    def test_accepts_context(self, exception_class: type) -> None:
        """All exceptions accept context kwargs."""
        error = exception_class("Error", custom_key="custom_value")
        assert error.context == {"custom_key": "custom_value"}


class TestRetryableError:
    """Tests for RetryableError with retry tracking."""

    def test_default_values(self) -> None:
        """RetryableError has sensible defaults."""
        error = RetryableError("Rate limited")
        assert error.attempt == 1
        assert error.max_attempts == 3
        assert error.retry_after == 1.0
        assert error.should_retry is True

    def test_custom_retry_params(self) -> None:
        """RetryableError accepts custom retry parameters."""
        error = RetryableError(
            "API error",
            attempt=2,
            max_attempts=5,
            retry_after=30.0,
        )
        assert error.attempt == 2
        assert error.max_attempts == 5
        assert error.retry_after == 30.0

    def test_should_retry_true(self) -> None:
        """should_retry returns True when attempts remain."""
        error = RetryableError("Error", attempt=1, max_attempts=3)
        assert error.should_retry is True

        error = RetryableError("Error", attempt=2, max_attempts=3)
        assert error.should_retry is True

    def test_should_retry_false(self) -> None:
        """should_retry returns False when max attempts reached."""
        error = RetryableError("Error", attempt=3, max_attempts=3)
        assert error.should_retry is False

        error = RetryableError("Error", attempt=5, max_attempts=3)
        assert error.should_retry is False

    def test_with_next_attempt_increments(self) -> None:
        """with_next_attempt creates new error with incremented attempt."""
        error1 = RetryableError("Error", attempt=1, max_attempts=3, retry_after=1.0)
        error2 = error1.with_next_attempt()

        assert error2.attempt == 2
        assert error2.max_attempts == 3
        assert error2.message == error1.message

    def test_with_next_attempt_applies_backoff(self) -> None:
        """with_next_attempt applies exponential backoff."""
        error1 = RetryableError("Error", retry_after=1.0)
        error2 = error1.with_next_attempt(backoff_multiplier=2.0)

        assert error2.retry_after == 2.0

        error3 = error2.with_next_attempt(backoff_multiplier=2.0)
        assert error3.retry_after == 4.0

    def test_with_next_attempt_preserves_context(self) -> None:
        """with_next_attempt preserves original context."""
        error1 = RetryableError("Error", chapter="Chapter 1", model="gpt-5")
        error2 = error1.with_next_attempt()

        assert error2.context == {"chapter": "Chapter 1", "model": "gpt-5"}

    def test_str_includes_attempt_info(self) -> None:
        """String representation includes attempt information."""
        error = RetryableError("API rate limited", attempt=2, max_attempts=5)
        result = str(error)

        assert "API rate limited" in result
        assert "[attempt 2/5]" in result

    def test_str_with_context_and_attempt(self) -> None:
        """String representation includes both context and attempt info."""
        error = RetryableError(
            "Error",
            attempt=1,
            max_attempts=3,
            chapter="Ch1",
        )
        result = str(error)

        assert "Error" in result
        assert "chapter='Ch1'" in result
        assert "[attempt 1/3]" in result


class TestExceptionUsagePatterns:
    """Tests demonstrating common usage patterns."""

    def test_catching_base_exception(self) -> None:
        """Can catch all mela_parser errors with base class."""

        def raise_error(error_type: str) -> None:
            if error_type == "epub":
                raise EpubProcessingError("EPUB error")
            elif error_type == "config":
                raise ConfigurationError("Config error")

        with pytest.raises(MelaParserError):
            raise_error("epub")

        with pytest.raises(MelaParserError):
            raise_error("config")

    def test_retry_loop_pattern(self) -> None:
        """Demonstrates retry loop pattern with RetryableError."""
        error = RetryableError("Transient error", max_attempts=3)
        attempts = 0

        while error.should_retry:
            attempts += 1
            error = error.with_next_attempt()

        assert attempts == 2  # Started at 1, retried twice to reach 3
