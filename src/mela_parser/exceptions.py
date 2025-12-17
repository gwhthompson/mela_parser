"""Custom exceptions for mela_parser.

This module defines a hierarchy of custom exceptions for better error handling
and more informative error messages throughout the recipe extraction pipeline.

The exception hierarchy follows best practices:
- Base exception for all mela_parser errors
- Specific exceptions for each phase of processing
- Rich context in error messages for debugging

Example:
    >>> try:
    ...     raise ExtractionError("Failed to extract recipe", chapter="Chapter 1")
    ... except MelaParserError as e:
    ...     print(f"Error in {e.context}: {e}")
"""


class MelaParserError(Exception):
    """Base exception for all mela_parser errors.

    All custom exceptions inherit from this base class, making it easy to
    catch any mela_parser-specific error.

    Attributes:
        message: Human-readable error description
        context: Additional context about where/when the error occurred
    """

    def __init__(self, message: str, **context: str | int | float | bool | None) -> None:
        """Initialize the exception with a message and optional context.

        Args:
            message: Human-readable error description
            **context: Additional context (e.g., chapter="Chapter 1", retry_count=3)
        """
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Format error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class EpubProcessingError(MelaParserError):
    """Error during EPUB file reading or parsing.

    Raised when:
    - EPUB file cannot be opened or read
    - EPUB structure is invalid or corrupted
    - Required EPUB metadata is missing
    - EPUB chapters cannot be accessed

    Example:
        >>> raise EpubProcessingError(
        ...     "Could not read EPUB file",
        ...     path="/path/to/book.epub",
        ...     reason="File not found"
        ... )
    """

    pass


class ConversionError(MelaParserError):
    """Error during HTML/EPUB to Markdown conversion.

    Raised when:
    - MarkItDown conversion fails
    - HTML content is malformed
    - Conversion produces empty or invalid output
    - Encoding issues prevent conversion

    Example:
        >>> raise ConversionError(
        ...     "Failed to convert chapter to markdown",
        ...     chapter="chapter01.xhtml",
        ...     error="Unsupported encoding"
        ... )
    """

    pass


class ExtractionError(MelaParserError):
    """Error during recipe extraction from content.

    Raised when:
    - LLM API call fails
    - Response parsing fails
    - Structured output is invalid
    - Extraction times out
    - Rate limits are exceeded

    Example:
        >>> raise ExtractionError(
        ...     "API rate limit exceeded",
        ...     chapter="Chapter 3",
        ...     retry_count=3,
        ...     model="gpt-5-nano"
        ... )
    """

    pass


class ValidationError(MelaParserError):
    """Error during recipe validation or quality checking.

    Raised when:
    - Recipe is missing required fields
    - Recipe fails quality checks
    - Recipe structure is invalid
    - Recipe content is incomplete

    Example:
        >>> raise ValidationError(
        ...     "Recipe missing required instructions",
        ...     recipe_title="Roasted Chicken",
        ...     missing_fields=["instructions"],
        ...     quality_score=0.3
        ... )
    """

    pass


class ImageProcessingError(MelaParserError):
    """Error during image extraction or processing.

    Raised when:
    - Image cannot be extracted from EPUB
    - Image format is unsupported
    - Image processing (resize, optimize) fails
    - AI image verification fails

    Example:
        >>> raise ImageProcessingError(
        ...     "Failed to resize image",
        ...     image_path="images/recipe_01.jpg",
        ...     target_size=(600, 800),
        ...     error="Unsupported format"
        ... )
    """

    pass


class ConfigurationError(MelaParserError):
    """Error in configuration or settings.

    Raised when:
    - Configuration file is invalid
    - Required settings are missing
    - Settings have invalid values
    - Environment variables are malformed

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid model name in configuration",
        ...     model="gpt-invalid",
        ...     valid_models=["gpt-5-nano", "gpt-5-mini"]
        ... )
    """

    pass


class DeduplicationError(MelaParserError):
    """Error during recipe deduplication.

    Raised when:
    - Duplicate detection fails
    - Similarity calculation errors
    - Merge operation fails

    Example:
        >>> raise DeduplicationError(
        ...     "Failed to calculate recipe similarity",
        ...     recipe1="Roasted Chicken",
        ...     recipe2="Roast Chicken",
        ...     error="Missing title field"
        ... )
    """

    pass


class RetryableError(MelaParserError):
    """Error that might succeed if retried.

    This exception indicates a transient failure that may resolve on retry,
    such as rate limits, temporary API unavailability, or network issues.

    Attributes:
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts allowed
        retry_after: Suggested delay before next retry (seconds)

    Example:
        >>> raise RetryableError(
        ...     "API rate limit exceeded",
        ...     attempt=2,
        ...     max_attempts=3,
        ...     retry_after=60.0
        ... )
    """

    def __init__(
        self,
        message: str,
        attempt: int = 1,
        max_attempts: int = 3,
        retry_after: float = 1.0,
        **context: str | int | float | bool | None,
    ) -> None:
        """Initialize retryable error with retry metadata.

        Args:
            message: Human-readable error description
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum number of attempts allowed
            retry_after: Suggested delay before next retry (seconds)
            **context: Additional context
        """
        super().__init__(message, **context)
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.retry_after = retry_after

    @property
    def should_retry(self) -> bool:
        """Check if another retry attempt should be made."""
        return self.attempt < self.max_attempts

    def with_next_attempt(self, backoff_multiplier: float = 2.0) -> "RetryableError":
        """Create a new error for the next retry attempt.

        Args:
            backoff_multiplier: Multiplier for retry_after delay

        Returns:
            New RetryableError with incremented attempt and backoff
        """
        return RetryableError(
            self.message,
            attempt=self.attempt + 1,
            max_attempts=self.max_attempts,
            retry_after=self.retry_after * backoff_multiplier,
            **self.context,
        )

    def __str__(self) -> str:
        """Format error with retry information."""
        base = super().__str__()
        return f"{base} [attempt {self.attempt}/{self.max_attempts}]"
