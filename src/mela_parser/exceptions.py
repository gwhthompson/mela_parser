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
