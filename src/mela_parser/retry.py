"""Async retry utilities with exponential backoff.

This module provides a reusable retry decorator for async functions,
implementing exponential backoff with configurable parameters.

Example:
    >>> from mela_parser.retry import with_retry
    >>> from openai import RateLimitError, APIError
    >>>
    >>> @with_retry(max_attempts=3, retryable=(RateLimitError, APIError))
    ... async def call_api():
    ...     return await client.chat.completions.create(...)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with exponential backoff retry.

    Wraps an async function to automatically retry on specified exceptions
    with exponential backoff between attempts.

    Args:
        max_attempts: Maximum number of attempts (including initial)
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        retryable: Tuple of exception types to retry on

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @with_retry(max_attempts=3, retryable=(ValueError,))
        ... async def flaky_operation():
        ...     # May fail sometimes
        ...     pass

    Note:
        The delay between attempts follows: delay = min(initial * base^attempt, max)
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for "
                            f"{func.__name__}: {e}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")

            # Should never reach here, but satisfy type checker
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry logic error: no exception captured")

        return wrapper

    return decorator


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff

    Example:
        >>> config = RetryConfig(max_attempts=5, initial_delay=0.5)
        >>> @with_retry(**config.to_kwargs())
        ... async def my_function():
        ...     pass
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ) -> None:
        """Initialize retry configuration."""
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def to_kwargs(self) -> dict[str, float | int]:
        """Convert to kwargs for with_retry decorator.

        Returns:
            Dictionary of retry parameters
        """
        return {
            "max_attempts": self.max_attempts,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
        }
