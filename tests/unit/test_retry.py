"""Unit tests for mela_parser.retry module.

Tests the async retry decorator and RetryConfig class.
"""

import asyncio

import pytest

from mela_parser.retry import RetryConfig, with_retry


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self) -> None:
        """RetryConfig has sensible defaults."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self) -> None:
        """RetryConfig accepts custom values."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0

    def test_to_kwargs(self) -> None:
        """to_kwargs returns dict suitable for with_retry decorator."""
        config = RetryConfig(max_attempts=5, initial_delay=2.0)
        kwargs = config.to_kwargs()

        assert kwargs == {
            "max_attempts": 5,
            "initial_delay": 2.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
        }


class TestWithRetryDecorator:
    """Tests for the with_retry async decorator."""

    async def test_success_no_retry(self) -> None:
        """Successful function does not retry."""
        call_count = 0

        @with_retry(max_attempts=3)
        async def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()
        assert result == "success"
        assert call_count == 1

    async def test_retries_on_exception(self) -> None:
        """Function retries on specified exception."""
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.01, retryable=(ValueError,))
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    async def test_raises_after_max_attempts(self) -> None:
        """Function raises after exhausting max attempts."""
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.01, retryable=(ValueError,))
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            await always_fails()

        assert call_count == 3

    async def test_does_not_retry_non_retryable(self) -> None:
        """Function does not retry on non-retryable exceptions."""
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.01, retryable=(ValueError,))
        async def raises_type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            await raises_type_error()

        assert call_count == 1  # Only one attempt

    async def test_exponential_backoff(self) -> None:
        """Delay increases exponentially between retries."""
        delays: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)
            # Don't actually sleep in tests

        # Temporarily replace asyncio.sleep
        asyncio.sleep = mock_sleep  # type: ignore[assignment]

        try:

            @with_retry(
                max_attempts=4,
                initial_delay=1.0,
                exponential_base=2.0,
                retryable=(ValueError,),
            )
            async def always_fails() -> str:
                raise ValueError("fail")

            with pytest.raises(ValueError, match="fail"):
                await always_fails()

            # Should have 3 delays (between 4 attempts)
            assert len(delays) == 3
            assert delays[0] == 1.0  # First delay
            assert delays[1] == 2.0  # 1.0 * 2
            assert delays[2] == 4.0  # 2.0 * 2

        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

    async def test_max_delay_cap(self) -> None:
        """Delay is capped at max_delay."""
        delays: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)

        asyncio.sleep = mock_sleep  # type: ignore[assignment]

        try:

            @with_retry(
                max_attempts=5,
                initial_delay=10.0,
                max_delay=15.0,
                exponential_base=2.0,
                retryable=(ValueError,),
            )
            async def always_fails() -> str:
                raise ValueError("fail")

            with pytest.raises(ValueError, match="fail"):
                await always_fails()

            # Delays should be: 10, 15 (capped), 15 (capped), 15 (capped)
            assert delays[0] == 10.0
            assert delays[1] == 15.0  # Capped at max
            assert delays[2] == 15.0  # Still capped
            assert delays[3] == 15.0  # Still capped

        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

    async def test_preserves_function_metadata(self) -> None:
        """Decorated function preserves original metadata."""

        @with_retry(max_attempts=3)
        async def documented_func() -> str:
            """This is a docstring."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    async def test_with_arguments(self) -> None:
        """Decorated function works with args and kwargs."""

        @with_retry(max_attempts=2, initial_delay=0.01, retryable=(ValueError,))
        async def func_with_args(a: int, b: str, c: bool = False) -> dict:
            return {"a": a, "b": b, "c": c}

        result = await func_with_args(1, "test", c=True)
        assert result == {"a": 1, "b": "test", "c": True}

    async def test_multiple_exception_types(self) -> None:
        """Can retry on multiple exception types."""
        call_count = 0

        @with_retry(
            max_attempts=4,
            initial_delay=0.01,
            retryable=(ValueError, TypeError, RuntimeError),
        )
        async def multi_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First")
            elif call_count == 2:
                raise TypeError("Second")
            elif call_count == 3:
                raise RuntimeError("Third")
            return "success"

        result = await multi_fail()
        assert result == "success"
        assert call_count == 4


class TestWithRetryConfigIntegration:
    """Tests combining RetryConfig with with_retry decorator."""

    async def test_config_to_kwargs_integration(self) -> None:
        """RetryConfig.to_kwargs works with with_retry decorator."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        call_count = 0

        # Pass arguments explicitly for type safety
        @with_retry(
            max_attempts=config.max_attempts,
            initial_delay=config.initial_delay,
            max_delay=config.max_delay,
            exponential_base=config.exponential_base,
            retryable=(ValueError,),
        )
        async def func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry")
            return "done"

        result = await func()
        assert result == "done"
        assert call_count == 2
